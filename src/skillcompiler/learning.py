"""Evidence-based extraction, program qualification, and bounded refinement."""

from copy import deepcopy
from .bank import Bank
from .contracts import digest, validate
from .tools import ToolContext
from .model import ResponseContractError
from . import prompts
from .alignment import integrate
from .schemas import PREDICATE, TRIAL

PROPOSAL_SCHEMA = {
    "type": "object",
    "properties": {
        "decision": {"enum": ["no_change", "propose"]},
        "skills": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "effects": {"type": "array", "items": PREDICATE},
                    "preconditions": {"type": "array", "items": PREDICATE},
                },
            },
        },
        "programs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"trials": {"type": "array", "items": TRIAL}},
            },
        },
        "skeletons": {"type": "array", "items": {"type": "object"}},
        "remove_ids": {"type": "array", "items": {"type": "string"}},
        "merges": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "kind": {"enum": ["skills", "programs", "skeletons"]},
                    "source_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 2,
                        "uniqueItems": True,
                    },
                    "target_id": {"type": "string"},
                },
                "required": ["kind", "source_ids", "target_id"],
                "additionalProperties": False,
            },
        },
        "evidence": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "episode": {"type": "string"},
                    "events": {
                        "type": "array",
                        "items": {"type": "integer", "minimum": 0},
                    },
                },
                "required": ["episode", "events"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["decision"],
    "additionalProperties": False,
}
BUILD_SCHEMA = {
    "type": "object",
    "properties": {"source": {"type": "string", "minLength": 1}},
    "required": ["source"],
    "additionalProperties": False,
}


def evidence_ok(proposal, episodes):
    available = {e["task"]["id"]: e for e in episodes}
    refs = proposal.get("evidence", [])
    if not refs:
        raise ValueError("An update requires observed training evidence")
    for ref in refs:
        if ref["episode"] not in available:
            raise ValueError("Unknown training episode")
        if not ref["events"] or any(
            i < 0 or i >= len(available[ref["episode"]]["events"])
            for i in ref["events"]
        ):
            raise ValueError("Invalid evidence event index")


class Learner:
    def __init__(self, model, worker=None, max_proposals=8, repair_limit=1):
        if repair_limit not in (0, 1):
            raise ValueError("Structural repair is bounded to one")
        self.model, self.worker = model, worker
        self.max_proposals = max_proposals
        self.repair_limit = repair_limit

    def update(self, bank, episodes, suite, key, refine=False):
        evidence_catalog = []
        for episode in episodes:
            events = []
            for index, event in enumerate(episode["events"]):
                paths = []
                if event["type"] == "tool" and isinstance(event.get("result"), dict):
                    paths.append(["result"])
                if event["type"] == "program" and isinstance(
                    event.get("result", {}).get("outputs"), dict
                ):
                    paths.append(["result", "outputs"])
                if event["type"] == "skeleton_node" and isinstance(
                    event.get("outputs"), dict
                ):
                    paths.append(["outputs"])
                events.append(
                    {
                        "index": index,
                        "type": event["type"],
                        "output_object_paths": paths,
                    }
                )
            evidence_catalog.append(
                {"episode": episode["task"]["id"], "events": events}
            )
        material = {
            "bank": bank.data(),
            "episodes": episodes,
            "evidence_catalog": evidence_catalog,
            "asset_contract": {
                "skill": "id, name, applicability, guidance, input_schema, output_schema, preconditions, effects, validator, program_ids or implementations",
                "program": "id, name, input_schema, output_schema, allowed_tools, trials",
                "trial": "episode, inputs, prefix (event indices), expected_from (event index and output path)",
                "skeleton": "id, name, complete goal contract (input_schema, output_schema, preconditions, effects); nodes ordered by after dependencies with operation contract, args using task/from/literal references; outputs bind final roles. Every goal effect must be covered, every prerequisite established by dependencies. Legacy skeletons without a complete contract are not eligible for verified online reuse.",
                "evidence": "episode ID and exact event indices; all proposed assets must be grounded",
                "semantic_contract": "Predicates have predicate, args of typed {kind: skill_input|skill_output, source_role}, effect_domain, cardinality, distinct_by. Preserve literals, types, resolution constraints and validator semantics.",
                "implementation": "program_id; input_map maps Program inputs to Skill inputs; output_map maps Skill outputs to Program outputs",
                "merge": "merges: kind, source_ids, new target_id. Include replacement asset. Sources must have equivalent contracts, two independent training episodes, and executable source replay. No unqualified removal.",
                "limits": {
                    "max_assets": self.max_proposals,
                    "program_synthesis_enabled": self.worker is not None,
                },
            },
        }
        prompt = prompts.REFINER if refine else prompts.EXTRACTOR
        try:
            proposal = self.model.ask(
                key + "/extract", prompt, material, PROPOSAL_SCHEMA
            )
        except ResponseContractError as exc:
            if not self.repair_limit:
                return bank, {"decision": "rejected", "error": str(exc)}
            try:
                proposal = self.model.ask(
                    key + "/repair",
                    prompts.REPAIR,
                    {
                        "original": material,
                        "previous_response": exc.response,
                        "errors": str(exc),
                    },
                    PROPOSAL_SCHEMA,
                )
                return self._apply(
                    bank, proposal, episodes, suite, key + "/repaired", refine
                )
            except (ValueError, KeyError, TypeError, SyntaxError) as second:
                return bank, {"decision": "rejected", "error": str(second)}
        try:
            return self._apply(bank, proposal, episodes, suite, key, refine)
        except (ValueError, KeyError, TypeError, SyntaxError) as exc:
            if not self.repair_limit:
                return bank, {"decision": "rejected", "error": str(exc)}
            try:
                repaired = self.model.ask(
                    key + "/repair",
                    prompts.REPAIR,
                    {"original": material, "proposal": proposal, "errors": str(exc)},
                    PROPOSAL_SCHEMA,
                )
                return self._apply(
                    bank, repaired, episodes, suite, key + "/repaired", refine
                )
            except (ValueError, KeyError, TypeError, SyntaxError) as second:
                return bank, {"decision": "rejected", "error": str(second)}

    def _apply(self, bank, proposal, episodes, suite, key, refine=False):
        if proposal["decision"] == "no_change":
            return bank, {"decision": "no_change"}
        evidence_ok(proposal, episodes)
        if (
            sum(len(proposal.get(k, [])) for k in ["skills", "programs", "skeletons"])
            > self.max_proposals
        ):
            raise ValueError("Proposal capacity exceeded")
        additions = {k: deepcopy(proposal.get(k, [])) for k in ("skills", "skeletons")}
        additions["programs"] = []
        qualification = []
        for spec in proposal.get("programs", []):
            if self.worker is None:
                raise ValueError("Program synthesis requires an isolated worker")
            if not spec.get("trials"):
                raise ValueError("Program needs observed replay cases")
            source = self.model.ask(
                key + "/build_" + spec["id"],
                prompts.PROGRAMMER,
                {
                    "operation": spec,
                    "episodes": episodes,
                    "tools": self._tools(suite, episodes),
                },
                BUILD_SCHEMA,
            )["source"]
            program = {k: v for k, v in spec.items() if k != "trials"}
            program.update(format="python", source=source)
            Bank(programs=[program])
            for trial in spec["trials"]:
                qualification.append(self._trial(program, trial, episodes, suite))
            if program["id"] in bank.programs and digest(program) != digest(
                bank.programs[program["id"]]
            ):
                qualification.extend(
                    self._regression(program, program["id"], episodes, suite)
                )
            additions["programs"].append(program)
        candidate, alignment = integrate(
            bank,
            additions,
            proposal.get("merges", []),
            proposal.get("remove_ids", []),
            refine=refine,
            evidence=proposal["evidence"],
        )
        from .semantic import implementations, executable_equal

        for merge in proposal.get("merges", []):
            for sid in merge["source_ids"]:
                if merge["kind"] == "skills":
                    if not implementations(bank.skills[sid]) and not any(
                        sid in episode.get("retrieved_skill_ids", [])
                        or any(
                            e.get("type") == "skill" and e.get("id") == sid
                            for e in episode["events"]
                        )
                        for episode in episodes
                        if episode["task"]["id"]
                        in {r["episode"] for r in proposal["evidence"]}
                    ):
                        raise ValueError("Skill merge lacks executed source evidence")
                    for binding in implementations(bank.skills[sid]):
                        pid = binding["program_id"]
                        program = candidate.programs.get(pid) or next(
                            p
                            for p in candidate.programs.values()
                            if executable_equal(bank.programs[pid], p)
                        )
                        qualification.extend(
                            self._regression(program, pid, episodes, suite)
                        )
                elif merge["kind"] == "programs":
                    qualification.extend(
                        self._regression(
                            candidate.programs[merge["target_id"]], sid, episodes, suite
                        )
                    )
                else:
                    if not any(
                        e.get("id") == sid and e.get("type") == "skeleton_output"
                        for episode in episodes
                        for e in episode["events"]
                    ):
                        raise ValueError(
                            "Skeleton merge lacks executed source evidence"
                        )
        return candidate, {
            "decision": "proposed",
            "before": bank.hash,
            "after": candidate.hash,
            "evidence": proposal["evidence"],
            "qualification": qualification,
            "alignment": alignment,
        }

    def _regression(self, program, source_id, episodes, suite):
        cases = []
        for episode in episodes:
            for i, event in enumerate(episode["events"]):
                if (
                    event.get("type") != "program"
                    or event.get("id") != source_id
                    or event.get("result", {}).get("status") != "ok"
                ):
                    continue
                if "tool_event_start" not in event:
                    raise ValueError("Source replay lacks an exact execution boundary")
                start = event["tool_event_start"]
                cases.append(
                    self._trial(
                        program,
                        {
                            "episode": episode["task"]["id"],
                            "inputs": event["arguments"],
                            "prefix": [
                                j
                                for j in range(start)
                                if episode["events"][j]["type"] == "tool"
                            ],
                            "expected_from": {
                                "event": i,
                                "path": ["result", "outputs"],
                            },
                        },
                        episodes,
                        suite,
                    )
                )
        if not cases:
            raise ValueError("Replacement/merge lacks observed source Program cases")
        return cases

    @staticmethod
    def _tools(suite, episodes):
        from .contracts import Task

        environment = suite.open(Task(**episodes[0]["task"]))
        try:
            return environment.tools()
        finally:
            environment.close()

    def _trial(self, program, trial, episodes, suite):
        from .contracts import Task

        episode = next(
            (e for e in episodes if e["task"]["id"] == trial["episode"]), None
        )
        if episode is None:
            raise ValueError("Trial uses an unknown source")
        ref = trial["expected_from"]
        if not 0 <= ref["event"] < len(episode["events"]):
            raise ValueError("Invalid trial event")
        event = episode["events"][ref["event"]]
        if event["type"] not in ("tool", "program", "skeleton_node"):
            raise ValueError("Expected output is not an executed observation")
        expected = event
        for component in ref["path"]:
            expected = expected[component]
        if not isinstance(expected, dict):
            raise ValueError("Trial reference must point to an observed output object")
        # The Host supplies the expected value from a real trace, never from a model assertion.
        environment = suite.open(Task(**episode["task"]))
        ctx = ToolContext(environment)
        try:
            last = -1
            for index in trial.get("prefix", []):
                if index <= last or index >= ref["event"]:
                    raise ValueError("Replay prefix must precede its observed output")
                step = episode["events"][index]
                last = index
                if step["type"] != "tool":
                    raise ValueError(
                        "Only executed native calls can form a replay prefix"
                    )
                ctx.call(step["name"], step["arguments"])
            from .runtime import Runtime

            Runtime._preflight(program, trial["inputs"], ctx)
            if program.get("format") == "ir":
                from .ir import execute

                result = execute(program, trial["inputs"], ctx)
            else:
                if self.worker is None:
                    raise ValueError("Replay requires an isolated worker")
                result = self.worker.execute(program, trial["inputs"], ctx)
            Runtime._effects(program, trial["inputs"], result, ctx)
            if result["status"] != "ok" or result["outputs"] != expected:
                raise ValueError("Program replay differs from observed training output")
            return {
                "program_id": program["id"],
                "episode": trial["episode"],
                "inputs_hash": digest(trial["inputs"]),
                "reference_hash": digest(expected),
                "result_hash": digest(result),
                "passed": True,
            }
        finally:
            environment.close()
