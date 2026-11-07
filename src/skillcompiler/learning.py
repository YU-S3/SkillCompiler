"""Evidence-based extraction, program qualification, and bounded refinement."""

from copy import deepcopy
from .bank import Bank
from .contracts import digest, validate
from .tools import ToolContext
from .model import BudgetExceeded, ResponseContractError
from . import prompts
from .alignment import integrate
from .qualification import certificate, program_digest
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
# Eq. 3 statistics are measured by the Host, never proposed by a model.
HOST_SKILL_FIELDS = ("usage", "utility", "active")


class ReplayFailure(ValueError):
    """A candidate Program ran but did not reproduce the observed reference."""

    def __init__(self, message, detail):
        super().__init__(message)
        self.detail = detail


def usage_statistics(episodes):
    """Local operation outcomes per Skill (App. C), not task success."""
    usage = {}

    def count(skill_id, ok):
        record = usage.setdefault(skill_id, {"attempts": 0, "successes": 0})
        record["attempts"] += 1
        record["successes"] += int(bool(ok))

    for episode in episodes:
        for event in episode["events"]:
            if event.get("type") == "skill":
                status = event.get("status", event.get("result", {}).get("status"))
                count(event["id"], status == "ok")
            elif event.get("type") in {
                "skeleton_node",
                "skeleton_error",
            } and event.get("guidance_skill_id"):
                count(event["guidance_skill_id"], event["type"] == "skeleton_node")
    return usage


def with_usage(bank, episodes):
    """Recompute u(a) and the active flag deterministically from all episodes."""
    data = bank.data()
    usage = usage_statistics(episodes)
    for skill in data["skills"]:
        for field in HOST_SKILL_FIELDS:
            skill.pop(field, None)
        record = usage.get(skill["id"])
        if record:
            skill["usage"] = record
            skill["utility"] = round(record["successes"] / record["attempts"], 6)
            skill["active"] = not (record["attempts"] >= 2 and not record["successes"])
    return Bank(**data)


def localize_failure(episode):
    """First failing operation in an episode, from observable trace events only."""
    events = episode["events"]
    skeleton = next(
        (e["skeleton"] for e in events if e.get("type") == "skeleton_construction"),
        {},
    )
    nodes = {n["id"]: n for n in skeleton.get("nodes", [])}
    base = {"episode": episode["task"]["id"]}
    for index, event in enumerate(events):
        kind = event.get("type")
        if kind == "skeleton_error":
            node = nodes.get(event["id"])
            return {
                **base,
                "event": index,
                "level": "skeleton_node" if node else "skeleton_goal",
                "operation": event["id"],
                "error": event["error"],
                "binding": {
                    k: node[k]
                    for k in (
                        "skill_id",
                        "program_id",
                        "preferred_program_id",
                        "execution_mode",
                    )
                    if node and k in node
                },
                "contract": deepcopy(node.get("contract")) if node else None,
            }
        if kind == "program_error" or (
            kind == "skill" and event.get("status") == "failed"
        ):
            return {
                **base,
                "event": index,
                "level": "program" if kind == "program_error" else "skill",
                "operation": event["id"],
                "error": event.get(
                    "error",
                    "status " + str(event.get("result", {}).get("status")),
                ),
                "binding": {"program_id": event.get("program_id", event["id"])},
            }
        if kind == "runtime_validation" and not event.get("passed"):
            return {
                **base,
                "event": index,
                "level": "task_local_program",
                "operation": event["program_id"],
                "error": event["error"],
            }
        if kind == "operation_error":
            return {
                **base,
                "event": index,
                "level": "operation",
                "operation": event["action"].get("id", event["action"]["action"]),
                "error": event["error"],
            }
    feedback = episode.get("feedback")
    if feedback and not feedback.get("success"):
        return {
            **base,
            "event": None,
            "level": "task_outcome",
            "operation": None,
            "error": "No local operation failed; task outcome failed with terminal_reason "
            + str(episode.get("terminal_reason")),
        }
    return None


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
    def __init__(
        self,
        model,
        worker=None,
        max_proposals=8,
        repair_limit=1,
        program_repair_limit=1,
    ):
        if repair_limit not in (0, 1):
            raise ValueError("Structural repair is bounded to one")
        if program_repair_limit not in range(4):
            raise ValueError("Replay repair is bounded to three")
        self.model, self.worker = model, worker
        self.max_proposals = max_proposals
        self.repair_limit = repair_limit
        self.program_repair_limit = program_repair_limit

    def update(self, bank, episodes, suite, key, refine=False):
        candidate, receipt = self._update(bank, episodes, suite, key, refine)
        # Utility is refreshed after every update, including no_change/rejected.
        candidate = with_usage(candidate, episodes)
        receipt["usage"] = usage_statistics(episodes)
        if receipt.get("decision") == "proposed":
            receipt["after"] = candidate.hash
        return candidate, receipt

    def _update(self, bank, episodes, suite, key, refine=False):
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
            "failure_localization": [
                f for f in (localize_failure(e) for e in episodes) if f
            ],
            "asset_contract": {
                "skill": "id, name, applicability, guidance, input_schema, output_schema, preconditions, effects, validator, program_ids or implementations. usage/utility/active are Host-measured and ignored if proposed.",
                "program": "id, name, input_schema, output_schema, allowed_tools, trials",
                "trial": "episode, inputs, prefix (event indices), expected_from (event index and output path)",
                "skeleton": "id, name, complete goal contract (input_schema, output_schema, preconditions, effects); nodes ordered by after dependencies with operation contract, args using task/from/literal references, and execution_mode programmatic (with allowed_tools) or dynamic; outputs bind final roles. Every goal effect must be covered, every prerequisite established by dependencies. Legacy skeletons without a complete contract are not eligible for verified online reuse.",
                "evidence": "episode ID and exact event indices; all proposed assets must be grounded",
                "semantic_contract": "Predicates have predicate, args of typed {kind: skill_input|skill_output, source_role}, effect_domain, cardinality, distinct_by. Preserve literals, types, resolution constraints and validator semantics.",
                "implementation": "program_id; input_map maps Program inputs to Skill inputs; output_map maps Skill outputs to Program outputs",
                "merge": "merges: kind, source_ids, new target_id. Include replacement asset. Sources must have equivalent contracts, two independent training episodes, and executable source replay. No unqualified removal.",
                "failure_localization": "Host-localized first failing operation per episode; revise that operation's preconditions/bindings/guidance, or repair its Program through the replay-qualified revision path.",
                "limits": {
                    "max_assets": self.max_proposals,
                    "program_synthesis_enabled": self.worker is not None,
                    "program_replay_repairs": self.program_repair_limit,
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
        for skill in additions["skills"]:
            for field in HOST_SKILL_FIELDS:
                skill.pop(field, None)
                prior = bank.skills.get(skill.get("id"), {})
                if field in prior:
                    # Carried unchanged; recomputed from all episodes after the update.
                    skill[field] = deepcopy(prior[field])
        additions["programs"] = []
        qualification = []
        certified = {}
        tools = None
        for spec in proposal.get("programs", []):
            if self.worker is None:
                raise ValueError("Program synthesis requires an isolated worker")
            if not spec.get("trials"):
                raise ValueError("Program needs observed replay cases")
            tools = tools or self._tools(suite, episodes)
            program, receipts = self._build(bank, spec, episodes, suite, tools, key)
            qualification.extend(receipts)
            certified.setdefault(program_digest(program), []).extend(receipts)
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
                            e.get("type") == "skill"
                            and e.get("id") == sid
                            and e.get("status", "ok") == "ok"
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
                        cases = self._regression(program, pid, episodes, suite)
                        qualification.extend(cases)
                        certified.setdefault(program_digest(program), []).extend(cases)
                elif merge["kind"] == "programs":
                    program = candidate.programs[merge["target_id"]]
                    cases = self._regression(program, sid, episodes, suite)
                    qualification.extend(cases)
                    certified.setdefault(program_digest(program), []).extend(cases)
                else:
                    if not any(
                        e.get("id") == sid and e.get("type") == "skeleton_output"
                        for episode in episodes
                        for e in episode["events"]
                    ):
                        raise ValueError(
                            "Skeleton merge lacks executed source evidence"
                        )
        if certified:
            # Eq. 4 Cert: bind passing replay receipts to the exact admitted body.
            tools = tools or self._tools(suite, episodes)
            for program in candidate.programs.values():
                receipts = certified.get(program_digest(program))
                if receipts:
                    program["qualification"] = certificate(program, tools, receipts)
            candidate.validate()
        return candidate, {
            "decision": "proposed",
            "before": bank.hash,
            "after": candidate.hash,
            "evidence": proposal["evidence"],
            "qualification": qualification,
            "alignment": alignment,
        }

    def _build(self, bank, spec, episodes, suite, tools, key):
        """PROGRAMMER build, replay qualification and bounded replay repair."""
        material = {"operation": spec, "episodes": episodes, "tools": tools}
        request = key + "/build_" + spec["id"]
        source = self.model.ask(request, prompts.PROGRAMMER, material, BUILD_SCHEMA)[
            "source"
        ]
        for attempt in range(self.program_repair_limit + 1):
            program = {
                k: v for k, v in spec.items() if k not in {"trials", "qualification"}
            }
            program.update(format="python", source=source)
            try:
                Bank(programs=[program])
            except (ValueError, SyntaxError) as exc:
                failure = {"stage": "structure", "error": str(exc)}
            else:
                try:
                    receipts = [
                        self._trial(program, trial, episodes, suite)
                        for trial in spec["trials"]
                    ]
                    prior = bank.programs.get(program["id"])
                    if prior and program_digest(prior) != program_digest(program):
                        receipts += self._regression(
                            program, program["id"], episodes, suite
                        )
                    return program, receipts
                except ReplayFailure as exc:
                    failure = {"stage": "replay", **exc.detail}
            if attempt == self.program_repair_limit:
                raise ValueError(
                    "Program was not qualified after "
                    + str(attempt)
                    + " replay repairs: "
                    + failure["error"]
                )
            # Discrepancies go back to the Programmer (Sec. 3.2.4).
            source = self.model.ask(
                request + "/repair_" + str(attempt + 1),
                prompts.PROGRAMMER,
                {**material, "previous_source": source, "replay_failure": failure},
                BUILD_SCHEMA,
            )["source"]

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

            detail = {
                "episode": trial["episode"],
                "inputs": deepcopy(trial["inputs"]),
                "expected": deepcopy(expected),
            }
            try:
                Runtime._preflight(program, trial["inputs"], ctx)
                if program.get("format") == "ir":
                    from .ir import execute

                    result = execute(program, trial["inputs"], ctx)
                else:
                    if self.worker is None:
                        raise ValueError("Replay requires an isolated worker")
                    result = self.worker.execute(program, trial["inputs"], ctx)
                Runtime._effects(program, trial["inputs"], result, ctx)
            except BudgetExceeded:
                raise
            except (ValueError, KeyError, RuntimeError, TimeoutError) as exc:
                # Worker/program errors are replay discrepancies, not crashes.
                raise ReplayFailure(
                    "Program replay failed: " + str(exc),
                    {**detail, "error": "Program replay failed: " + str(exc)},
                ) from exc
            if result["status"] != "ok" or result["outputs"] != expected:
                raise ReplayFailure(
                    "Program replay differs from observed training output",
                    {
                        **detail,
                        "observed": deepcopy(result),
                        "error": "Program replay differs from observed training output",
                    },
                )
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
