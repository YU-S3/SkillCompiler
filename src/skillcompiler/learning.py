"""Evidence-based extraction, program qualification, and bounded refinement."""

from copy import deepcopy
from .bank import Bank
from .contracts import digest, validate
from .tools import ToolContext
from .model import ResponseContractError
from . import prompts

PROPOSAL_SCHEMA = {
    "type": "object",
    "properties": {
        "decision": {"enum": ["no_change", "propose"]},
        "skills": {"type": "array", "items": {"type": "object"}},
        "programs": {"type": "array", "items": {"type": "object"}},
        "skeletons": {"type": "array", "items": {"type": "object"}},
        "remove_ids": {"type": "array", "items": {"type": "string"}},
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
        material = {
            "bank": bank.data(),
            "episodes": episodes,
            "asset_contract": {
                "skill": "id, name, applicability, guidance, input_schema, output_schema, program_ids",
                "program": "id, name, input_schema, output_schema, allowed_tools, trials",
                "trial": "episode, inputs, prefix (event indices), expected_from (event index and output path)",
                "skeleton": "id, name, nodes ordered by after dependencies; node args use task/from/literal references",
                "evidence": "episode ID and exact event indices; all proposed assets must be grounded",
                "limits": {"max_assets": self.max_proposals},
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
                return self._apply(bank, proposal, episodes, suite, key + "/repaired")
            except (ValueError, KeyError, TypeError, SyntaxError) as second:
                return bank, {"decision": "rejected", "error": str(second)}
        try:
            return self._apply(bank, proposal, episodes, suite, key)
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
                return self._apply(bank, repaired, episodes, suite, key + "/repaired")
            except (ValueError, KeyError, TypeError, SyntaxError) as second:
                return bank, {"decision": "rejected", "error": str(second)}

    def _apply(self, bank, proposal, episodes, suite, key):
        if proposal["decision"] == "no_change":
            return bank, {"decision": "no_change"}
        evidence_ok(proposal, episodes)
        if (
            sum(len(proposal.get(k, [])) for k in ["skills", "programs", "skeletons"])
            > self.max_proposals
        ):
            raise ValueError("Proposal capacity exceeded")
        data = bank.data()
        indexes = {k: {a["id"]: a for a in v} for k, v in data.items()}
        for asset_id in proposal.get("remove_ids", []):
            matches = [v for v in indexes.values() if asset_id in v]
            if not matches:
                raise ValueError("Cannot remove unknown asset")
            for v in matches:
                del v[asset_id]
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
            indexes["programs"][program["id"]] = program
        for kind in ["skills", "skeletons"]:
            for asset in proposal.get(kind, []):
                indexes[kind][asset["id"]] = deepcopy(asset)
        candidate = Bank(**{k: list(v.values()) for k, v in indexes.items()})
        return candidate, {
            "decision": "proposed",
            "before": bank.hash,
            "after": candidate.hash,
            "evidence": proposal["evidence"],
            "qualification": qualification,
        }

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
            result = self.worker.execute(program, trial["inputs"], ctx)
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
