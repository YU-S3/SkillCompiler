"""Default online planning path; legacy component fixtures explicitly opt out."""

from copy import deepcopy
import os
import pytest
from skillcompiler.bank import Bank
from skillcompiler.contracts import Task, validate, read
from skillcompiler.planning import validate_analysis, skeleton_analysis
from skillcompiler.semantic import match_requirement, align, contract
from skillcompiler.runtime import Runtime
from skillcompiler.model import BudgetExceeded, ResponseContractError
from skillcompiler.learning import Learner
from skillcompiler.pipeline import Pipeline
from skillcompiler.worker import ProgramWorker
from examples.key_value import Suite, Environment
from examples.offline_demo import (
    DemoModel,
    lookup_analysis,
    lookup_contract,
    object_schema,
)


def asset(sid="lookup", i="key", o="value"):
    return {
        "id": sid,
        "name": "Read",
        "guidance": "Use the observed value only.",
        **lookup_contract(i, o),
    }


def skeleton():
    a = lookup_analysis()
    return {
        "id": "complete",
        "name": "Read",
        "contract": a["contract"],
        "nodes": [
            {
                "id": "read",
                "skill_id": "lookup",
                "contract": lookup_contract(),
                "args": {"key": {"task": "key"}},
                "after": [],
            }
        ],
        "outputs": a["outputs"],
    }


def test_directional_matching_is_not_merge_identity():
    required, offered = asset(), asset("better", "query", "result")
    offered["effects"][0]["cardinality"] = 2
    match = match_requirement(required, offered)
    assert match["status"] == "exact"
    assert match["proof"]["input_map"] == {"key": "query"}
    assert match["proof"]["output_map"] == {"value": "result"}
    assert align(required, offered)["status"] == "different"
    offered["preconditions"] = deepcopy(offered["effects"])
    assert match_requirement(required, offered)["status"] == "different"
    assert match_requirement(required, required, max_states=1)["status"] == "unknown"


@pytest.mark.parametrize(
    "mutation,message",
    [
        (
            lambda a: a["contract"]["effects"][0].update(predicate="uncovered"),
            "complete task goal",
        ),
        (lambda a: a["operations"][0]["after"].append("read"), "dependency order"),
        (
            lambda a: a["operations"][0]["args"].update(key={"task": "hidden_gold"}),
            "Unknown referenced role",
        ),
        (
            lambda a: a["operations"][0]["contract"]["input_schema"]["properties"][
                "key"
            ].update(type="integer"),
            "incompatible types",
        ),
        (
            lambda a: a["operations"][0]["contract"].update(
                preconditions=[{"predicate": "ready"}]
            ),
            "prerequisites",
        ),
        (lambda a: a["outputs"].update(answer={"literal": 42}), "complete task goal"),
        (
            lambda a: a["contract"]["effects"][0].update(effect_domain="world"),
            "complete task goal",
        ),
        (
            lambda a: a["contract"]["effects"][0].update(cardinality=2),
            "complete task goal",
        ),
    ],
)
def test_incomplete_or_invalid_analysis_rejected(mutation, message):
    a = lookup_analysis()
    mutation(a)
    with pytest.raises(ValueError, match=message):
        validate_analysis(a, {"key": "item_0"})


def test_required_inputs_cannot_be_invented():
    with pytest.raises(ValueError):
        validate_analysis(lookup_analysis(), {})


def test_default_planner_constructs_and_rebinds_real_program():
    s = asset("opaque", "query", "result")
    s["program_ids"] = ["read_program"]
    p = {
        "id": "read_program",
        "name": "Read",
        "format": "ir",
        "allowed_tools": ["lookup"],
        "input_schema": object_schema("query", "string"),
        "output_schema": object_schema("result", "integer"),
        "instructions": [
            {
                "op": "ACTION",
                "action_type": "lookup",
                "argument_mapping": {"key": {"source": "input", "field": "query"}},
                "result_variable": "value",
            },
            {
                "op": "RETURN",
                "output_sources": {
                    "result": {"source": "local", "variable": "value", "field": "value"}
                },
            },
        ],
    }
    # Deliberately lexical-near but semantically wrong: must not enter compatible catalog.
    wrong = asset("distractor")
    wrong["name"] = "Read the requested key"
    wrong["effects"][0]["predicate"] = "different"
    bank = Bank([s, wrong], [p])
    before = bank.hash

    class Model(DemoModel):
        def ask(self, key, prompt, material, schema):
            if key.endswith("/construct"):
                assert [s["id"] for s in material["compatible_operations"]["read"]] == [
                    "opaque"
                ]
            return super().ask(key, prompt, material, schema)

    suite = Suite()
    t = suite.tasks("test")[0]
    trace = Runtime(Model()).run(t, suite.open(t), bank, "test/0")
    assert trace["answer"] == 3 and bank.hash == before
    built = next(e for e in trace["events"] if e["type"] == "skeleton_construction")
    assert built["skeleton"]["nodes"][0]["args"] == {"query": {"task": "key"}}
    assert built["skeleton"]["outputs"]["answer"]["field"] == "result"
    assert any(e["type"] == "program" for e in trace["events"])


def test_complete_skeleton_reuse_and_legacy_exclusion():
    complete = skeleton()
    legacy = deepcopy(complete)
    legacy.pop("contract")
    legacy["id"] = "legacy"
    bank = Bank([asset()], skeletons=[complete, legacy])

    class Model(DemoModel):
        def ask(self, key, prompt, material, schema):
            if key.endswith("/construct"):
                assert [s["id"] for s in material["compatible_skeletons"]] == [
                    "complete"
                ]
                return {"skeleton_id": "complete", "choices": {"read": "lookup"}}
            return super().ask(key, prompt, material, schema)

    suite = Suite()
    t = suite.tasks("test")[0]
    trace = Runtime(Model()).run(t, suite.open(t), bank, "test/0")
    assert trace["answer"] == 3
    assert (
        next(e for e in trace["events"] if e["type"] == "skeleton_construction")["mode"]
        == "reuse"
    )


def test_persistent_skeleton_must_cover_goal_and_keep_operation_contract():
    s = skeleton()
    s["contract"]["effects"][0]["predicate"] = "other"
    with pytest.raises(ValueError, match="complete task goal"):
        Bank([asset()], skeletons=[s])
    s = skeleton()
    s["nodes"][0]["contract"]["effects"][0]["predicate"] = "other"
    with pytest.raises(ValueError, match="no longer meets"):
        Bank([asset()], skeletons=[s])


def test_planning_repair_fallback_is_bounded_and_budget_propagates():
    class Model(DemoModel):
        def __init__(self):
            self.repairs = 0

        def ask(self, key, prompt, material, schema):
            if key.endswith(("/analyze", "/analysis_repair")):
                self.repairs += 1
                return {"decision": "plan"}
            return super().ask(key, prompt, material, schema)

    suite = Suite()
    t = suite.tasks("test")[0]
    m = Model()
    trace = Runtime(m).run(t, suite.open(t), Bank(), "test/0")
    assert trace["answer"] == 3 and m.repairs == 2
    assert any(e["type"] == "planning_rejected" for e in trace["events"])

    class Limited:
        def ask(self, *args):
            raise BudgetExceeded("no allowance")

    with pytest.raises(BudgetExceeded):
        Runtime(Limited()).run(t, suite.open(t), Bank(), "test/0")


def test_unverified_effect_cannot_finish_skeleton():
    class Unverified(Environment):
        def verify_effects(self, *args):
            return False

    t = Suite().tasks("test")[0]
    trace = Runtime(DemoModel()).run(t, Unverified(t), Bank(), "test/0")
    assert any(e["type"] == "skeleton_error" for e in trace["events"])
    assert not any(e["type"] == "skeleton_output" for e in trace["events"])


def test_default_full_pipeline_isolation_freeze_and_idempotence(tmp_path):
    class Counted(DemoModel):
        def __init__(self):
            self.calls = []

        def ask(self, key, prompt, material, schema):
            self.calls.append(key)
            value = super().ask(key, prompt, material, schema)
            validate(value, schema)
            return value

    m = Counted()
    pipeline = Pipeline(Suite(), Runtime(m), Learner(m), tmp_path, batch_size=1)
    result = pipeline.run()
    n = len(m.calls)
    assert result["test"]["successes"] == 2
    assert result["frozen_hash_before"] == result["frozen_hash_after"]
    assert pipeline.run() == result and len(m.calls) == n
    assert read(tmp_path / "manifest.json")["runtime"]["planning_enabled"] is True
    assert any(
        e["type"] == "skeleton_construction"
        for e in read(tmp_path / "test/00000.json")["events"]
    )


def test_planner_real_docker_program_execution():
    image = os.environ.get("SKILLCOMPILER_TEST_IMAGE")
    if not image:
        pytest.skip("Set SKILLCOMPILER_TEST_IMAGE for actual Docker")
    s = asset()
    s["program_ids"] = ["read_program"]
    p = {
        "id": "read_program",
        "name": "Read",
        "allowed_tools": ["lookup"],
        **lookup_contract(),
        "source": "def run(ctx, inputs):\n    return {'status':'ok','outputs':ctx.call('lookup', inputs)}\n",
    }
    suite = Suite()
    t = suite.tasks("test")[0]
    bank = Bank([s], [p])
    before = bank.hash
    trace = Runtime(DemoModel(), ProgramWorker(image)).run(
        t, suite.open(t), bank, "test/0"
    )
    assert trace["answer"] == 3 and bank.hash == before
    assert any(
        e["type"] == "program" and e["result"]["status"] == "ok"
        for e in trace["events"]
    )


def two_step_analysis():
    a = lookup_analysis()
    second = {
        "input_schema": {
            "type": "object",
            "properties": {"key": {"type": "string"}, "value": {"type": "integer"}},
            "required": ["key", "value"],
            "additionalProperties": False,
        },
        "output_schema": object_schema("text", "string"),
        "preconditions": [
            {
                "predicate": "observed_value",
                "effect_domain": "evidence",
                "args": {
                    "query": {"kind": "skill_input", "source_role": "key"},
                    "result": {"kind": "skill_input", "source_role": "value"},
                },
            }
        ],
        "effects": [
            {
                "predicate": "formatted_value",
                "effect_domain": "evidence",
                "args": {
                    "query": {"kind": "skill_input", "source_role": "key"},
                    "result": {"kind": "skill_output", "source_role": "text"},
                },
            }
        ],
    }
    a["operations"].append(
        {
            "id": "format",
            "intent": "Format the observed value",
            "contract": second,
            "args": {
                "key": {"task": "key"},
                "value": {"from": "read", "field": "value"},
            },
            "after": ["read"],
        }
    )
    a["contract"]["output_schema"] = object_schema("answer", "string")
    a["contract"]["effects"][0]["predicate"] = "formatted_value"
    a["outputs"] = {"answer": {"from": "format", "field": "text"}}
    return a


def test_multistep_dependencies_and_native_execution():
    a = two_step_analysis()
    assert validate_analysis(a, {"key": "item_0"})["operations"] == 2
    bad = deepcopy(a)
    bad["operations"][1]["after"] = []
    with pytest.raises(ValueError, match="dependency"):
        validate_analysis(bad, {"key": "item_0"})

    class Env(Environment):
        def verify_effects(self, effects, inputs, variables, result):
            if effects[0]["predicate"] == "formatted_value":
                ref = effects[0]["args"]["result"]["source_role"]
                return inputs["key"] in self.observed and result["outputs"][ref] == str(
                    self.observed[inputs["key"]]
                )
            return super().verify_effects(effects, inputs, variables, result)

    class Model(DemoModel):
        def ask(self, key, prompt, m, schema):
            if key.endswith("/analyze"):
                return a
            if "/node_format/" in key:
                return {
                    "action": "finish",
                    "answer": {"text": str(m["inputs"]["value"])},
                }
            return super().ask(key, prompt, m, schema)

    t = Suite().tasks("test")[0]
    trace = Runtime(Model()).run(t, Env(t), Bank(), "test/0")
    assert trace["answer"] == "3"
    assert [e["id"] for e in trace["events"] if e["type"] == "skeleton_node"] == [
        "read",
        "format",
    ]


def test_default_planner_calls_runtime_programmer_in_container():
    image = os.environ.get("SKILLCOMPILER_TEST_IMAGE")
    if not image:
        pytest.skip("Set SKILLCOMPILER_TEST_IMAGE for actual Docker")

    class Model(DemoModel):
        def ask(self, key, prompt, m, schema):
            if key.endswith("/build"):
                return {
                    "source": "def run(ctx, inputs):\n    return {'status':'ok','outputs':ctx.call('lookup', inputs)}\n"
                }
            if "/node_" in key and not any(
                e["type"] == "program" for e in m["history"]
            ):
                return {
                    "action": "programmer",
                    "operation": {
                        "name": "Read",
                        "instruction": "Read key via lookup.",
                        "allowed_tools": ["lookup"],
                        **lookup_contract(),
                    },
                    "arguments": m["inputs"],
                }
            return super().ask(key, prompt, m, schema)

    suite = Suite()
    t = suite.tasks("test")[0]
    bank = Bank()
    before = bank.hash
    trace = Runtime(Model(), ProgramWorker(image)).run(t, suite.open(t), bank, "test/0")
    assert trace["answer"] == 3 and trace["programmer_calls"] == 1
    assert bank.hash == before and not bank.programs
    assert any(
        e["type"] == "runtime_qualification" and e["status"] == "ok"
        for e in trace["events"]
    )


def test_merge_rebinds_node_contract_without_changing_goal():
    from skillcompiler.alignment import integrate

    parent = Bank(
        [asset("a"), asset("b")],
        skeletons=[
            {**skeleton(), "nodes": [{**skeleton()["nodes"][0], "skill_id": "a"}]}
        ],
    )
    child, receipts = integrate(
        parent,
        {
            "skills": [asset("merged", "query", "result")],
            "programs": [],
            "skeletons": [],
        },
        [{"kind": "skills", "source_ids": ["a", "b"], "target_id": "merged"}],
        [],
        refine=True,
        evidence=[{"episode": "t0", "events": [0]}, {"episode": "t1", "events": [0]}],
    )
    plan = child.skeletons["complete"]
    assert plan["contract"] == parent.skeletons["complete"]["contract"]
    assert plan["nodes"][0]["contract"]["input_schema"] == object_schema(
        "query", "string"
    )
    assert (
        validate_analysis(skeleton_analysis(plan, child), {"key": "item_0"})["status"]
        == "valid"
    )
