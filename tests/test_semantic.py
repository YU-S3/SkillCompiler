from copy import deepcopy
import os
import pytest
from skillcompiler.bank import Bank
from skillcompiler.semantic import align, verify, contract, lexical_similarity
from skillcompiler.alignment import integrate, skeleton_shape
from skillcompiler.runtime import Runtime
from skillcompiler.learning import Learner
from skillcompiler.tools import ToolContext
from skillcompiler.model import BudgetExceeded
from skillcompiler.pipeline import Pipeline
from skillcompiler.contracts import read, digest
from skillcompiler.worker import ProgramWorker
from examples.key_value import Suite


def schema(role, typ):
    return {
        "type": "object",
        "properties": {role: {"type": typ}},
        "required": [role],
        "additionalProperties": False,
    }


def skill(sid="a", i="key", o="value", pid=None):
    s = {
        "id": sid,
        "name": "Lookup",
        "guidance": "Return the public lookup observation.",
        "input_schema": schema(i, "string"),
        "output_schema": schema(o, "integer"),
        "effects": [
            {
                "predicate": "observed_value",
                "effect_domain": "evidence",
                "cardinality": 1,
                "args": {
                    "query": {"kind": "skill_input", "source_role": i},
                    "result": {"kind": "skill_output", "source_role": o},
                },
            }
        ],
    }
    if pid:
        s["program_ids"] = [pid]
    return s


def program(pid="p"):
    return {
        "id": pid,
        "name": "Lookup",
        "format": "python",
        "input_schema": schema("key", "string"),
        "output_schema": schema("value", "integer"),
        "allowed_tools": ["lookup"],
        "source": "def run(ctx, inputs):\n    return {'status':'ok', 'outputs':ctx.call('lookup', inputs)}\n",
    }


def skeleton(sid="s", skill_id="a", i="key", o="value"):
    return {
        "id": sid,
        "name": "Lookup",
        "nodes": [{"id": "n", "skill_id": skill_id, "args": {i: {"task": "key"}}}],
        "outputs": {"answer": {"from": "n", "field": o}},
    }


EVIDENCE = [
    {"episode": "train_0", "events": [0]},
    {"episode": "train_1", "events": [0]},
]


def test_alpha_identity_proof_and_literal_preservation():
    a, b = skill(), skill("b", "query", "result")
    a["validator"] = {"constant": "$key", "flag": True}
    b["validator"] = deepcopy(a["validator"])
    result = align(a, b)
    assert result["status"] == "exact" and verify(a, b, result["proof"])
    assert result["proof"]["input_map"] == {"key": "query"}
    tampered = deepcopy(result["proof"])
    tampered["input_map"] = {"key": "invented"}
    assert not verify(a, b, tampered)
    b["validator"]["flag"] = 1
    assert align(a, b)["status"] == "different"


@pytest.mark.parametrize(
    "field,value",
    [
        ("predicate", "other"),
        ("cardinality", 2),
        ("effect_domain", "world"),
        ("distinct_by", "identity"),
    ],
)
def test_non_equivalent_effects_never_merge(field, value):
    a, b = skill(), skill("b", "query", "result")
    b["effects"][0][field] = value
    assert align(a, b)["status"] == "different"


def test_resolution_types_and_preconditions_are_identity():
    a, b = skill(), skill("b")
    b["input_schema"]["properties"]["key"]["semantic_type"] = "resolved_handle"
    assert align(a, b)["status"] == "different"
    b = skill("b")
    b["preconditions"] = deepcopy(b["effects"])
    assert align(a, b)["status"] == "different"


def test_bounded_search_and_unknown_refs_fail_closed():
    a, b = skill(), skill("b")
    assert align(a, b, max_states=1)["status"] == "unknown"
    b["effects"][0]["args"]["query"]["source_role"] = "made_up"
    assert align(a, b)["status"] == "unknown"
    b = skill("b")
    b["input_schema"]["$ref"] = "#/missing"
    assert align(a, b)["status"] == "unknown"


def test_exact_alignment_rebinds_new_skeleton_and_program_abi():
    parent = Bank(skills=[skill("kept", "query", "result")], programs=[program()])
    before = parent.hash
    child, receipt = integrate(
        parent,
        {
            "skills": [skill("alias", pid="p")],
            "skeletons": [skeleton(skill_id="alias")],
        },
    )
    assert set(child.skills) == {"kept"} and parent.hash == before
    assert child.skills["kept"]["implementations"] == [
        {
            "program_id": "p",
            "input_map": {"key": "query"},
            "output_map": {"result": "value"},
        }
    ]
    assert child.skeletons["s"]["nodes"][0]["args"] == {"query": {"task": "key"}}
    assert child.skeletons["s"]["outputs"]["answer"]["field"] == "result"
    assert verify(skill("alias", pid="p"), parent.skills["kept"], receipt[0]["proof"])


def test_explicit_merge_is_atomic_and_rewires_all_consumers():
    parent = Bank(
        skills=[skill("a"), skill("b", "query", "result")],
        skeletons=[skeleton("sa", "a"), skeleton("sb", "b", "query", "result")],
    )
    before = parent.hash
    merge = {"kind": "skills", "source_ids": ["a", "b"], "target_id": "merged"}
    child, receipts = integrate(
        parent,
        {"skills": [skill("merged", "item", "answer")]},
        [merge],
        refine=True,
        evidence=EVIDENCE,
    )
    assert set(child.skills) == {"merged"} and len(receipts[0]["parents"]) == 2
    assert parent.hash == before
    for s in child.skeletons.values():
        assert s["nodes"][0]["skill_id"] == "merged"
        assert s["outputs"]["answer"]["field"] == "answer"
    bad = skill("merged")
    bad["effects"][0]["cardinality"] = 2
    with pytest.raises(ValueError, match="exact proof"):
        integrate(parent, {"skills": [bad]}, [merge], refine=True, evidence=EVIDENCE)
    assert parent.hash == before


@pytest.mark.parametrize("refine,evidence", [(False, EVIDENCE), (True, EVIDENCE[:1])])
def test_merge_requires_independent_train_evidence(refine, evidence):
    with pytest.raises(ValueError, match="independent"):
        integrate(
            Bank(skills=[skill("a"), skill("b")]),
            {"skills": [skill("c")]},
            [{"kind": "skills", "source_ids": ["a", "b"], "target_id": "c"}],
            refine=refine,
            evidence=evidence,
        )


def test_no_silent_deletion_or_text_only_semantic_equivalence():
    a, b = skill(), skill("b")
    a.pop("effects")
    b.pop("effects")
    parent = Bank(skills=[a])
    child, _ = integrate(parent, {"skills": [b]})
    assert len(child.skills) == 2
    with pytest.raises(ValueError, match="deletion"):
        integrate(parent, {}, removals=["a"])


def test_program_and_skeleton_merge_identity_guards():
    parent = Bank(programs=[program("p"), program("q")])
    merge = {"kind": "programs", "source_ids": ["p", "q"], "target_id": "r"}
    child, _ = integrate(
        parent, {"programs": [program("r")]}, [merge], refine=True, evidence=EVIDENCE
    )
    assert set(child.programs) == {"r"}
    changed = program("r")
    changed["allowed_tools"] = []
    with pytest.raises(ValueError, match="executable"):
        integrate(
            parent, {"programs": [changed]}, [merge], refine=True, evidence=EVIDENCE
        )
    a, b = skeleton(), skeleton("b")
    b["nodes"][0]["id"] = "renamed"
    b["outputs"]["answer"]["from"] = "renamed"
    assert skeleton_shape(a) == skeleton_shape(b)
    b["nodes"][0]["args"]["key"] = {"literal": "not the task"}
    assert skeleton_shape(a) != skeleton_shape(b)


def test_retrieval_uses_dice_and_contract_filter():
    assert lexical_similarity("a a b", "a b") == 0.8
    other = skill("other")
    other["effects"][0]["predicate"] = "different"
    bank = Bank(skills=[skill(), other])
    assert [
        s["id"] for s in bank.retrieve("Lookup", requirement=skill("query"))["skills"]
    ] == ["a"]


def test_new_program_and_skeleton_alignment_reuses_existing_layers():
    parent = Bank(
        programs=[program("p")], skills=[skill("a", pid="p")], skeletons=[skeleton()]
    )
    before = parent.hash
    child, receipts = integrate(
        parent,
        {
            "programs": [program("alias_program")],
            "skills": [skill("alias_skill", pid="alias_program")],
            "skeletons": [skeleton("alias_skeleton", "alias_skill")],
        },
    )
    assert (
        set(child.programs) == {"p"}
        and set(child.skills) == {"a"}
        and set(child.skeletons) == {"s"}
    )
    assert {r["kind"] for r in receipts} == {"skills", "programs", "skeletons"}
    assert parent.hash == before


def test_mixed_layer_merge_closes_every_reference():
    parent = Bank(
        programs=[program("p"), program("q")],
        skills=[skill("a", pid="p"), skill("b", pid="q")],
        skeletons=[skeleton("sa", "a"), skeleton("sb", "b")],
    )
    # Deliberately supplied in reverse dependency order.
    merges = [
        {"kind": "skeletons", "source_ids": ["sa", "sb"], "target_id": "sc"},
        {"kind": "programs", "source_ids": ["p", "q"], "target_id": "r"},
        {"kind": "skills", "source_ids": ["a", "b"], "target_id": "c"},
    ]
    child, receipts = integrate(
        parent,
        {
            "programs": [program("r")],
            "skills": [skill("c")],
            "skeletons": [skeleton("sc", "c")],
        },
        merges,
        refine=True,
        evidence=EVIDENCE,
    )
    assert (
        set(child.programs) == {"r"}
        and set(child.skills) == {"c"}
        and set(child.skeletons) == {"sc"}
    )
    assert all(i["program_id"] == "r" for i in child.skills["c"]["implementations"])
    assert child.skeletons["sc"]["nodes"][0]["skill_id"] == "c"
    assert len(receipts) == 3


class FakeWorker:
    def execute(self, p, inputs, ctx):
        return {"status": "ok", "outputs": ctx.call("lookup", inputs)}


class BuildModel:
    def __init__(self, fail_first=False, fail_all=False, provider_error=None):
        self.calls = []
        self.fail_first, self.fail_all, self.provider_error = (
            fail_first,
            fail_all,
            provider_error,
        )

    def ask(self, key, prompt, m, s):
        self.calls.append((key, deepcopy(m)))
        if key.endswith("/extract"):
            return {"decision": "no_change"}
        if key.endswith(("/build", "/repair")):
            if self.provider_error:
                raise self.provider_error
            bad = self.fail_all or self.fail_first and key.endswith("/build")
            return {"source": "def broken(" if bad else program()["source"]}
        if key.endswith("/plan"):
            return {"skeleton_id": "s", "inputs": {}}
        completed = [e for e in m["history"] if e["type"] == "program"]
        if completed:
            value = completed[-1]["result"]["outputs"]["value"]
            return {
                "action": "finish",
                "answer": {"answer": value} if "/node_" in key else value,
            }
        return {
            "action": "programmer",
            "operation": {
                "name": "Lookup",
                "instruction": "Read the public key.",
                "input_schema": schema("key", "string"),
                "output_schema": schema("value", "integer"),
                "allowed_tools": ["lookup"],
            },
            "arguments": {"key": m["task"]["inputs"]["key"]},
        }


@pytest.mark.parametrize("phase", ["train", "val", "test"])
def test_programmer_in_each_phase_without_persistent_bank_write(phase):
    model, suite, bank = BuildModel(), Suite(), Bank()
    t = suite.tasks(phase)[0]
    trace = Runtime(model, FakeWorker(), planning_enabled=False).run(
        t, suite.open(t), bank, phase + "/0"
    )
    assert trace["answer"] is not None and not bank.programs
    assert trace["bank_hash"] == bank.hash != trace["task_overlay_hash"]
    assert trace["programmer_calls"] == 1
    assert sum(e["type"] == "tool" for e in trace["events"]) == 1
    receipt = next(e for e in trace["events"] if e["type"] == "runtime_programmer")
    assert receipt["scope"] == "task_local" and receipt["admission"] == "structural"
    assert all("feedback" not in m for _, m in model.calls)


def test_dynamic_skeleton_can_request_programmer():
    bank = Bank(
        skeletons=[
            {
                "id": "s",
                "name": "lookup",
                "nodes": [{"id": "n", "args": {"key": {"task": "key"}}}],
                "outputs": {"answer": {"from": "n", "field": "answer"}},
            }
        ]
    )
    t, model, suite = Suite().tasks("test")[0], BuildModel(), Suite()
    result = Runtime(model, FakeWorker(), planning_enabled=False).run(
        t, suite.open(t), bank, "test/0"
    )
    assert result["terminal_reason"] == "finished" and result["programmer_calls"] == 1


def test_programmer_structural_repair_and_total_limits():
    suite = Suite()
    t = suite.tasks("test")[0]
    m = BuildModel(fail_first=True)
    trace = Runtime(m, FakeWorker(), planning_enabled=False).run(
        t, suite.open(t), Bank(), "test/0"
    )
    assert trace["answer"] is not None
    assert len([k for k, _ in m.calls if k.endswith(("/build", "/repair"))]) == 2
    m = BuildModel(fail_all=True)
    trace = Runtime(
        m, FakeWorker(), max_steps=5, max_programs_per_task=1, planning_enabled=False
    ).run(t, suite.open(t), Bank(), "test/0")
    assert trace["answer"] is None and trace["programmer_calls"] == 1
    assert len([k for k, _ in m.calls if k.endswith(("/build", "/repair"))]) == 2


@pytest.mark.parametrize("error", [BudgetExceeded("budget"), RuntimeError("HTTP 504")])
def test_provider_failures_propagate_without_becoming_program_repairs(error):
    suite = Suite()
    t = suite.tasks("test")[0]
    m = BuildModel(provider_error=error)
    with pytest.raises(type(error)):
        Runtime(m, FakeWorker(), planning_enabled=False).run(
            t, suite.open(t), Bank(), "test/0"
        )
    assert not any(k.endswith("/repair") for k, _ in m.calls)


def test_programmer_disabled_and_missing_worker_never_build():
    suite = Suite()
    t = suite.tasks("test")[0]
    for kwargs in (
        {"worker": None},
        {"worker": FakeWorker(), "allow_programmer": False},
    ):
        m = BuildModel()
        trace = Runtime(m, max_steps=2, **kwargs, planning_enabled=False).run(
            t, suite.open(t), Bank(), "test/0"
        )
        assert trace["programmer_calls"] == 0 and not any(
            k.endswith("/build") for k, _ in m.calls
        )


def test_programmer_full_pipeline_resume_and_trace_artifacts(tmp_path):
    m, suite = BuildModel(), Suite()
    pipeline = Pipeline(
        suite,
        Runtime(m, FakeWorker(), planning_enabled=False),
        Learner(m),
        tmp_path,
        refinement_rounds=0,
    )
    result = pipeline.run()
    count = len(m.calls)
    assert (
        result["test"]["successes"] == 2 and result["frozen_hash_before"] == Bank().hash
    )
    assert pipeline.run() == result and len(m.calls) == count
    row = read(tmp_path / "test/00000.json")
    assert any(
        e["type"] == "runtime_programmer" and "source" in e["program"]
        for e in row["events"]
    )


@pytest.fixture
def docker_worker():
    image = os.environ.get("SKILLCOMPILER_TEST_IMAGE")
    if not image:
        pytest.skip("Set SKILLCOMPILER_TEST_IMAGE for actual Docker verification")
    return ProgramWorker(image, timeout=30)


def test_real_docker_test_runtime_programmer(docker_worker, tmp_path):
    suite, model, bank = Suite(), BuildModel(fail_first=True), Bank()
    bank.write(tmp_path / "frozen")
    t = suite.tasks("test")[0]
    trace = Runtime(model, docker_worker, planning_enabled=False).run(
        t, suite.open(t), bank, "test/0"
    )
    assert trace["answer"] == 3 and Bank.load(tmp_path / "frozen").hash == bank.hash
    assert sum(e["type"] == "tool" for e in trace["events"]) == 1
    assert (
        next(e for e in trace["events"] if e["type"] == "runtime_programmer")[
            "attempts"
        ]
        == 2
    )


def test_real_docker_learning_merge_replays_and_retains_parent(docker_worker):
    suite = Suite()
    p = program()
    parent = Bank(
        skills=[skill("a", pid="p"), skill("b", pid="p")],
        programs=[p],
        skeletons=[skeleton()],
    )
    episodes = []
    runtime = Runtime(None, docker_worker, planning_enabled=False)
    for t in suite.tasks("train"):
        ctx = ToolContext(suite.open(t))
        runtime.program(parent, "p", t.inputs, ctx)
        episodes.append({"task": t.public(), "events": ctx.events})
    proposal = {
        "decision": "propose",
        "skills": [skill("merged", "query", "result")],
        "merges": [{"kind": "skills", "source_ids": ["a", "b"], "target_id": "merged"}],
        "evidence": [{"episode": e["task"]["id"], "events": [0, 1]} for e in episodes],
    }

    class M:
        def ask(self, *a):
            return proposal

    before = parent.hash
    child, receipt = Learner(M(), docker_worker, repair_limit=0).update(
        parent, episodes, suite, "refine", refine=True
    )
    assert receipt["decision"] == "proposed" and len(receipt["qualification"]) == 4
    assert set(child.skills) == {"merged"} and parent.hash == before
    env = suite.open(suite.tasks("test")[0])
    env.verify_effects = lambda effects, inputs, locals, result: result["outputs"] == {
        "result": 3
    }
    output = runtime.skill(child, "merged", {"query": "item_0"}, ToolContext(env))
    assert output["outputs"] == {"result": 3}


def test_identity_failure_is_not_a_programmer_repair():
    from skillcompiler.programmer import Programmer

    class M:
        calls = 0

        def ask(self, *a):
            self.calls += 1
            raise ValueError("Cached request identity changed")

    m, suite = M(), Suite()
    t = suite.tasks("test")[0]
    spec = {
        k: v
        for k, v in program().items()
        if k in {"name", "input_schema", "output_schema", "allowed_tools"}
    }
    spec["instruction"] = "Read the key"
    with pytest.raises(ValueError, match="identity"):
        Programmer(m).compile(spec, t.inputs, t, ToolContext(suite.open(t)), "x")
    assert m.calls == 1


def test_runtime_preconditions_and_ambiguous_bindings_block_execution():
    suite = Suite()
    t = suite.tasks("test")[0]
    ctx = ToolContext(suite.open(t))
    p = program()
    p["preconditions"] = [{"predicate": "ready", "args": {}}]
    with pytest.raises(ValueError, match="Preconditions"):
        Runtime(None, FakeWorker(), planning_enabled=False).program(
            Bank(programs=[p]), "p", t.inputs, ctx
        )
    assert ctx.calls == 0
    s = skill(pid="p")
    s["program_ids"].append("q")
    with pytest.raises(ValueError, match="Ambiguous"):
        Runtime(None, FakeWorker(), planning_enabled=False).skill(
            Bank(skills=[s], programs=[program(), program("q")]), "a", t.inputs, ctx
        )
    assert ctx.calls == 0


def test_invalid_merge_mapping_cannot_replace_source_binding():
    parent = Bank(
        skills=[skill("a", pid="p"), skill("b", pid="p")],
        programs=[program(), program("other")],
    )
    target = skill("c", pid="other")
    with pytest.raises(ValueError, match="unproved implementation"):
        integrate(
            parent,
            {"skills": [target]},
            [{"kind": "skills", "source_ids": ["a", "b"], "target_id": "c"}],
            refine=True,
            evidence=EVIDENCE,
        )


def test_revision_cannot_silently_change_consumer_abi_or_program_binding():
    parent = Bank(
        skills=[skill("a", pid="p")],
        programs=[program(), program("q")],
        skeletons=[skeleton()],
    )
    with pytest.raises(ValueError, match="role names"):
        integrate(
            parent,
            {"skills": [skill("a", "query", "result", "p")]},
            refine=True,
            evidence=EVIDENCE,
        )
    with pytest.raises(ValueError, match="replay-qualified"):
        integrate(
            parent, {"skills": [skill("a", pid="q")]}, refine=True, evidence=EVIDENCE
        )


def test_real_docker_replay_rejection_retains_original_bank(docker_worker):
    suite, p = Suite(), program()
    parent = Bank(programs=[p])
    before = parent.hash
    episodes = []
    for t in suite.tasks("train"):
        ctx = ToolContext(suite.open(t))
        Runtime(None, docker_worker, planning_enabled=False).program(
            parent, "p", t.inputs, ctx
        )
        episodes.append({"task": t.public(), "events": ctx.events})

    class M:
        def ask(self, key, prompt, material, schema):
            if "/build_" in key:
                return {
                    "source": "def run(ctx,inputs):\n return {'status':'ok','outputs':{'value':999}}"
                }
            spec = {k: v for k, v in p.items() if k not in {"source", "format"}}
            spec["trials"] = [
                {
                    "episode": episodes[0]["task"]["id"],
                    "inputs": episodes[0]["task"]["inputs"],
                    "prefix": [],
                    "expected_from": {"event": 0, "path": ["result"]},
                }
            ]
            return {
                "decision": "propose",
                "programs": [spec],
                "evidence": [
                    {"episode": e["task"]["id"], "events": [0]} for e in episodes
                ],
            }

    child, receipt = Learner(M(), docker_worker, repair_limit=0).update(
        parent, episodes, suite, "refine", refine=True
    )
    assert receipt["decision"] == "rejected" and child.hash == parent.hash == before


def test_real_transport_programmer_accounting_and_cached_resume(
    docker_worker, tmp_path, monkeypatch
):
    import json
    from skillcompiler.model import Ledger, ChatModel

    suite = Suite()
    task = suite.tasks("test")[0]
    decision = BuildModel().ask(
        "step_0", "", {"history": [], "task": task.public()}, {}
    )
    responses = [
        decision,
        {"source": "def broken("},
        {"source": program()["source"]},
        {"action": "finish", "answer": 3},
    ]
    payloads = []

    class Response:
        ok = True

        def __init__(self, value):
            self.value = value

        def json(self):
            return {
                "choices": [{"message": {"content": json.dumps(self.value)}}],
                "usage": {"total_tokens": 20},
            }

    def post(*a, **kw):
        payloads.append(kw["json"])
        return Response(responses.pop(0))

    monkeypatch.setenv("MODEL_API_KEY", "not-a-real-key")
    monkeypatch.setattr("skillcompiler.model.requests.post", post)
    ledger = Ledger(tmp_path / "usage.json", 1000000, 10)
    model = ChatModel(
        {"base_url": "https://invalid.test", "model": "fixture", "max_tokens": 1000},
        tmp_path / "requests",
        ledger,
    )
    runtime = Runtime(model, docker_worker, planning_enabled=False)
    bank = Bank()
    first = runtime.run(task, suite.open(task), bank, "test/0")
    assert first["answer"] == 3 and len(payloads) == 4
    second = runtime.run(task, suite.open(task), bank, "test/0")
    assert first == second and len(payloads) == 4
    assert (
        ledger.summary()["known_tokens"] == 80
        and ledger.summary()["unknown_reserved"] == 0
    )
    assert all(
        "feedback" not in json.loads(p["messages"][1]["content"])["input"]
        for p in payloads
    )
    assert "not-a-real-key" not in "".join(
        p.read_text() for p in tmp_path.rglob("*.json")
    )


def test_runtime_execution_failure_does_not_repeat_side_effect(docker_worker):
    suite, t = Suite(), Suite().tasks("test")[0]

    class M(BuildModel):
        def ask(self, key, prompt, m, schema):
            if key.endswith("/build"):
                self.calls.append((key, m))
                return {
                    "source": "def run(ctx,inputs):\n ctx.call('lookup',inputs)\n raise ValueError('after effect')"
                }
            if any(e["type"] == "operation_error" for e in m.get("history", [])):
                return {"action": "finish", "answer": None}
            return super().ask(key, prompt, m, schema)

    m = M()
    trace = Runtime(m, docker_worker, planning_enabled=False).run(
        t, suite.open(t), Bank(), "test/0"
    )
    assert sum(e["type"] == "tool" for e in trace["events"]) == 1
    assert not any(k.endswith("/repair") for k, _ in m.calls)
    assert any(
        e["type"] == "runtime_qualification" and e["status"] == "failed"
        for e in trace["events"]
    )
