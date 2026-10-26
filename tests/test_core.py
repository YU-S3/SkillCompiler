import json
from pathlib import Path
import pytest
from skillcompiler.bank import Bank
from skillcompiler.contracts import Task, Feedback, read, save
from skillcompiler.learning import Learner, evidence_ok
from skillcompiler.model import Ledger, BudgetExceeded, ChatModel
from skillcompiler.pipeline import Pipeline
from skillcompiler.runtime import Runtime
from skillcompiler.tools import ToolContext, resolve
from skillcompiler.locking import run_lock
from examples.key_value import Suite


class ScriptedModel:
    def __init__(self):
        self.calls = []

    def ask(self, key, prompt, material, schema):
        self.calls.append((key, material))
        if key.endswith("/extract"):
            episode = material["episodes"][0]
            return {
                "decision": "propose",
                "skills": [
                    {
                        "id": "lookup",
                        "name": "Lookup",
                        "guidance": "Read the named key with lookup before returning its observed value.",
                    }
                ],
                "evidence": [{"episode": episode["task"]["id"], "events": [0]}],
            }
        if "/step_" in key:
            tools = [e for e in material["history"] if e["type"] == "tool"]
            if tools:
                return {"action": "finish", "answer": tools[-1]["result"]["value"]}
            return {
                "action": "tool",
                "id": "lookup",
                "arguments": {"key": material["task"]["inputs"]["key"]},
            }
        raise AssertionError(key)


def test_pipeline_real_tools_selection_freeze_resume(tmp_path):
    model = ScriptedModel()
    suite = Suite()
    p = Pipeline(
        suite,
        Runtime(model),
        Learner(model),
        tmp_path,
        batch_size=1,
        refinement_rounds=1,
    )
    result = p.run()
    count = len(model.calls)
    assert result["test"]["successes"] == 2
    assert result["frozen_hash_before"] == result["frozen_hash_after"]
    assert p.run() == result and len(model.calls) == count
    learning = [m for k, m in model.calls if k.endswith("/extract")]
    assert all(
        e["task"]["id"].startswith("train_") for m in learning for e in m["episodes"]
    )
    assert all("value" not in t.inputs for t in suite.tasks("train"))
    # A tampered checkpoint may not be adopted during resume.
    path = tmp_path / "learning/0/bank/bank.json"
    data = read(path)
    data["skills"][0]["guidance"] = "altered"
    save(path, data)
    with pytest.raises(ValueError, match="checkpoint"):
        p.run()


def test_split_isolation(tmp_path):
    class Overlap(Suite):
        def tasks(self, split):
            return super().tasks("train")

    with pytest.raises(ValueError, match="overlap"):
        Pipeline(Overlap(), None, None, tmp_path).run()


def test_configuration_identity(tmp_path):
    m = ScriptedModel()
    p = Pipeline(Suite(), Runtime(m), Learner(m), tmp_path, config_identity="a")
    p.run()
    p.config_identity = "b"
    with pytest.raises(ValueError, match="identity"):
        p.run()


def test_bank_export_does_not_mutate_source(tmp_path):
    b = Bank(
        programs=[
            {
                "id": "p",
                "source": "def run(ctx, inputs): return inputs",
                "input_schema": {},
                "output_schema": {},
                "allowed_tools": [],
            }
        ]
    )
    before = b.hash
    b.write(tmp_path)
    assert b.hash == before == Bank.load(tmp_path).hash


@pytest.mark.parametrize("case", ["duplicate", "reference", "forward", "source_escape"])
def test_invalid_bank(case, tmp_path):
    with pytest.raises(ValueError):
        if case == "duplicate":
            Bank(skills=[{"id": "x"}, {"id": "x"}])
        elif case == "reference":
            Bank(
                skills=[
                    {
                        "id": "x",
                        "name": "x",
                        "guidance": "x",
                        "program_ids": ["missing"],
                    }
                ]
            )
        elif case == "forward":
            Bank(
                skeletons=[
                    {"id": "x", "nodes": [{"id": "a", "args": {"x": {"from": "b"}}}]}
                ]
            )
        else:
            save(
                tmp_path / "bank.json",
                {
                    "skills": [],
                    "skeletons": [],
                    "programs": [{"source_file": "../secret"}],
                },
            )
            Bank.load(tmp_path)


def test_tools_and_explicit_references():
    suite = Suite()
    ctx = ToolContext(suite.open(suite.tasks("train")[0]), 1)
    assert ctx.call("lookup", {"key": "item_0"}) == {"value": 3}
    with pytest.raises(RuntimeError, match="budget"):
        ctx.call("lookup", {"key": "item_0"})
    assert resolve(
        {"x": [{"task": "k"}, {"from": "a", "field": "values.0"}]},
        {"k": 1},
        {"a": {"values": [2]}},
    ) == {"x": [1, 2]}


def test_dynamic_skeleton_uses_real_calls():
    suite = Suite()
    task = suite.tasks("train")[0]

    class Model:
        def ask(self, key, prompt, m, schema):
            if key.endswith("/plan"):
                return {"skeleton_id": "s", "inputs": {}}
            if not m["history"]:
                return {"action": "tool", "id": "lookup", "arguments": m["inputs"]}
            return {
                "action": "finish",
                "answer": {"answer": m["history"][-1]["result"]["value"]},
            }

    b = Bank(
        skeletons=[
            {
                "id": "s",
                "name": "Lookup",
                "nodes": [
                    {
                        "id": "read",
                        "goal": "Read the key",
                        "args": {"key": {"task": "key"}},
                    }
                ],
                "outputs": {"answer": {"from": "read", "field": "answer"}},
            }
        ]
    )
    result = Runtime(Model()).run(task, suite.open(task), b, "t")
    assert result["answer"] == 3
    assert [e["type"] for e in result["events"]] == [
        "tool",
        "skeleton_node",
        "skeleton_output",
    ]


def test_evidence_failure_and_legal_abstention():
    episodes = [{"task": {"id": "t"}, "events": [{}]}]
    for refs in (
        [{"episode": "unknown", "events": [0]}],
        [{"episode": "t", "events": [-1]}],
    ):
        with pytest.raises(ValueError):
            evidence_ok({"evidence": refs}, episodes)

    class Model:
        def ask(self, *a):
            return {"decision": "no_change", "programs": [{"ignored": "unused"}]}

    b, r = Learner(Model()).update(Bank(), [], None, "x")
    assert r["decision"] == "no_change" and not b.skills


def test_ledger_unknown_is_reserved(tmp_path):
    l = Ledger(tmp_path / "usage.json", 100, 3)
    l.reserve("a", 60)
    l.complete("a", error="network")
    with pytest.raises(BudgetExceeded):
        l.reserve("b", 50)
    l.reserve("b", 40)
    l.complete("b", {"total_tokens": 10, "prompt_tokens": 4, "completion_tokens": 6})
    assert l.summary() == {"requests": 2, "known_tokens": 10, "unknown_reserved": 60}


def test_http_retry_cache_and_actual_usage(tmp_path, monkeypatch):
    import requests

    monkeypatch.setenv("MODEL_API_KEY", "fixture-only")
    raw = {
        "choices": [{"message": {"content": '{"ok":true}'}}],
        "usage": {"total_tokens": 12},
    }

    class Response:
        ok = True

        def json(self):
            return raw

    calls = []

    def post(*a, **kw):
        calls.append(kw)
        if len(calls) == 1:
            raise requests.Timeout()
        return Response()

    monkeypatch.setattr("skillcompiler.model.requests.post", post)
    monkeypatch.setattr("skillcompiler.model.time.sleep", lambda n: None)
    l = Ledger(tmp_path / "ledger.json", 100000, 3)
    m = ChatModel(
        {"model": "fixture", "base_url": "https://invalid.test", "max_tokens": 100},
        tmp_path / "requests",
        l,
    )
    assert m.ask("one", "base", {}, {"type": "object"}) == {"ok": True}
    assert (
        m.ask("one", "base", {}, {"type": "object"}) == {"ok": True} and len(calls) == 2
    )
    assert l.summary()["unknown_reserved"] > 0 and l.summary()["known_tokens"] == 12
    assert "fixture-only" not in "".join(
        p.read_text() for p in tmp_path.rglob("*.json")
    )
    with pytest.raises(ValueError, match="identity"):
        m.ask("one", "different", {}, {})
    with pytest.raises(ValueError, match="escapes"):
        m.ask("../../x", "base", {}, {})


def test_exclusive_run_lock(tmp_path):
    with run_lock(tmp_path):
        with pytest.raises(OSError):
            with run_lock(tmp_path):
                pass


def test_core_has_no_benchmark_dispatch():
    root = Path(__file__).resolve().parents[1]
    names = [p.name for p in (root / "benchmark").iterdir() if p.is_dir()]
    text = "\n".join(
        p.read_text().lower() for p in (root / "src/skillcompiler").glob("*.py")
    )
    assert not any(name.lower() in text for name in names)


def test_response_contract_error_has_one_learning_repair():
    from skillcompiler.model import ResponseContractError

    class Model:
        def __init__(self):
            self.calls = 0

        def ask(self, *args):
            self.calls += 1
            raise ResponseContractError("Invalid JSON", {"choices": []})

    m = Model()
    bank, receipt = Learner(m).update(Bank(), [], None, "learn")
    assert m.calls == 2 and receipt["decision"] == "rejected"


def test_runtime_invalid_response_preserves_executed_tools():
    from skillcompiler.model import ResponseContractError

    class Model:
        def ask(self, key, prompt, m, schema):
            if not m["history"]:
                return {
                    "action": "tool",
                    "id": "lookup",
                    "arguments": {"key": "item_0"},
                }
            raise ResponseContractError("Invalid JSON", {"choices": []})

    suite = Suite()
    task = suite.tasks("train")[0]
    result = Runtime(Model()).run(task, suite.open(task), Bank(), "t")
    assert result["terminal_reason"] == "response_contract_error"
    assert result["events"][0]["result"] == {"value": 3}


@pytest.mark.parametrize(
    "name",
    ["alfworld", "searchqa", "officeqa", "docvqa", "livemath", "spreadsheetbench"],
)
def test_distributed_bank_roundtrip(name, tmp_path):
    bank = Bank.load(Path(__file__).resolve().parents[1] / "benchmark" / name)
    before = bank.hash
    bank.write(tmp_path)
    assert Bank.load(tmp_path).hash == before == bank.hash
