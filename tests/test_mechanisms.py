"""Fabricated tasks that force each paper mechanism onto its intended path.

Eq. 3 utility ranking, Eq. 4 Env/Cert eligibility, Sec. 3.2.2 executability,
Sec. 3.2.4 isolated validation and replay repair, Sec. 3.4.1 failure
localization and Sec. 3.4.2 re-execution after an update.
"""

from copy import deepcopy
import os
import pytest
from skillcompiler.bank import Bank
from skillcompiler.contracts import read
from skillcompiler.learning import Learner, localize_failure, usage_statistics
from skillcompiler.pipeline import Pipeline
from skillcompiler.qualification import Ineligible, certificate
from skillcompiler.runtime import Runtime
from skillcompiler.tools import ToolContext
from skillcompiler.worker import ProgramWorker
from examples.key_value import Suite, Environment
from examples.offline_demo import DemoModel, lookup_analysis, lookup_contract

GOOD = "def run(ctx, inputs):\n    return {'status': 'ok', 'outputs': ctx.call('lookup', inputs)}\n"
# Runs, but reports a value that was never observed.
OFF_BY_ONE = (
    "def run(ctx, inputs):\n"
    "    value = ctx.call('lookup', inputs)['value']\n"
    "    return {'status': 'ok', 'outputs': {'value': value + 1}}\n"
)


class InProcessWorker:
    """Stands in for the container worker; executes the real Program source."""

    def __init__(self):
        self.runs = []

    def execute(self, program, inputs, ctx):
        self.runs.append(program["source"])
        scope = {}
        exec(program["source"], scope)
        return scope["run"](ctx, deepcopy(inputs))


def task(split="test", index=0):
    return Suite().tasks(split)[index]


def lookup_skill(sid="lookup", pid=None, **extra):
    s = {
        "id": sid,
        "name": "Read a named value",
        "guidance": "Call lookup with the requested key and return the observed value.",
        **lookup_contract(),
        **extra,
    }
    if pid:
        s["program_ids"] = [pid]
    return s


def reader(pid="reader", source=GOOD):
    return {
        "id": pid,
        "name": "Read",
        "format": "python",
        "allowed_tools": ["lookup"],
        "input_schema": lookup_contract()["input_schema"],
        "output_schema": lookup_contract()["output_schema"],
        "source": source,
    }


def certified(program, environment=None):
    tools = (environment or Environment(task())).tools()
    program["qualification"] = certificate(
        program, tools, [{"passed": True, "episode": "fixture"}]
    )
    return program


def kinds(trace, kind):
    return [e for e in trace["events"] if e["type"] == kind]


# Eq. 3 -------------------------------------------------------------------


def test_eq3_utility_and_active_flag_rank_after_contract_filter():
    a = lookup_skill("a", utility=0.9, active=False)
    b = lookup_skill("b", utility=0.5, active=True)
    c = lookup_skill("c")  # never used: utility 0, active by default
    bank = Bank([a, b, c])
    order = [s["id"] for s in bank.retrieve(task().instruction, 8)["skills"]]
    # Same Dice score: 0.05 + 0.05 > 0.09 > 0.05
    assert order == ["b", "a", "c"]
    with pytest.raises(ValueError):
        Bank([lookup_skill("x", utility=1.5)])
    with pytest.raises(ValueError):
        Bank([lookup_skill("x", usage={"attempts": 1, "successes": 2})])


def test_usage_is_host_measured_from_local_outcomes():
    episodes = [
        {
            "task": {"id": "train_0"},
            "events": [
                {"type": "skill", "id": "a", "status": "ok"},
                {"type": "skill", "id": "a", "status": "failed", "error": "x"},
                {"type": "skeleton_error", "id": "n", "guidance_skill_id": "b"},
                {"type": "skeleton_error", "id": "n", "guidance_skill_id": "b"},
            ],
            "feedback": {"success": True},
        }
    ]
    assert usage_statistics(episodes) == {
        "a": {"attempts": 2, "successes": 1},
        "b": {"attempts": 2, "successes": 0},
    }

    class Proposes:
        def ask(self, key, prompt, material, schema):
            # A model-claimed utility must be ignored.
            return {
                "decision": "propose",
                "skills": [lookup_skill("a", utility=1.0, active=True)],
                "evidence": [{"episode": "train_0", "events": [0]}],
            }

    bank = Bank([lookup_skill("a"), lookup_skill("b")])
    child, receipt = Learner(Proposes()).update(bank, episodes, Suite(), "k")
    assert receipt["decision"] == "proposed"
    assert child.skills["a"]["utility"] == 0.5 and child.skills["a"]["active"]
    assert child.skills["b"]["utility"] == 0.0 and not child.skills["b"]["active"]
    assert [s["id"] for s in child.retrieve(task().instruction, 8)["skills"]] == [
        "a",
        "b",
    ]


# Eq. 4 -------------------------------------------------------------------


def test_eq4_env_and_certificate_gate_program_reuse():
    t = task()
    runtime = Runtime(None, InProcessWorker(), planning_enabled=False)
    plain = Bank(programs=[reader()])
    with pytest.raises(Ineligible, match="no validation certificate"):
        runtime.program(plain, "reader", t.inputs, ToolContext(Environment(t)), True)
    # Body changed after certification.
    tampered = certified(reader())
    tampered["source"] = OFF_BY_ONE
    with pytest.raises(Ineligible, match="current Program body"):
        runtime.program(
            Bank(programs=[tampered]), "reader", t.inputs, ToolContext(Environment(t)), True
        )

    class Changed(Environment):
        def tools(self):
            tools = super().tools()
            tools[0]["output_schema"]["properties"]["value"]["type"] = "number"
            return tools

    class Missing(Environment):
        def tools(self):
            return []

    bank = Bank(programs=[certified(reader())])
    with pytest.raises(Ineligible, match="different tool interfaces"):
        runtime.program(bank, "reader", t.inputs, ToolContext(Changed(t)), True)
    ctx = ToolContext(Missing(t))
    with pytest.raises(Ineligible, match="lacks permitted tools: lookup"):
        runtime.program(bank, "reader", t.inputs, ctx)
    assert ctx.calls == 0
    ctx = ToolContext(Environment(t))
    assert runtime.program(bank, "reader", t.inputs, ctx, True)["outputs"] == {
        "value": 3
    }


def test_ineligible_binding_falls_back_to_agent_guidance():
    tampered = certified(reader())
    tampered["source"] = OFF_BY_ONE
    bank = Bank([lookup_skill(pid="reader")], [tampered])
    t = task()
    worker = InProcessWorker()
    trace = Runtime(DemoModel(), worker, allow_programmer=False).run(
        t, Environment(t), bank, "test/0"
    )
    assert trace["answer"] == 3 and not worker.runs
    assert kinds(trace, "implementation_ineligible")[0]["implementations"][0][
        "reason"
    ].startswith("Certificate does not match")
    fallback = kinds(trace, "implementation_fallback")[0]
    assert fallback["continuation"] == "agent_guided"
    node = kinds(trace, "skeleton_node")[0]
    assert node["execution"] == "agent_guided"


def test_failed_skill_execution_is_a_recorded_local_outcome():
    t = task()
    bank = Bank([lookup_skill(pid="reader")], [certified(reader(source=OFF_BY_ONE))])
    ctx = ToolContext(Environment(t))
    with pytest.raises(ValueError, match="effects were not verified"):
        Runtime(None, InProcessWorker(), planning_enabled=False).skill(
            bank, "lookup", t.inputs, ctx
        )
    event = kinds({"events": ctx.events}, "skill")[0]
    assert event["status"] == "failed"
    assert usage_statistics([{"events": ctx.events}]) == {
        "lookup": {"attempts": 1, "successes": 0}
    }


# Sec. 3.2.2 --------------------------------------------------------------


def test_executability_downgrades_unverifiable_operations():
    class Model(DemoModel):
        def __init__(self, tools):
            self.tools, self.keys = tools, []

        def ask(self, key, prompt, material, schema):
            self.keys.append(key)
            if key.endswith("/analyze"):
                a = lookup_analysis()
                a["operations"][0]["allowed_tools"] = self.tools
                return a
            return super().ask(key, prompt, material, schema)

    class NoPredicates(Environment):
        def predicate_contracts(self):
            return []

    t = task()
    for env, tools, reason in (
        (Environment(t), ["missing_tool"], "unknown tools: missing_tool"),
        (NoPredicates(t), ["lookup"], "predicates not verifiable"),
    ):
        m = Model(tools)
        trace = Runtime(m, InProcessWorker()).run(t, env, Bank(), "test/0")
        mode = kinds(trace, "task_analysis")[0]["executability"][0]
        assert mode["requested"] == "programmatic" and mode["mode"] == "agent"
        assert any(reason in r for r in mode["reasons"])
        node = kinds(trace, "skeleton_construction")[0]["skeleton"]["nodes"][0]
        assert node["execution_mode"] == "dynamic"
        # Agent-guided: no Programmer request, the answer comes from the agent.
        assert trace["answer"] == 3 and not any("/programmer/" in k for k in m.keys)


# Sec. 3.2.4: runtime Programmer with isolated validation ------------------


class ProgrammerModel(DemoModel):
    def __init__(self, sources):
        self.sources, self.materials = sources, {}

    def ask(self, key, prompt, material, schema):
        if key.endswith("/build"):
            self.materials[key] = deepcopy(material)
            return {"source": self.sources.pop(0)}
        return super().ask(key, prompt, material, schema)


def test_programmatic_node_compiles_validates_in_sandbox_and_repairs():
    t = task()
    live, opened = Environment(t), []

    def sandbox():
        opened.append(Environment(t))
        return opened[-1]

    m = ProgrammerModel([OFF_BY_ONE, GOOD])
    bank = Bank()
    trace = Runtime(m, InProcessWorker()).run(t, live, bank, "test/0", sandbox=sandbox)
    assert trace["answer"] == 3 and not bank.programs and trace["programmer_calls"] == 1
    checks = kinds(trace, "runtime_validation")
    assert [c["passed"] for c in checks] == [False, True]
    assert "effects were not verified" in checks[0]["error"]
    repair = m.materials["test/0/node_read/programmer/validation_repair_1/build"]
    assert repair["validation_feedback"]["previous_source"] == OFF_BY_ONE
    receipt = kinds(trace, "runtime_programmer")[0]
    assert receipt["admission"] == "isolated_validation"
    assert receipt["validation_attempts"] == 2
    assert receipt["program"]["qualification"]["scope"] == "task_local"
    assert kinds(trace, "skeleton_node")[0]["execution"] == "implementation"
    # The faulty candidate only ever touched isolated instances.
    assert len(opened) == 2
    assert [e["type"] for e in trace["events"]].count("tool") == 1


def test_sandbox_validation_and_repair_in_real_container():
    image = os.environ.get("SKILLCOMPILER_TEST_IMAGE")
    if not image:
        pytest.skip("Set SKILLCOMPILER_TEST_IMAGE for actual Docker")
    t = task()
    m = ProgrammerModel([OFF_BY_ONE, GOOD])
    trace = Runtime(m, ProgramWorker(image)).run(
        t, Environment(t), Bank(), "test/0", sandbox=lambda: Environment(t)
    )
    assert trace["answer"] == 3
    assert [c["passed"] for c in kinds(trace, "runtime_validation")] == [False, True]
    assert kinds(trace, "runtime_qualification")[0]["status"] == "ok"


def test_exhausted_validation_falls_back_without_touching_live_state():
    t = task()
    m = ProgrammerModel([OFF_BY_ONE, OFF_BY_ONE])
    trace = Runtime(m, InProcessWorker()).run(
        t, Environment(t), Bank(), "test/0", sandbox=lambda: Environment(t)
    )
    assert trace["answer"] == 3
    assert [c["passed"] for c in kinds(trace, "runtime_validation")] == [False, False]
    fallback = kinds(trace, "implementation_fallback")[0]
    assert "Isolated validation failed" in fallback["reason"]
    assert not kinds(trace, "runtime_programmer") and not kinds(trace, "program")
    assert kinds(trace, "skeleton_node")[0]["execution"] == "agent_guided"


def test_native_programmer_replays_live_prefix_in_sandbox():
    class Model:
        def ask(self, key, prompt, m, schema):
            if key.endswith("/build"):
                return {"source": GOOD}
            done = [e for e in m["history"] if e["type"] == "program"]
            if done:
                return {"action": "finish", "answer": done[-1]["result"]["outputs"]["value"]}
            if not any(e["type"] == "tool" for e in m["history"]):
                return {"action": "tool", "id": "lookup", "arguments": m["task"]["inputs"]}
            return {
                "action": "programmer",
                "operation": {
                    "name": "Read",
                    "instruction": "Read the key.",
                    "allowed_tools": ["lookup"],
                    **lookup_contract(),
                },
                "arguments": m["task"]["inputs"],
            }

    t = task()
    trace = Runtime(Model(), InProcessWorker(), planning_enabled=False).run(
        t, Environment(t), Bank(), "test/0", sandbox=lambda: Environment(t)
    )
    assert trace["answer"] == 3
    check = kinds(trace, "runtime_validation")[0]
    assert check["passed"] and check["replayed_prefix"] == 1
    # Without an isolated instance only structural admission applies.
    trace = Runtime(Model(), InProcessWorker(), planning_enabled=False).run(
        t, Environment(t), Bank(), "test/0"
    )
    assert trace["answer"] == 3 and not kinds(trace, "runtime_validation")
    assert kinds(trace, "runtime_programmer")[0]["admission"] == "structural"


# Sec. 3.2.4: training replay qualification, repair and certification -----


def training_episodes():
    suite = Suite()
    rows = []
    for t in suite.tasks("train"):
        env = suite.open(t)
        trace = Runtime(DemoModel(), planning_enabled=False).run(t, env, Bank(), "train")
        rows.append({**trace, "feedback": vars(suite.score(t, trace["answer"], env))})
    return rows


def test_learner_repairs_replay_failure_and_certifies_program():
    episodes = training_episodes()

    class Model:
        def __init__(self):
            self.materials = {}

        def ask(self, key, prompt, material, schema):
            self.materials[key] = deepcopy(material)
            if key.endswith("/build_reader"):
                return {"source": OFF_BY_ONE}
            if key.endswith("/build_reader/repair_1"):
                return {"source": GOOD}
            spec = {k: v for k, v in reader().items() if k not in {"source", "format"}}
            spec["trials"] = [
                {
                    "episode": e["task"]["id"],
                    "inputs": e["task"]["inputs"],
                    "prefix": [],
                    "expected_from": {"event": 0, "path": ["result"]},
                }
                for e in episodes
            ]
            return {
                "decision": "propose",
                "skills": [lookup_skill(pid="reader")],
                "programs": [spec],
                "evidence": [{"episode": e["task"]["id"], "events": [0]} for e in episodes],
            }

    m, worker = Model(), InProcessWorker()
    child, receipt = Learner(m, worker).update(Bank(), episodes, Suite(), "learn")
    assert receipt["decision"] == "proposed", receipt
    failure = m.materials["learn/build_reader/repair_1"]["replay_failure"]
    assert failure["stage"] == "replay" and failure["expected"] == {"value": 3}
    assert failure["observed"]["outputs"] == {"value": 4}
    assert m.materials["learn/build_reader/repair_1"]["previous_source"] == OFF_BY_ONE
    record = child.programs["reader"]["qualification"]
    assert record["scope"] == "persistent" and len(record["trials"]) == 2
    # The certified Program is now reused automatically by a programmatic node.
    t = task()
    trace = Runtime(DemoModel(), worker).run(t, Environment(t), child, "test/0")
    assert trace["answer"] == 3
    assert kinds(trace, "program")[0]["id"] == "reader"
    assert kinds(trace, "skeleton_node")[0]["execution"] == "implementation"
    # The same body re-proposed is a certification, not a new revision.
    again, _ = Learner(m, worker).update(child, episodes, Suite(), "learn2")
    assert again.programs["reader"]["source"] == GOOD


def test_unrepairable_program_is_rejected_after_bounded_repairs():
    episodes = training_episodes()

    class Model:
        keys = []

        def ask(self, key, prompt, material, schema):
            self.keys.append(key)
            if "/build_reader" in key:
                return {"source": OFF_BY_ONE}
            spec = {k: v for k, v in reader().items() if k not in {"source", "format"}}
            spec["trials"] = [
                {
                    "episode": episodes[0]["task"]["id"],
                    "inputs": episodes[0]["task"]["inputs"],
                    "expected_from": {"event": 0, "path": ["result"]},
                }
            ]
            return {
                "decision": "propose",
                "programs": [spec],
                "evidence": [{"episode": episodes[0]["task"]["id"], "events": [0]}],
            }

    m = Model()
    child, receipt = Learner(
        m, InProcessWorker(), repair_limit=0, program_repair_limit=2
    ).update(Bank(), episodes, Suite(), "learn")
    assert receipt["decision"] == "rejected"
    assert "not qualified after 2 replay repairs" in receipt["error"]
    assert [k for k in m.keys if "/build_reader" in k] == [
        "learn/build_reader",
        "learn/build_reader/repair_1",
        "learn/build_reader/repair_2",
    ]
    assert not child.programs


# Sec. 3.4.1 / 3.4.2 ------------------------------------------------------


def test_failure_localization_names_the_first_failing_operation():
    node = {
        "id": "read",
        "skill_id": "lookup",
        "execution_mode": "programmatic",
        "contract": lookup_contract(),
    }
    episode = {
        "task": {"id": "train_0"},
        "events": [
            {"type": "skeleton_construction", "skeleton": {"nodes": [node]}},
            {"type": "tool", "name": "lookup"},
            {"type": "skeleton_error", "id": "read", "error": "boom"},
            {"type": "operation_error", "action": {"action": "tool"}, "error": "late"},
        ],
        "feedback": {"success": False},
    }
    found = localize_failure(episode)
    assert found["level"] == "skeleton_node" and found["event"] == 2
    assert found["binding"] == {"skill_id": "lookup", "execution_mode": "programmatic"}
    assert found["contract"] == lookup_contract()
    outcome = localize_failure(
        {"task": {"id": "t"}, "events": [], "feedback": {"success": False}}
    )
    assert outcome["level"] == "task_outcome"
    assert localize_failure({"task": {"id": "t"}, "events": [], "feedback": {"success": True}}) is None


def test_pipeline_reexecutes_failed_training_tasks_after_update(tmp_path):
    class Model:
        def __init__(self):
            self.extract = {}

        def ask(self, key, prompt, m, schema):
            if key.endswith("/extract"):
                self.extract[key] = deepcopy(m)
                if m["bank"]["skills"]:
                    return {"decision": "no_change"}
                return {
                    "decision": "propose",
                    "skills": [lookup_skill()],
                    "evidence": [
                        {"episode": e["task"]["id"], "events": [0]} for e in m["episodes"]
                    ],
                }
            calls = [e for e in m["history"] if e["type"] == "tool"]
            if not calls:
                return {"action": "tool", "id": "lookup", "arguments": m["task"]["inputs"]}
            # Without a Skill the agent misreads the result (a fabricated failure).
            value = calls[-1]["result"]["value"]
            return {"action": "finish", "answer": value if m["skills"] else -1}

    m = Model()
    result = Pipeline(
        Suite(),
        Runtime(m, planning_enabled=False),
        Learner(m),
        tmp_path,
        batch_size=2,
        refinement_rounds=0,
    ).run()
    assert result["test"]["successes"] == 2
    first = read(tmp_path / "train/0/summary.json")
    retried = read(tmp_path / "train/0/replan_0/summary.json")
    assert first["successes"] == 0 and retried["successes"] == 2
    assert retried["bank_hash"] != first["bank_hash"]
    episodes = m.extract["learning/0/replan_0/extract"]["episodes"]
    assert all(e["feedback"]["success"] for e in episodes)
    previous = episodes[0]["previous_attempt"]
    assert previous["bank_hash"] == first["bank_hash"]
    assert previous["failure_localization"]["level"] == "task_outcome"
    assert (tmp_path / "learning/0/replan_0/receipt.json").exists()
    # Successful tasks are not re-executed again.
    assert not (tmp_path / "train/0/replan_1").exists()
