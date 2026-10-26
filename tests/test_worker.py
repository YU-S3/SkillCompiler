"""Opt-in real container tests: set SKILLCOMPILER_TEST_IMAGE to a local digest."""

from pathlib import Path
import os
import pytest
from skillcompiler.bank import Bank
from skillcompiler.worker import ProgramWorker
from skillcompiler.tools import ToolContext
from skillcompiler.learning import Learner
from examples.key_value import Suite


@pytest.fixture
def worker():
    image = os.environ.get("SKILLCOMPILER_TEST_IMAGE")
    if not image:
        pytest.skip("Set SKILLCOMPILER_TEST_IMAGE to run Docker integration tests")
    return ProgramWorker(image, timeout=30)


class PublicEnvironment:
    def observe(self):
        return {}

    def tools(self):
        return [{"name": "read_result", "input_schema": {"type": "object"}}]

    def call(self, name, args):
        if name != "read_result" or args["result_id"] != "observed":
            raise ValueError("Unknown acquired result")
        return {"accepted": True, "data": 0.25, "next_offset": None}


CHOICES = [
    {"label": "A", "text": "For every positive x, x squared is positive."},
    {"label": "B", "text": "A stronger result can be proven."},
]
SOURCE = {
    "question": "Who wrote the example?",
    "context": "[DOC] [TLE] Example [PAR] The example was written by Ada.",
}
CASES = [
    ("searchqa", "program_01", SOURCE),
    ("searchqa", "program_02", SOURCE),
    (
        "searchqa",
        "program_03",
        {**SOURCE, "answer": "Ada", "quotes": ["The example was written by Ada."]},
    ),
    ("searchqa", "program_04", {"answer": "Ada"}),
    ("docvqa", "program_01", {"kind": "count", "value": "N = 12"}),
    ("livemath", "program_01", {"choices": CHOICES}),
    (
        "livemath",
        "program_02",
        {
            "labels": ["A", "B"],
            "meta_labels": ["B"],
            "meta_count": 1,
            "meta_label": "B",
        },
    ),
    ("livemath", "program_03", {"choices": CHOICES}),
    (
        "officeqa",
        "program_01",
        {
            "submit_answer": True,
            "value_ref": {"result_id": "observed", "path": ["value"]},
            "source_scale": "fraction",
            "unit": "ratio",
            "format": {"output_scale": "percent", "style": "percent", "decimals": 2},
        },
    ),
    (
        "officeqa",
        "program_02",
        {
            "operation": "midpoint_submit",
            "submit_answer": True,
            "values": [10, 20],
            "unit": "count",
            "decimals": [2],
            "absolute": False,
            "output_percent": True,
        },
    ),
    (
        "officeqa",
        "program_03",
        {
            "operation": "growth_submit",
            "submit_answer": True,
            "values": [10, 20],
            "unit": "count",
            "decimals": [3, 3, 3],
            "intervals": 2,
        },
    ),
    (
        "officeqa",
        "program_04",
        {
            "operation": "paired_statistics",
            "submit_answer": True,
            "values": [10, 20, 30],
            "other_values": [20, 40, 60],
            "periods": ["a", "b", "c"],
            "expected_periods": ["a", "b", "c"],
            "statistic": "share_and_correlation",
            "unit": "count",
            "decimals": [2, 2],
        },
    ),
    (
        "officeqa",
        "program_05",
        {
            "operation": "baseline_z",
            "submit_answer": True,
            "values": [1, 2, 3],
            "unit": "count",
            "decimals": [2],
            "spread": "population",
            "target": 4,
            "threshold": 2,
            "two_sided": True,
        },
    ),
]


@pytest.mark.parametrize("name,pid,inputs", CASES)
def test_exported_program(worker, name, pid, inputs):
    bank = Bank.load(Path(__file__).resolve().parents[1] / "benchmark" / name)
    result = worker.execute(
        bank.programs[pid], inputs, ToolContext(PublicEnvironment())
    )
    assert result["status"] == "ok"
    if name == "docvqa":
        assert result["outputs"]["answer"] == "12"
    if name == "livemath" and pid == "program_02":
        assert result["outputs"]["answer"] == "B"
    if name == "officeqa" and pid == "program_01":
        assert result["outputs"]["answer"] == "25%"


def test_real_learner_builder_replay(worker):
    suite = Suite()
    task = suite.tasks("train")[0]
    ctx = ToolContext(suite.open(task))
    ctx.call("lookup", {"key": "item_0"})
    episode = {
        "task": task.public(),
        "events": ctx.events,
        "feedback": {"score": 1, "success": True},
    }

    class Model:
        def ask(self, key, prompt, m, schema):
            if "/build_" in key:
                return {
                    "source": "def run(ctx,inputs):\n    return {'status':'ok','outputs':ctx.call('lookup',inputs)}\n"
                }
            return {
                "decision": "propose",
                "evidence": [{"episode": task.id, "events": [0]}],
                "skills": [
                    {
                        "id": "s",
                        "name": "Lookup",
                        "guidance": "Read the caller-selected key.",
                        "program_ids": ["p"],
                    }
                ],
                "programs": [
                    {
                        "id": "p",
                        "name": "Lookup",
                        "allowed_tools": ["lookup"],
                        "input_schema": {"type": "object"},
                        "output_schema": {"type": "object"},
                        "trials": [
                            {
                                "episode": task.id,
                                "inputs": {"key": "item_0"},
                                "prefix": [],
                                "expected_from": {"event": 0, "path": ["result"]},
                            }
                        ],
                    }
                ],
            }

    bank, receipt = Learner(Model(), worker).update(Bank(), [episode], suite, "learn")
    assert len(bank.programs) == 1 and receipt["qualification"][0]["passed"]
    # The same source is rejected when its tool permission is removed.
    p = bank.programs["p"]
    p["allowed_tools"] = []
    with pytest.raises(RuntimeError, match="permissions"):
        worker.execute(p, {"key": "item_0"}, ToolContext(suite.open(task)))


def test_container_isolation(worker):
    p = {
        "id": "p",
        "input_schema": {},
        "output_schema": {},
        "allowed_tools": [],
        "source": """import os, socket
def run(ctx,inputs):
    try:
        open('/program/source.py','w').write('changed')
        writable=True
    except OSError: writable=False
    try:
        socket.create_connection(('1.1.1.1',443),timeout=.2)
        network=True
    except OSError: network=False
    return {'status':'ok','outputs':{'writable':writable,'network':network,'uid':os.getuid()}}
""",
    }
    r = worker.execute(p, {}, ToolContext(PublicEnvironment()))
    assert r["outputs"] == {"writable": False, "network": False, "uid": 1000}
