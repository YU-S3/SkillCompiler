from skillcompiler.ir import execute
from skillcompiler.tools import ToolContext
import pytest


class Environment:
    def __init__(self):
        self.calls = []

    def tools(self):
        return [{"name": "visit", "input_schema": {"type": "object"}}]

    def call(self, name, args):
        self.calls.append(args["item"])
        return {"visited": args["item"]}

    def verify_effects(self, effects, inputs, variables, result):
        return result.get("visited") in self.calls


def program():
    return {
        "allowed_tools": ["visit"],
        "max_actions": 2,
        "instructions": [
            {
                "op": "FOR_EACH",
                "iteration_variable": "item",
                "max_iterations": 2,
                "collection_source": {"source": "tool_input", "field": "items"},
                "body": [
                    {
                        "op": "ACTION",
                        "action_type": "visit",
                        "argument_mapping": {
                            "item": {"source": "local_variable", "field": "item"}
                        },
                    },
                    {
                        "op": "RETURN",
                        "output_sources": {
                            "visited": {"source": "local_variable", "field": "item"}
                        },
                    },
                ],
            }
        ],
        "final_effects": [{"predicate": "visited"}],
    }


def test_bounded_ir_and_exact_variable():
    env = Environment()
    r = execute(program(), {"items": ["a", "b"]}, ToolContext(env))
    assert r == {"status": "ok", "outputs": {"visited": "a"}} and env.calls == ["a"]


def test_ir_permissions_and_effects():
    p = program()
    p["allowed_tools"] = []
    with pytest.raises(ValueError, match="permissions"):
        execute(p, {"items": ["a"]}, ToolContext(Environment()))
    env = Environment()
    env.verify_effects = lambda *a: False
    with pytest.raises(ValueError, match="effects"):
        execute(program(), {"items": ["a"]}, ToolContext(env))
