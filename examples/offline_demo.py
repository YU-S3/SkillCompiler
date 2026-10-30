"""A deterministic integration demo. It does not measure language-model quality."""

import argparse
import json
from copy import deepcopy
from skillcompiler.learning import Learner
from skillcompiler.pipeline import Pipeline
from skillcompiler.runtime import Runtime
from examples.key_value import Suite


def object_schema(role, kind):
    return {
        "type": "object",
        "properties": {role: {"type": kind}},
        "required": [role],
        "additionalProperties": False,
    }


def lookup_contract(input_role="key", output_role="value"):
    return {
        "input_schema": object_schema(input_role, "string"),
        "output_schema": object_schema(output_role, "integer"),
        "preconditions": [],
        "effects": [
            {
                "predicate": "observed_value",
                "effect_domain": "evidence",
                "args": {
                    "query": {"kind": "skill_input", "source_role": input_role},
                    "result": {"kind": "skill_output", "source_role": output_role},
                },
            }
        ],
    }


def lookup_analysis():
    return {
        "decision": "plan",
        "contract": lookup_contract(output_role="answer"),
        "operations": [
            {
                "id": "read",
                "intent": "Read the requested key",
                "contract": lookup_contract(),
                "args": {"key": {"task": "key"}},
                "after": [],
            }
        ],
        "outputs": {"answer": {"from": "read", "field": "value"}},
    }


class DemoModel:
    def ask(self, key, prompt, material, schema):
        if key.endswith("/analyze"):
            return lookup_analysis()
        if key.endswith("/construct"):
            return {
                "skeleton_id": "",
                "choices": {
                    oid: rows[0]["id"] if rows else ""
                    for oid, rows in material["compatible_operations"].items()
                },
            }
        if key.endswith("/extract"):
            episode = material["episodes"][0]
            return {
                "decision": "propose",
                "skills": [
                    {
                        "id": "lookup",
                        "name": "Read a named value",
                        "guidance": "Call lookup with the requested key and return the observed value.",
                        **lookup_contract(),
                    }
                ],
                "evidence": [{"episode": episode["task"]["id"], "events": [0]}],
            }
        calls = [event for event in material["history"] if event["type"] == "tool"]
        if calls:
            answer = (
                calls[-1]["result"] if "/node_" in key else calls[-1]["result"]["value"]
            )
            return {"action": "finish", "answer": answer}
        return {
            "action": "tool",
            "id": "lookup",
            "arguments": {"key": material["task"]["inputs"]["key"]},
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="runs/offline-demo")
    args = parser.parse_args()
    model = DemoModel()
    result = Pipeline(
        Suite(), Runtime(model), Learner(model), args.output, batch_size=1
    ).run()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
