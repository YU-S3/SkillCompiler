"""A deterministic integration demo. It does not measure language-model quality."""

import argparse
import json
from skillcompiler.learning import Learner
from skillcompiler.pipeline import Pipeline
from skillcompiler.runtime import Runtime
from examples.key_value import Suite


class DemoModel:
    def ask(self, key, prompt, material, schema):
        if key.endswith("/extract"):
            episode = material["episodes"][0]
            return {
                "decision": "propose",
                "skills": [
                    {
                        "id": "lookup",
                        "name": "Read a named value",
                        "guidance": "Call lookup with the requested key and return the observed value.",
                    }
                ],
                "evidence": [{"episode": episode["task"]["id"], "events": [0]}],
            }
        calls = [event for event in material["history"] if event["type"] == "tool"]
        if calls:
            return {"action": "finish", "answer": calls[-1]["result"]["value"]}
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
