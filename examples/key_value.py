"""A synthetic tool environment; no external dataset or service."""

from skillcompiler.contracts import Task, Feedback


class Environment:
    def __init__(self, task):
        self.task = task
        self.values = {task.inputs["key"]: int(task.inputs["key"].split("_")[-1]) + 3}
        self.observed = {}

    def observe(self):
        return {"available_keys": list(self.values)}

    def tools(self):
        return [
            {
                "name": "lookup",
                "description": "Read a value by key",
                "input_schema": {
                    "type": "object",
                    "properties": {"key": {"type": "string"}},
                    "required": ["key"],
                    "additionalProperties": False,
                },
                "output_schema": {
                    "type": "object",
                    "properties": {"value": {"type": "integer"}},
                    "required": ["value"],
                    "additionalProperties": False,
                },
            }
        ]

    def call(self, name, arguments):
        if name != "lookup":
            raise ValueError("Unknown tool")
        self.observed[arguments["key"]] = self.values[arguments["key"]]
        return {"value": self.observed[arguments["key"]]}

    def predicate_contracts(self):
        return [
            {
                "predicate": "observed_value",
                "effect_domain": "evidence",
                "args": {"query": "input key", "result": "observed integer"},
                "description": "A lookup call has returned this value for this key.",
            }
        ]

    def verify_effects(self, effects, inputs, variables, result):
        def value(ref):
            if ref["kind"] in {"skill_input", "tool_input"}:
                return inputs[ref["source_role"]]
            if ref["kind"] in {"skill_output", "tool_output"}:
                return result["outputs"][ref["source_role"]]
            raise ValueError("Unsupported predicate role")

        for effect in effects:
            if (
                effect["predicate"] != "observed_value"
                or effect.get("effect_domain", "world") != "evidence"
                or effect.get("cardinality", 1) != 1
                or effect.get("distinct_by", "")
            ):
                return False
            query = value(effect["args"]["query"])
            if (
                query not in self.observed
                or value(effect["args"]["result"]) != self.observed[query]
            ):
                return False
        return True

    def verify_preconditions(self, conditions, inputs):
        return self.verify_effects(conditions, inputs, {}, {"outputs": {}})

    def close(self):
        pass


class Suite:
    def __init__(self, train_size=2, val_size=2, test_size=2):
        self.sizes = {"train": train_size, "val": val_size, "test": test_size}
        if any(type(n) is not int or n < 1 for n in self.sizes.values()):
            raise ValueError("Each split must contain at least one task")

    def tasks(self, split):
        return [
            Task(
                split + "_" + str(i),
                "Look up the requested key and return its value as a JSON integer, without an enclosing object.",
                {"key": f"item_{i}"},
            )
            for i in range(self.sizes[split])
        ]

    def open(self, task):
        return Environment(task)

    def score(self, task, answer, environment):
        success = answer == environment.values[task.inputs["key"]]
        return Feedback(float(success), success)
