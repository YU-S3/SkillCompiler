"""A synthetic tool environment; no external dataset or service."""

from skillcompiler.contracts import Task, Feedback


class Environment:
    def __init__(self, task):
        self.task = task
        self.values = {task.inputs["key"]: int(task.inputs["key"].split("_")[-1]) + 3}

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
            }
        ]

    def call(self, name, arguments):
        if name != "lookup":
            raise ValueError("Unknown tool")
        return {"value": self.values[arguments["key"]]}

    def close(self):
        pass


class Suite:
    def tasks(self, split):
        return [
            Task(
                split + "_" + str(i),
                "Look up the requested key and return its value.",
                {"key": f"item_{i}"},
            )
            for i in range(2)
        ]

    def open(self, task):
        return Environment(task)

    def score(self, task, answer, environment):
        success = answer == environment.values[task.inputs["key"]]
        return Feedback(float(success), success)
