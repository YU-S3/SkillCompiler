"""Bounded public tool access shared by the agent and isolated Programs."""

from .contracts import validate


class ToolContext:
    def __init__(self, environment, limit=64):
        self.environment = environment
        self.limit = limit
        self.events = []
        self.calls = 0

    def observe(self):
        return self.environment.observe()

    def available_tools(self):
        return self.environment.tools()

    def remaining_calls(self):
        return max(0, self.limit - self.calls)

    def call(self, name, arguments):
        if not self.remaining_calls():
            raise RuntimeError("Native tool call budget exhausted")
        specs = {t["name"]: t for t in self.available_tools()}
        if name not in specs:
            raise ValueError("Unknown public tool: " + name)
        validate(arguments, specs[name].get("input_schema", {"type": "object"}))
        self.calls += 1
        try:
            result = self.environment.call(name, arguments)
            if "output_schema" in specs[name]:
                validate(result, specs[name]["output_schema"])
            self.events.append(
                {"type": "tool", "name": name, "arguments": arguments, "result": result}
            )
            return result
        except Exception as exc:
            self.events.append(
                {
                    "type": "tool_error",
                    "name": name,
                    "arguments": arguments,
                    "error": str(exc),
                }
            )
            raise

    def read_result(self, result_id, offset=0, limit=None, path=None):
        if not self.remaining_calls():
            raise RuntimeError("Native tool call budget exhausted")
        reader = getattr(self.environment, "read_result", None)
        if reader is None:
            raise ValueError("Environment has no result reader")
        self.calls += 1
        result = reader(result_id, offset=offset, limit=limit, path=path)
        self.events.append(
            {
                "type": "read_result",
                "result_id": result_id,
                "offset": offset,
                "limit": limit,
                "path": path,
                "result": result,
            }
        )
        return result


def resolve(value, inputs, outputs):
    """Explicit references; no eval, string interpolation, or hidden task inference."""
    if isinstance(value, list):
        return [resolve(v, inputs, outputs) for v in value]
    if not isinstance(value, dict):
        return value
    if set(value) == {"task"}:
        return inputs[value["task"]]
    if "from" in value:
        result = outputs[value["from"]]
        for part in str(value.get("field", "")).split("."):
            if part:
                result = result[int(part)] if isinstance(result, list) else result[part]
        return result
    if set(value) == {"literal"}:
        return value["literal"]
    if "unresolved" in value:
        raise ValueError("Unresolved input: " + str(value["unresolved"]))
    return {k: resolve(v, inputs, outputs) for k, v in value.items()}
