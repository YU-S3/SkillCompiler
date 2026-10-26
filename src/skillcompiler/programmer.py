"""On-demand compilation into a task-local overlay, including held-out execution."""

from copy import deepcopy
from .bank import Bank
from .contracts import digest, validate
from .model import ResponseContractError
from .learning import BUILD_SCHEMA
from . import prompts

OPERATION_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "minLength": 1},
        "instruction": {"type": "string", "minLength": 1},
        "input_schema": {"type": "object"},
        "output_schema": {"type": "object"},
        "allowed_tools": {
            "type": "array",
            "items": {"type": "string"},
            "uniqueItems": True,
        },
        "preconditions": {"type": "array", "items": {"type": "object"}},
        "effects": {"type": "array", "items": {"type": "object"}},
    },
    "required": [
        "name",
        "instruction",
        "input_schema",
        "output_schema",
        "allowed_tools",
    ],
    "additionalProperties": False,
}


class Programmer:
    def __init__(self, model, repair_limit=1):
        if repair_limit not in (0, 1):
            raise ValueError("Runtime structural repair is bounded to one")
        self.model, self.repair_limit = model, repair_limit

    def compile(self, spec, inputs, task, ctx, key):
        validate(spec, OPERATION_SCHEMA)
        validate(inputs, spec["input_schema"])
        tools = {t["name"]: t for t in ctx.available_tools()}
        if set(spec["allowed_tools"]) - set(tools):
            raise ValueError("Programmer requested tools outside the public interface")
        material = {
            "operation": deepcopy(spec),
            "inputs": deepcopy(inputs),
            "task": task.public(),
            "observation": ctx.observe(),
            "tools": [tools[k] for k in spec["allowed_tools"]],
            "history": deepcopy(ctx.events),
            "remaining_tool_calls": ctx.remaining_calls(),
            "scope": "task-local; structural repair only; live execution is not proof of general correctness",
        }
        previous, error = None, None
        for attempt in range(self.repair_limit + 1):
            received = False
            try:
                response = self.model.ask(
                    key + ("/build" if attempt == 0 else "/repair"),
                    prompts.PROGRAMMER if attempt == 0 else prompts.REPAIR,
                    (
                        material
                        if attempt == 0
                        else {
                            "original": material,
                            "previous_response": previous,
                            "errors": error,
                        }
                    ),
                    BUILD_SCHEMA,
                )
                received = True
                previous = response
                program = {
                    k: deepcopy(v) for k, v in spec.items() if k != "instruction"
                }
                program.update(format="python", source=response["source"])
                program["id"] = "local_" + digest([key, program])[:24]
                Bank(programs=[program])
                return program, {
                    "attempts": attempt + 1,
                    "request_key": key,
                    "source_hash": digest(program["source"]),
                    "admission": "structural",
                    "scope": "task_local",
                    "program": deepcopy(program),
                }
            except (ValueError, KeyError, SyntaxError, TypeError) as exc:
                if not received and not isinstance(exc, ResponseContractError):
                    raise
                if isinstance(exc, ResponseContractError):
                    previous = exc.response
                error = str(exc)
                if attempt == self.repair_limit:
                    raise ValueError(
                        "Programmer structural admission failed: " + error
                    ) from exc
