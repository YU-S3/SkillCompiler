"""One execution loop for every task adapter."""

from copy import deepcopy
from .contracts import digest, validate
from .tools import ToolContext, resolve
from .model import ResponseContractError
from . import prompts

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "skeleton_id": {"type": "string"},
        "inputs": {"type": "object"},
        "reason": {"type": "string"},
    },
    "required": ["skeleton_id", "inputs"],
    "additionalProperties": False,
}
STEP_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"enum": ["tool", "program", "finish"]},
        "id": {"type": "string"},
        "arguments": {"type": "object"},
        "answer": {},
    },
    "required": ["action"],
    "additionalProperties": False,
}


class Runtime:
    def __init__(
        self, model, worker=None, max_steps=32, max_tool_calls=64, retrieval_limit=8
    ):
        self.model, self.worker = model, worker
        self.max_steps, self.max_tool_calls, self.retrieval_limit = (
            max_steps,
            max_tool_calls,
            retrieval_limit,
        )

    def program(self, bank, program_id, inputs, ctx):
        program = bank.programs[program_id]
        validate(inputs, program["input_schema"])
        if program.get("format", "python") == "ir":
            from .ir import execute

            result = execute(program, inputs, ctx)
        else:
            if self.worker is None:
                raise RuntimeError("Configure a container worker to use Programs")
            result = self.worker.execute(program, inputs, ctx)
        if result["status"] == "ok":
            validate(result["outputs"], program["output_schema"])
        ctx.events.append(
            {
                "type": "program",
                "id": program_id,
                "arguments": deepcopy(inputs),
                "result": deepcopy(result),
            }
        )
        return result

    def run(self, task, environment, bank, key):
        ctx = ToolContext(environment, self.max_tool_calls)
        try:
            return self._execute(task, ctx, bank, key)
        except ResponseContractError as exc:
            return {
                "task": task.public(),
                "answer": None,
                "events": ctx.events,
                "terminal_reason": "response_contract_error",
                "error": str(exc),
                "bank_hash": bank.hash,
            }

    def _execute(self, task, ctx, bank, key):
        retrieved = bank.retrieve(task.instruction, self.retrieval_limit)
        outputs = {}
        answer = None
        reason = "step_limit"
        steps = 0
        if retrieved["skeletons"]:
            plan = self.model.ask(
                key + "/plan",
                prompts.ANALYZER,
                {
                    "task": task.public(),
                    "observation": ctx.observe(),
                    "tools": ctx.available_tools(),
                    "skills": retrieved,
                },
                PLAN_SCHEMA,
            )
            if plan["skeleton_id"]:
                if plan["skeleton_id"] not in {s["id"] for s in retrieved["skeletons"]}:
                    raise ValueError("Planner selected an unknown skeleton")
                skeleton = bank.skeletons[plan["skeleton_id"]]
                skeleton_inputs = {
                    "goal": task.instruction,
                    **task.inputs,
                    **plan["inputs"],
                }
                for node in skeleton["nodes"]:
                    if steps >= self.max_steps:
                        break
                    steps += 1
                    try:
                        args = {
                            k: resolve(v, skeleton_inputs, outputs)
                            for k, v in node.get("args", {}).items()
                        }
                        pid = node.get("program_id")
                        if (
                            not pid
                            and node.get("skill_id")
                            and node.get("execution_mode") != "dynamic"
                        ):
                            available = bank.skills[node["skill_id"]].get(
                                "program_ids", []
                            )
                            pid = available[0] if available else None
                        if pid:
                            result = self.program(bank, pid, args, ctx)
                            if result["status"] != "ok":
                                break
                            outputs[node["id"]] = result["outputs"]
                        else:
                            required_outputs = [
                                ref.get("field", "").split(".")[0]
                                for ref in skeleton.get("outputs", {}).values()
                                if ref.get("from") == node["id"] and ref.get("field")
                            ]
                            schema = node.get(
                                "output_schema",
                                {"type": "object", "required": required_outputs},
                            )
                            guidance = [
                                bank.skills[sid]
                                for sid in node.get("reference_skill_ids", [])
                            ]
                            if node.get("skill_id"):
                                guidance.append(bank.skills[node["skill_id"]])
                            while steps <= self.max_steps:
                                decision = self.model.ask(
                                    key + "/node_" + node["id"] + "/" + str(steps),
                                    prompts.RUNTIME,
                                    {
                                        "task": task.public(),
                                        "operation": node.get("goal", node["id"]),
                                        "inputs": args,
                                        "skills": guidance,
                                        "observation": ctx.observe(),
                                        "tools": ctx.available_tools(),
                                        "history": ctx.events,
                                        "finish_output_schema": schema,
                                        "programs": [
                                            {
                                                k: v
                                                for k, v in p.items()
                                                if k not in {"source", "instructions"}
                                            }
                                            for p in bank.programs.values()
                                        ],
                                    },
                                    STEP_SCHEMA,
                                )
                                if decision["action"] == "finish":
                                    validate(decision.get("answer"), schema)
                                    outputs[node["id"]] = decision["answer"]
                                    break
                                if decision["action"] == "tool":
                                    ctx.call(
                                        decision["id"], decision.get("arguments", {})
                                    )
                                else:
                                    self.program(
                                        bank,
                                        decision["id"],
                                        decision.get("arguments", {}),
                                        ctx,
                                    )
                                steps += 1
                            if node["id"] not in outputs:
                                break
                        ctx.events.append(
                            {
                                "type": "skeleton_node",
                                "id": node["id"],
                                "outputs": outputs[node["id"]],
                            }
                        )
                    except (ValueError, KeyError, RuntimeError, TimeoutError) as exc:
                        ctx.events.append(
                            {
                                "type": "skeleton_error",
                                "id": node["id"],
                                "error": str(exc),
                            }
                        )
                        break
                if len(outputs) == len(skeleton["nodes"]) and skeleton.get("outputs"):
                    final = {
                        k: resolve(v, skeleton_inputs, outputs)
                        for k, v in skeleton["outputs"].items()
                    }
                    ctx.events.append(
                        {
                            "type": "skeleton_output",
                            "id": skeleton["id"],
                            "outputs": final,
                        }
                    )
                    if "answer" in final:
                        return {
                            "task": task.public(),
                            "answer": final["answer"],
                            "events": ctx.events,
                            "terminal_reason": "finished",
                            "bank_hash": bank.hash,
                        }
        for step in range(steps, self.max_steps):
            program_catalog = [
                {k: v for k, v in p.items() if k not in {"source", "instructions"}}
                for p in bank.programs.values()
            ]
            decision = self.model.ask(
                key + "/step_" + str(step),
                prompts.RUNTIME,
                {
                    "task": task.public(),
                    "observation": ctx.observe(),
                    "tools": ctx.available_tools(),
                    "skills": retrieved["skills"],
                    "programs": program_catalog,
                    "history": ctx.events,
                    "remaining_tool_calls": ctx.remaining_calls(),
                },
                STEP_SCHEMA,
            )
            action = decision["action"]
            if action == "finish":
                answer = decision.get("answer")
                reason = "finished"
                break
            try:
                if action == "tool":
                    ctx.call(decision["id"], decision.get("arguments", {}))
                elif action == "program":
                    self.program(
                        bank, decision["id"], decision.get("arguments", {}), ctx
                    )
            except (ValueError, KeyError, RuntimeError, TimeoutError) as exc:
                ctx.events.append(
                    {"type": "operation_error", "action": decision, "error": str(exc)}
                )
        return {
            "task": task.public(),
            "answer": answer,
            "events": ctx.events,
            "terminal_reason": reason,
            "bank_hash": bank.hash,
            "trace_hash_basis": "public input and observable execution only",
        }
