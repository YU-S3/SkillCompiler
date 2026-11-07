"""One execution loop for every task adapter."""

from copy import deepcopy
from .contracts import digest, validate
from .tools import ToolContext, resolve
from .model import BudgetExceeded, ResponseContractError
from . import prompts
from .programmer import Programmer, OPERATION_SCHEMA
from .qualification import (
    Ineligible,
    certificate,
    certificate_problem,
    environment_problem,
)
from .semantic import implementations, implementation_maps

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
        "action": {"enum": ["tool", "skill", "program", "programmer", "finish"]},
        "id": {"type": "string"},
        "arguments": {"type": "object"},
        "answer": {},
        "operation": OPERATION_SCHEMA,
    },
    "required": ["action"],
    "additionalProperties": False,
}
EXECUTION_ERRORS = (ValueError, KeyError, RuntimeError, TimeoutError)


def _provider_failure(exc):
    """Provider and ledger failures end the run instead of becoming outcomes."""
    return isinstance(exc, BudgetExceeded) or getattr(exc, "provider_failure", False)


class Runtime:
    def __init__(
        self,
        model,
        worker=None,
        max_steps=32,
        max_tool_calls=64,
        retrieval_limit=8,
        allow_programmer=True,
        max_programs_per_task=2,
        programmer_repair_limit=1,
        planning_enabled=True,
        planning_repair_limit=1,
        validation_repair_limit=1,
    ):
        self.model, self.worker = model, worker
        self.max_steps, self.max_tool_calls, self.retrieval_limit = (
            max_steps,
            max_tool_calls,
            retrieval_limit,
        )
        if max_programs_per_task < 0:
            raise ValueError("Programmer task limit cannot be negative")
        if validation_repair_limit not in range(4):
            raise ValueError("Isolated validation repair is bounded to three")
        self.allow_programmer = allow_programmer
        self.max_programs_per_task = max_programs_per_task
        self.programmer_repair_limit = programmer_repair_limit
        self.validation_repair_limit = validation_repair_limit
        self.programmer = Programmer(model, programmer_repair_limit)
        from .planning import Planner

        self.planning_enabled = planning_enabled
        self.planning_repair_limit = planning_repair_limit
        self.planner = Planner(model, retrieval_limit, planning_repair_limit)

    @staticmethod
    def _preflight(asset, inputs, ctx):
        if asset.get("preconditions") or asset.get("entry_constraints"):
            verifier = getattr(ctx.environment, "verify_preconditions", None)
            conditions = asset.get("preconditions", []) + asset.get(
                "entry_constraints", []
            )
            if (
                verifier is None
                or verifier(deepcopy(conditions), deepcopy(inputs)) is not True
            ):
                raise ValueError("Preconditions are unresolved or not satisfied")

    @staticmethod
    def _effects(asset, inputs, result, ctx):
        if result["status"] == "ok" and asset.get("validator"):
            verifier = getattr(ctx.environment, "verify_contract", None)
            if (
                verifier is None
                or verifier(
                    deepcopy(asset["validator"]), deepcopy(inputs), deepcopy(result)
                )
                is not True
            ):
                raise ValueError("Semantic validator did not verify the result")
        if result["status"] == "ok" and asset.get("effects"):
            verifier = getattr(ctx.environment, "verify_effects", None)
            if (
                verifier is None
                or verifier(
                    deepcopy(asset["effects"]), deepcopy(inputs), {}, deepcopy(result)
                )
                is not True
            ):
                raise ValueError("Declared effects were not verified")

    @staticmethod
    def _ineligible(program, ctx, certified):
        """Env and (for automatic reuse) Cert of Eq. 4; None when eligible."""
        return environment_problem(program, ctx) or (
            certificate_problem(program, ctx) if certified else None
        )

    def _admit(self, program, inputs, ctx):
        """Bind and Pre of Eq. 4, checked before any side effect."""
        try:
            validate(inputs, program["input_schema"])
            self._preflight(program, inputs, ctx)
        except (ValueError, KeyError) as exc:
            raise Ineligible(str(exc)) from exc

    def _invoke(self, program, inputs, ctx):
        if program.get("format", "python") == "ir":
            from .ir import execute

            result = execute(program, inputs, ctx)
        else:
            if self.worker is None:
                raise RuntimeError("Configure a container worker to use Programs")
            result = self.worker.execute(program, inputs, ctx)
        if result["status"] == "ok":
            validate(result["outputs"], program["output_schema"])
        self._effects(program, inputs, result, ctx)
        return result

    def skill(self, bank, skill_id, inputs, ctx, preferred=None):
        """Automatic invocation needs a unique eligible (or preferred) implementation."""
        skill = bank.skills[skill_id]
        try:
            validate(inputs, skill.get("input_schema", {"type": "object"}))
            self._preflight(skill, inputs, ctx)
        except (ValueError, KeyError) as exc:
            raise Ineligible(str(exc)) from exc
        ready, rejected = [], []
        for binding in implementations(skill):
            pid = binding["program_id"]
            if preferred and pid != preferred:
                continue
            program = bank.programs[pid]
            try:
                imap, omap = implementation_maps(skill, program, binding)
                args = (
                    {p: inputs[s] for p, s in imap.items() if s in inputs}
                    if imap
                    else deepcopy(inputs)
                )
                self._admit(program, args, ctx)
                problem = self._ineligible(program, ctx, certified=True)
                if problem:
                    raise Ineligible(problem)
                item = (pid, args, omap)
                if item not in ready:
                    ready.append(item)
            except (ValueError, KeyError) as exc:
                rejected.append({"program_id": pid, "reason": str(exc)})
        if rejected:
            ctx.events.append(
                {
                    "type": "implementation_ineligible",
                    "skill_id": skill_id,
                    "implementations": rejected,
                }
            )
        if len(ready) != 1:
            raise Ineligible(
                "No ready implementation"
                + "".join("; " + r["program_id"] + ": " + r["reason"] for r in rejected)
                if not ready
                else "Ambiguous implementations require an explicit Program choice"
            )
        pid, args, mapping = ready[0]
        try:
            result = deepcopy(self.program(bank, pid, args, ctx))
            if result["status"] == "ok":
                result["outputs"] = (
                    {
                        s: result["outputs"][p]
                        for s, p in mapping.items()
                        if p in result["outputs"]
                    }
                    if mapping
                    else result["outputs"]
                )
                validate(
                    result["outputs"], skill.get("output_schema", {"type": "object"})
                )
                self._effects(skill, inputs, result, ctx)
        except EXECUTION_ERRORS as exc:
            # A local operation outcome, recorded for Eq. 3 utility.
            ctx.events.append(
                {
                    "type": "skill",
                    "id": skill_id,
                    "program_id": pid,
                    "arguments": deepcopy(inputs),
                    "status": "failed",
                    "error": str(exc),
                }
            )
            raise
        ctx.events.append(
            {
                "type": "skill",
                "id": skill_id,
                "program_id": pid,
                "arguments": deepcopy(inputs),
                "status": "ok" if result["status"] == "ok" else "failed",
                "result": deepcopy(result),
            }
        )
        return result

    def program(self, bank, program_id, inputs, ctx, require_certificate=False):
        program = bank.programs[program_id]
        self._admit(program, inputs, ctx)
        problem = self._ineligible(program, ctx, require_certificate)
        if problem:
            raise Ineligible(problem)
        start = len(ctx.events)
        try:
            result = self._invoke(program, inputs, ctx)
        except EXECUTION_ERRORS as exc:
            ctx.events.append(
                {
                    "type": "program_error",
                    "id": program_id,
                    "arguments": deepcopy(inputs),
                    "error": str(exc),
                    "tool_event_start": start,
                }
            )
            raise
        ctx.events.append(
            {
                "type": "program",
                "id": program_id,
                "arguments": deepcopy(inputs),
                "result": deepcopy(result),
                "tool_event_start": start,
            }
        )
        return result

    def run(self, task, environment, bank, key, sandbox=None):
        """sandbox() opens a fresh instance of the same task for isolated validation."""
        ctx = ToolContext(environment, self.max_tool_calls)
        ctx.programmer_calls = 0
        ctx.sandbox = sandbox
        frozen_hash = bank.hash
        overlay = bank.clone()
        try:
            trace = self._execute(task, ctx, overlay, key)
        except ResponseContractError as exc:
            trace = {
                "task": task.public(),
                "answer": None,
                "events": ctx.events,
                "terminal_reason": "response_contract_error",
                "error": str(exc),
                "bank_hash": bank.hash,
            }
        if bank.hash != frozen_hash:
            raise ValueError("Runtime modified its parent Bank")
        trace["bank_hash"] = frozen_hash
        trace["task_overlay_hash"] = overlay.hash
        trace["programmer_calls"] = ctx.programmer_calls
        trace["retrieved_skill_ids"] = [
            s["id"]
            for s in bank.retrieve(task.instruction, self.retrieval_limit)["skills"]
        ]
        return trace

    def _programmer_policy(self, ctx):
        return {
            "enabled": self.allow_programmer and self.worker is not None,
            "scope": "task_local",
            "remaining": max(0, self.max_programs_per_task - ctx.programmer_calls),
            "structural_repair_limit": self.programmer_repair_limit,
            "validation": (
                "isolated_prefix_replay"
                if getattr(ctx, "sandbox", None)
                else "structural_only"
            ),
            "validation_repair_limit": self.validation_repair_limit,
        }

    def _program_catalog(self, bank, ctx):
        return [
            {
                **{
                    k: v
                    for k, v in p.items()
                    if k not in {"source", "instructions", "qualification"}
                },
                "reuse_eligible": self._ineligible(p, ctx, certified=True) is None,
            }
            for p in bank.programs.values()
        ]

    def _sandbox_validate(self, program, inputs, ctx):
        """Replay the live tool prefix in a fresh instance, then run the candidate."""
        prefix = [e for e in ctx.events if e["type"] == "tool"]
        environment = ctx.sandbox()
        isolated = ToolContext(environment, len(prefix) + ctx.remaining_calls())
        stage = "prefix"
        try:
            for event in prefix:
                isolated.call(event["name"], event["arguments"])
            stage = "program"
            self._admit(program, inputs, isolated)
            result = self._invoke(program, inputs, isolated)
            if result["status"] != "ok":
                raise ValueError("Program returned status " + str(result["status"]))
            return {
                "passed": True,
                "replayed_prefix": len(prefix),
                "receipt": {
                    "episode": "isolated_validation",
                    "inputs_hash": digest(inputs),
                    "reference_hash": digest([e["arguments"] for e in prefix]),
                    "result_hash": digest(result),
                    "passed": True,
                },
            }
        except EXECUTION_ERRORS as exc:
            return {
                "passed": False,
                "stage": stage,
                "replayed_prefix": len(prefix),
                "error": str(exc),
            }
        finally:
            environment.close()

    def _ask_programmer(self, *args, **kwargs):
        try:
            return self.programmer.compile(*args, **kwargs)
        except RuntimeError as exc:
            exc.provider_failure = True
            raise

    def _compile(self, spec, args, task, ctx, bank, key):
        """Programmer synthesis, isolated validation and bounded repair (Sec. 3.2.4)."""
        policy = self._programmer_policy(ctx)
        if not policy["enabled"] or not policy["remaining"]:
            raise ValueError("Runtime Programmer is disabled or exhausted")
        self._preflight(spec, args, ctx)
        ctx.programmer_calls += 1
        # Provider/ledger failures propagate. A failed live execution is never
        # automatically repeated by structural repair.
        program, receipt = self._ask_programmer(
            spec, args, task, ctx, key + "/programmer"
        )
        if getattr(ctx, "sandbox", None):
            for attempt in range(self.validation_repair_limit + 1):
                check = self._sandbox_validate(program, args, ctx)
                ctx.events.append(
                    {
                        "type": "runtime_validation",
                        "program_id": program["id"],
                        "attempt": attempt + 1,
                        **{k: v for k, v in check.items() if k != "receipt"},
                    }
                )
                if check["passed"]:
                    break
                if check["stage"] == "prefix" or attempt == self.validation_repair_limit:
                    raise ValueError("Isolated validation failed: " + check["error"])
                program, receipt = self._ask_programmer(
                    spec,
                    args,
                    task,
                    ctx,
                    key + "/programmer/validation_repair_" + str(attempt + 1),
                    feedback={
                        "previous_source": program["source"],
                        "discrepancy": check["error"],
                    },
                )
            program["qualification"] = certificate(
                program, ctx.available_tools(), [check["receipt"]], "task_local"
            )
            receipt.update(
                admission="isolated_validation",
                validation_attempts=attempt + 1,
                program=deepcopy(program),
            )
        if program["id"] in bank.programs:
            raise ValueError("Task-local Program ID collision")
        bank.programs[program["id"]] = program
        ctx.events.append({"type": "runtime_programmer", **receipt})
        return program["id"]

    def _run_compiled(self, bank, pid, args, ctx):
        try:
            result = self.program(bank, pid, args, ctx)
        except EXECUTION_ERRORS as exc:
            ctx.events.append(
                {
                    "type": "runtime_qualification",
                    "program_id": pid,
                    "status": "failed",
                    "error": str(exc),
                    "scope": "task_local",
                    "persistent_admission": False,
                }
            )
            raise
        ctx.events.append(
            {
                "type": "runtime_qualification",
                "program_id": pid,
                "status": result["status"],
                "scope": "task_local",
                "persistent_admission": False,
            }
        )
        return result

    def _act(self, decision, task, ctx, bank, key):
        action = decision["action"]
        args = decision.get("arguments", {})
        try:
            if action == "programmer":
                pid = self._compile(decision["operation"], args, task, ctx, bank, key)
                return self._run_compiled(bank, pid, args, ctx)
            if action == "program":
                # An agent's explicit choice still needs Env and a valid certificate.
                return self.program(
                    bank, decision["id"], args, ctx, require_certificate=True
                )
            if action == "skill":
                return self.skill(bank, decision["id"], args, ctx)
            if action == "tool":
                return ctx.call(decision["id"], args)
            raise ValueError("Unsupported Runtime action")
        except EXECUTION_ERRORS as exc:
            if _provider_failure(exc):
                raise
            raise ValueError("Execution failed: " + str(exc)) from exc

    @staticmethod
    def _node_spec(node):
        c = node["contract"]
        return {
            "name": node["id"],
            "instruction": node.get("goal", node["id"]),
            "input_schema": c["input_schema"],
            "output_schema": node.get("output_schema", c["output_schema"]),
            "allowed_tools": node.get("allowed_tools", []),
            "preconditions": c.get("preconditions", []),
            "effects": c.get("effects", []),
        }

    def _bound(self, node, args, task, ctx, bank, key):
        """Programmatic node: certified reuse, else Programmer, else agent guidance."""
        mode = node.get("execution_mode")
        if mode == "dynamic":
            return None
        try:
            if node.get("program_id"):
                return self.program(
                    bank, node["program_id"], args, ctx, require_certificate=True
                )
            if node.get("skill_id") and implementations(bank.skills[node["skill_id"]]):
                return self.skill(
                    bank, node["skill_id"], args, ctx, node.get("preferred_program_id")
                )
            if mode != "programmatic":
                return None
            reason = "No reusable implementation"
        except Ineligible as exc:
            reason = str(exc)
        except EXECUTION_ERRORS as exc:
            # An eligible implementation ran and failed: a localized node failure,
            # not a silent fallback after possible side effects.
            if _provider_failure(exc):
                raise
            raise ValueError("Implementation failed: " + str(exc)) from exc
        if mode == "programmatic" and node.get("contract"):
            try:
                pid = self._compile(
                    self._node_spec(node),
                    args,
                    task,
                    ctx,
                    bank,
                    key + "/node_" + node["id"],
                )
                return self._run_compiled(bank, pid, args, ctx)
            except EXECUTION_ERRORS as exc:
                if _provider_failure(exc):
                    raise
                reason += "; Programmer: " + str(exc)
        ctx.events.append(
            {
                "type": "implementation_fallback",
                "node": node["id"],
                "reason": reason,
                "continuation": "agent_guided",
            }
        )
        return None

    def _execute(self, task, ctx, bank, key):
        retrieved = bank.retrieve(task.instruction, self.retrieval_limit)
        prepared = (
            self.planner.prepare(task, ctx, bank, key)
            if self.planning_enabled
            else None
        )
        if self.planning_enabled:
            retrieved["skeletons"] = []
        if prepared:
            try:
                self._preflight(prepared[0]["contract"], prepared[1], ctx)
            except (ValueError, KeyError) as exc:
                ctx.events.append({"type": "planning_rejected", "error": str(exc)})
                prepared = None
        if prepared:
            skeleton, inputs = prepared
            bank.skeletons[skeleton["id"]] = skeleton
            bank.validate()
            retrieved["skeletons"] = [skeleton]
            plan = {"skeleton_id": skeleton["id"], "inputs": inputs}
        outputs = {}
        answer = None
        reason = "step_limit"
        steps = 0
        if retrieved["skeletons"]:
            plan = (
                plan
                if prepared
                else self.model.ask(
                    key + "/plan",
                    prompts.SKELETON_SELECTOR,
                    {
                        "task": task.public(),
                        "observation": ctx.observe(),
                        "tools": ctx.available_tools(),
                        "skills": retrieved,
                    },
                    PLAN_SCHEMA,
                )
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
                    guided = None
                    try:
                        args = {
                            k: resolve(v, skeleton_inputs, outputs)
                            for k, v in node.get("args", {}).items()
                        }
                        self._preflight(node.get("contract", {}), args, ctx)
                        result = self._bound(node, args, task, ctx, bank, key)
                        if result is not None:
                            if result["status"] != "ok":
                                raise ValueError(
                                    "Node implementation returned status "
                                    + str(result["status"])
                                )
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
                            node_step_schema = deepcopy(STEP_SCHEMA)
                            node_step_schema["properties"]["answer"] = schema
                            node_step_schema["allOf"] = [
                                {
                                    "if": {
                                        "properties": {"action": {"const": "finish"}}
                                    },
                                    "then": {"required": ["answer"]},
                                }
                            ]
                            guidance = [
                                bank.skills[sid]
                                for sid in node.get("reference_skill_ids", [])
                            ]
                            if node.get("skill_id"):
                                guided = node["skill_id"]
                                guidance.append(bank.skills[guided])
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
                                        "programmer": self._programmer_policy(ctx),
                                        "finish_output_schema": schema,
                                        "programs": self._program_catalog(bank, ctx),
                                    },
                                    node_step_schema,
                                )
                                if decision["action"] == "finish":
                                    validate(decision.get("answer"), schema)
                                    outputs[node["id"]] = decision["answer"]
                                    break
                                self._act(
                                    decision,
                                    task,
                                    ctx,
                                    bank,
                                    key + "/node_" + node["id"] + "/" + str(steps),
                                )
                                steps += 1
                            if node["id"] not in outputs:
                                raise ValueError(
                                    "Agent-guided operation exhausted its step budget"
                                )
                        self._effects(
                            node.get("contract", {}),
                            args,
                            {"status": "ok", "outputs": outputs[node["id"]]},
                            ctx,
                        )
                        ctx.events.append(
                            {
                                "type": "skeleton_node",
                                "id": node["id"],
                                "execution": "implementation"
                                if result is not None
                                else "agent_guided",
                                **({"guidance_skill_id": guided} if guided else {}),
                                "outputs": outputs[node["id"]],
                            }
                        )
                    except BudgetExceeded:
                        raise
                    except (ValueError, KeyError, TimeoutError) as exc:
                        outputs.pop(node["id"], None)
                        ctx.events.append(
                            {
                                "type": "skeleton_error",
                                "id": node["id"],
                                **({"guidance_skill_id": guided} if guided else {}),
                                "error": str(exc),
                            }
                        )
                        break
                if len(outputs) == len(skeleton["nodes"]) and skeleton.get("outputs"):
                    final = {
                        k: resolve(v, skeleton_inputs, outputs)
                        for k, v in skeleton["outputs"].items()
                    }
                    try:
                        if "contract" in skeleton:
                            validate(final, skeleton["contract"]["output_schema"])
                            self._effects(
                                skeleton["contract"],
                                skeleton_inputs,
                                {"status": "ok", "outputs": final},
                                ctx,
                            )
                    except (ValueError, KeyError) as exc:
                        ctx.events.append(
                            {
                                "type": "skeleton_error",
                                "id": skeleton["id"],
                                "error": str(exc),
                            }
                        )
                        final = {}
                    if final:
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
            decision = self.model.ask(
                key + "/step_" + str(step),
                prompts.RUNTIME,
                {
                    "task": task.public(),
                    "observation": ctx.observe(),
                    "tools": ctx.available_tools(),
                    "skills": retrieved["skills"],
                    "programs": self._program_catalog(bank, ctx),
                    "history": ctx.events,
                    "remaining_tool_calls": ctx.remaining_calls(),
                    "programmer": self._programmer_policy(ctx),
                },
                STEP_SCHEMA,
            )
            action = decision["action"]
            if action == "finish":
                answer = decision.get("answer")
                reason = "finished"
                break
            try:
                self._act(decision, task, ctx, bank, key + "/step_" + str(step))
            except (ValueError, KeyError, TimeoutError) as exc:
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
