"""Public task analysis, contract retrieval and validated skeleton construction."""

from copy import deepcopy
from .contracts import digest, validate
from .semantic import contract, match_requirement, encoded, effects_cover
from .model import ResponseContractError
from . import prompts

from .schemas import CONTRACT as CONTRACT_SCHEMA, BINDING

OP_SCHEMA = {
    "type": "object",
    "properties": {
        "id": {"type": "string", "pattern": "^[A-Za-z][A-Za-z0-9_]{0,63}$"},
        "intent": {"type": "string", "minLength": 1},
        "contract": CONTRACT_SCHEMA,
        "args": {"type": "object", "additionalProperties": BINDING},
        "after": {"type": "array", "items": {"type": "string"}, "uniqueItems": True},
    },
    "required": ["id", "intent", "contract", "args", "after"],
    "additionalProperties": False,
}
ANALYSIS_SCHEMA = {
    "oneOf": [
        {
            "type": "object",
            "properties": {
                "decision": {"const": "direct"},
                "reason": {"type": "string", "minLength": 1},
            },
            "required": ["decision", "reason"],
            "additionalProperties": False,
        },
        {
            "type": "object",
            "properties": {
                "decision": {"const": "plan"},
                "contract": CONTRACT_SCHEMA,
                "operations": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 8,
                    "items": OP_SCHEMA,
                },
                "outputs": {"type": "object", "additionalProperties": BINDING},
            },
            "required": ["decision", "contract", "operations", "outputs"],
            "additionalProperties": False,
        },
    ]
}


def _root_schema(value, field):
    result = value
    for part in field.split("."):
        if result.get("type") == "array" and part.isdigit():
            result = result.get("items", {})
        else:
            if part not in result.get("properties", {}):
                raise ValueError("Unknown referenced role: " + field)
            result = result["properties"][part]
    return result


def _schema_accepts(source, target):
    # Exact constraints, with human documentation ignored. No guessed subtype proof.
    clean = lambda s: {k: v for k, v in s.items() if k not in {"title", "description"}}
    return encoded(clean(source)) == encoded(clean(target))


def _binding_schema(binding, inputs, outputs, ancestors=None):
    if not isinstance(binding, dict):
        raise ValueError("Bindings must explicitly use task/from/literal")
    if set(binding) == {"task"}:
        return _root_schema(inputs, binding["task"])
    if set(binding) == {"from", "field"}:
        node = binding["from"]
        if node not in outputs or ancestors is not None and node not in ancestors:
            raise ValueError("Data dependency is missing, forward or cyclic")
        return _root_schema(outputs[node], binding["field"])
    if set(binding) == {"literal"}:
        return None
    raise ValueError("Unknown or ambiguous binding")


def _bind_predicate(value, inputs, outputs):
    if isinstance(value, list):
        return [_bind_predicate(v, inputs, outputs) for v in value]
    if not isinstance(value, dict) or value.get("kind") == "constant":
        return deepcopy(value)
    if value.get("kind") in {
        "skill_input",
        "tool_input",
        "skill_output",
        "tool_output",
    }:
        mapping = inputs if value["kind"].endswith("input") else outputs
        if value.get("source_role") not in mapping:
            raise ValueError(
                "Unbound predicate role: "
                + str(value.get("source_role"))
                + "; available roles at this boundary: "
                + ", ".join(sorted(mapping))
            )
        return deepcopy(mapping[value["source_role"]])
    return {k: _bind_predicate(v, inputs, outputs) for k, v in value.items()}


def validate_analysis(analysis, public_inputs=None):
    validate(analysis, ANALYSIS_SCHEMA)
    if analysis["decision"] == "direct":
        return {"status": "direct", "reason": analysis["reason"]}
    goal = contract(analysis["contract"])
    declared_inputs = goal["input_schema"].get("properties", {})
    # Bank admission verifies structure without fabricating task input values.
    bound = {}
    if public_inputs is not None:
        bound = {k: public_inputs[k] for k in declared_inputs if k in public_inputs}
        validate(bound, goal["input_schema"])
        if set(declared_inputs) - set(public_inputs):
            raise ValueError("Task contract invented an unavailable input")
    results, ancestors, offered = {}, {}, []
    input_refs = {k: {"task": k} for k in declared_inputs}
    known = _bind_predicate(goal["preconditions"], input_refs, {})
    for operation in analysis["operations"]:
        oid, deps = operation["id"], operation["after"]
        if oid in results or set(deps) - set(results):
            raise ValueError("Operations must have unique IDs in dependency order")
        parents = set(deps)
        for dep in deps:
            parents.update(ancestors[dep])
        ancestors[oid] = parents
        item = contract(operation["contract"])
        props = item["input_schema"].get("properties", {})
        if set(operation["args"]) - set(props) or set(
            item["input_schema"].get("required", [])
        ) - set(operation["args"]):
            raise ValueError("Operation argument roles do not satisfy its contract")
        for role, binding in operation["args"].items():
            schema = _binding_schema(binding, goal["input_schema"], results, parents)
            if schema is None:
                validate(binding["literal"], props[role])
            elif not _schema_accepts(schema, props[role]):
                raise ValueError("Data edge has incompatible types or constraints")
        out_refs = {
            r: {"from": oid, "field": r}
            for r in item["output_schema"].get("properties", {})
        }
        pre = _bind_predicate(item["preconditions"], operation["args"], {})
        available = known + [e for origin, e in offered if origin in parents]
        if not effects_cover(pre, available):
            raise ValueError(
                "Operation prerequisites are not established by its dependencies"
            )
        offered.extend(
            (oid, effect)
            for effect in _bind_predicate(item["effects"], operation["args"], out_refs)
        )
        results[oid] = item["output_schema"]
    final_props = goal["output_schema"].get("properties", {})
    if set(analysis["outputs"]) != set(final_props):
        raise ValueError("Final output roles do not match the task contract")
    for role, binding in analysis["outputs"].items():
        schema = _binding_schema(binding, goal["input_schema"], results)
        if schema is None:
            validate(binding["literal"], final_props[role])
        elif not _schema_accepts(schema, final_props[role]):
            raise ValueError("Final output binding has incompatible type")
    required = _bind_predicate(goal["effects"], input_refs, analysis["outputs"])
    if not effects_cover(required, [e for _, e in offered]):
        raise ValueError("Skeleton does not cover the complete task goal")
    return {
        "status": "valid",
        "goal_hash": digest(goal),
        "analysis_hash": digest(analysis),
        "operations": len(results),
        "goal_effects": len(required),
        "inputs": bound,
    }


def skeleton_analysis(skeleton, bank):
    if "contract" not in skeleton:
        raise ValueError("Skeleton has no complete task contract")
    operations = []
    for node in skeleton["nodes"]:
        requirement = node.get("contract")
        if requirement is None:
            requirement = bank.skills.get(node.get("skill_id")) or bank.programs.get(
                node.get("program_id")
            )
        if requirement is None:
            raise ValueError("Skeleton node has no operation contract")
        for field, catalog in (
            ("skill_id", bank.skills),
            ("program_id", bank.programs),
        ):
            if (
                node.get(field)
                and match_requirement(requirement, catalog[node[field]])["status"]
                != "exact"
            ):
                raise ValueError(
                    "Skeleton implementation no longer meets its operation"
                )
        operations.append(
            {
                "id": node["id"],
                "intent": node.get("goal", node["id"]),
                "contract": contract(requirement),
                "args": deepcopy(node.get("args", {})),
                "after": node.get("after", []),
            }
        )
    return {
        "decision": "plan",
        "contract": skeleton["contract"],
        "operations": operations,
        "outputs": skeleton.get("outputs", {}),
    }


class Planner:
    def __init__(self, model, retrieval_limit=8, repair_limit=1):
        if repair_limit not in (0, 1):
            raise ValueError("Planning structural repair is bounded to one")
        self.model, self.limit, self.repair_limit = model, retrieval_limit, repair_limit

    def prepare(self, task, ctx, bank, key):
        public = {"goal": task.instruction, **task.inputs}
        material = {
            "task": task.public(),
            "public_inputs": public,
            "observation": ctx.observe(),
            "predicate_contracts": deepcopy(
                getattr(ctx.environment, "predicate_contracts", lambda: [])()
            ),
            "tools": ctx.available_tools(),
            "operation_interfaces": [
                {k: v for k, v in s.items() if k != "source"}
                for s in bank.skills.values()
            ],
        }
        analysis, error = None, None
        for attempt in range(self.repair_limit + 1):
            try:
                analysis = self.model.ask(
                    key + ("/analyze" if attempt == 0 else "/analysis_repair"),
                    prompts.ANALYZER if attempt == 0 else prompts.REPAIR,
                    (
                        material
                        if attempt == 0
                        else {
                            "original": material,
                            "previous_response": analysis,
                            "errors": error,
                        }
                    ),
                    ANALYSIS_SCHEMA,
                )
            except ResponseContractError as exc:
                analysis, error = exc.response, str(exc)
                continue
            try:
                check = validate_analysis(analysis, public)
                break
            except (ValueError, KeyError, TypeError) as exc:
                error = str(exc)
        else:
            ctx.events.append(
                {
                    "type": "planning_rejected",
                    "error": error,
                    "attempts": self.repair_limit + 1,
                }
            )
            return None
        ctx.events.append(
            {
                "type": "task_analysis",
                "analysis": deepcopy(analysis),
                "validation": check,
            }
        )
        if analysis["decision"] == "direct":
            return None
        catalog, proofs = {}, {}
        for op in analysis["operations"]:
            found = bank.retrieve(op["intent"], self.limit, requirement=op["contract"])
            catalog[op["id"]] = found["skills"]
            for s in found["skills"]:
                proofs[(op["id"], s["id"])] = match_requirement(op["contract"], s)[
                    "proof"
                ]
        complete = []
        for s in bank.skeletons.values():
            try:
                # Reusable skeletons retain the public task ABI. Role aliases
                # require explicit rebinding, not an invented positional match.
                if (
                    match_requirement(analysis["contract"], s.get("contract", {}))[
                        "status"
                    ]
                    != "exact"
                ):
                    continue
                validate_analysis(skeleton_analysis(s, bank), public)
                relation = match_requirement(analysis["contract"], s["contract"])[
                    "proof"
                ]
                if any(
                    k != v
                    for boundary in ("input_map", "output_map")
                    for k, v in relation[boundary].items()
                ):
                    continue
                complete.append(s)
            except (ValueError, KeyError, TypeError):
                continue
        schema = {
            "type": "object",
            "properties": {
                "skeleton_id": {"enum": [""] + [s["id"] for s in complete]},
                "choices": {
                    "type": "object",
                    "properties": {
                        oid: {"enum": [""] + [s["id"] for s in found]}
                        for oid, found in catalog.items()
                    },
                    "required": list(catalog),
                    "additionalProperties": False,
                },
            },
            "required": ["skeleton_id", "choices"],
            "additionalProperties": False,
        }
        try:
            choice = self.model.ask(
                key + "/construct",
                prompts.CONSTRUCTOR,
                {
                    "task": task.public(),
                    "analysis": analysis,
                    "compatible_operations": catalog,
                    "compatible_skeletons": complete,
                },
                schema,
            )
            validate(choice, schema)
        except (ResponseContractError, ValueError, KeyError, TypeError) as exc:
            ctx.events.append({"type": "construction_rejected", "error": str(exc)})
            return None
        if choice["skeleton_id"]:
            skeleton = deepcopy(bank.skeletons[choice["skeleton_id"]])
            mode = "reuse"
        else:
            from .alignment import _rewire_skeleton

            skeleton = {
                "id": "local_plan_" + digest([key, analysis, choice])[:20],
                "name": task.instruction,
                "contract": deepcopy(analysis["contract"]),
                "nodes": [],
                "outputs": deepcopy(analysis["outputs"]),
            }
            substitutions = {}
            for op in analysis["operations"]:
                node = {
                    "id": op["id"],
                    "goal": op["intent"],
                    "args": deepcopy(op["args"]),
                    "after": list(op["after"]),
                    "contract": deepcopy(op["contract"]),
                    "output_schema": deepcopy(op["contract"]["output_schema"]),
                }
                sid = choice["choices"][op["id"]]
                if sid:
                    alias = "operation_" + op["id"]
                    node["skill_id"] = alias
                    substitutions[alias] = (sid, proofs[(op["id"], sid)])
                else:
                    node["execution_mode"] = "dynamic"
                skeleton["nodes"].append(node)
            _rewire_skeleton(skeleton, substitutions)
            for node in skeleton["nodes"]:
                if node.get("skill_id"):
                    node["contract"] = contract(bank.skills[node["skill_id"]])
            mode = "construct"
        try:
            validation = validate_analysis(skeleton_analysis(skeleton, bank), public)
        except (ValueError, KeyError, TypeError) as exc:
            ctx.events.append({"type": "construction_rejected", "error": str(exc)})
            return None
        ctx.events.append(
            {
                "type": "skeleton_construction",
                "mode": mode,
                "skeleton": deepcopy(skeleton),
                "retrieval_proofs": [
                    {"operation": oid, "skill": sid, "proof": proof}
                    for (oid, sid), proof in proofs.items()
                ],
                "validation": validation,
            }
        )
        return skeleton, validation["inputs"]
