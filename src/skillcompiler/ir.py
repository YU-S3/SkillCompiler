"""Bounded declarative control flow. Domain matching/effects belong to the adapter."""

from copy import deepcopy


def execute(program, inputs, ctx):
    local = {}
    calls = 0
    returned = None
    limit = program.get("max_actions", 64)
    instruction_count = 0

    def value(ref):
        if not isinstance(ref, dict):
            return ref
        source = ref.get("source", ref.get("kind"))
        field = ref.get("field", ref.get("source_role"))
        if source in {"tool_input", "skill_input", "input"}:
            return inputs[field]
        if source in {"literal", "constant"}:
            return ref.get("value", ref.get("constant"))
        if source in {"loop_variable", "variable", "local", "local_variable"}:
            result = local[
                ref.get("variable", ref.get("name", ref.get("source_step", field)))
            ]
            return result.get(field, result) if isinstance(result, dict) else result
        if "constant" in ref and ref["constant"] is not None:
            return ref["constant"]
        if field in local:
            return local[field]
        raise ValueError("Unsupported/unresolved IR reference: " + str(ref))

    def query(spec):
        if spec.get("source") in {"tool_input", "local_variable"}:
            result = value(spec)
            return result if isinstance(result, list) else [result]
        fn = getattr(ctx.environment, "query", None)
        if fn:
            return fn(deepcopy(spec), deepcopy(inputs), deepcopy(local))
        raise ValueError(
            "This IR requires an adapter query(spec, inputs, variables) interface"
        )

    def condition(spec):
        op = spec["op"]
        if op == "exists":
            return bool(query(spec["match"]))
        if op == "equals":
            return value(spec) == spec["value"]
        if op == "not":
            return not condition(spec["condition"])
        if op in {"all", "and"}:
            return all(condition(x) for x in spec["conditions"])
        if op in {"any", "or"}:
            return any(condition(x) for x in spec["conditions"])
        fn = getattr(ctx.environment, "condition", None)
        if fn:
            return bool(fn(spec, inputs, local))
        raise ValueError("Unsupported IR condition: " + op)

    def nodes(items):
        nonlocal calls, returned, instruction_count
        for node in items:
            if returned is not None:
                return
            instruction_count += 1
            if instruction_count > 4096:
                raise RuntimeError("IR instruction budget exhausted")
            op = node["op"]
            if op == "IF":
                nodes(
                    node["then_branch"]
                    if condition(node["condition"])
                    else node.get("else_branch", [])
                )
            elif op == "FOR_EACH":
                maximum = node["max_iterations"]
                if not isinstance(maximum, int) or not 0 <= maximum <= 256:
                    raise ValueError("Invalid bounded loop")
                for item in query(node["collection_source"])[:maximum]:
                    local[node["iteration_variable"]] = item
                    nodes(node["body"])
                    if returned is not None:
                        break
            elif op == "ACTION":
                if calls >= limit:
                    raise RuntimeError("IR action budget exhausted")
                if node["action_type"] not in program["allowed_tools"]:
                    raise ValueError("IR tool is outside Program permissions")
                args = {k: value(v) for k, v in node["argument_mapping"].items()}
                result = ctx.call(node["action_type"], args)
                calls += 1
                if node.get("result_variable"):
                    local[node["result_variable"]] = deepcopy(result)
                if node.get("expected_effects"):
                    verifier = getattr(ctx.environment, "verify_effects", None)
                    if verifier is None or not verifier(
                        node["expected_effects"], inputs, local, result
                    ):
                        raise ValueError("Declared operation effects were not verified")
            elif op == "RETURN":
                returned = {k: value(v) for k, v in node["output_sources"].items()}
            else:
                raise ValueError("Unsupported IR instruction: " + op)

    nodes(program["instructions"])
    if returned is not None and program.get("final_effects"):
        verifier = getattr(ctx.environment, "verify_effects", None)
        if verifier is None or not verifier(
            program["final_effects"], {**inputs, **returned}, local, returned
        ):
            raise ValueError("Final operation effects were not verified")
    return {
        "status": "ok" if returned is not None else "not_found",
        "outputs": returned or {},
    }
