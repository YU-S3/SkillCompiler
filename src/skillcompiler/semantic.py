"""Bounded, proof-producing contract alignment, independent of natural language.

Adapted from the original contract canonicalizer and identity matcher. Role
renaming is a bijection; literals, domains, multiplicity and validators remain
part of identity. Exhaustion/unsupported contracts never authorize a merge.
"""

from collections import Counter
from copy import deepcopy
from itertools import permutations
import json
import re
from .contracts import digest


def encoded(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def lexical_similarity(left, right):
    a = Counter(re.findall(r"\w+", left.casefold()))
    b = Counter(re.findall(r"\w+", right.casefold()))
    return (
        2 * sum((a & b).values()) / (sum(a.values()) + sum(b.values()))
        if a and b
        else 0.0
    )


def role_schema(schema, mapping):
    """Rename only top-level interface roles, never nested property names."""
    result = deepcopy(schema)
    if "properties" in result:
        result["properties"] = {mapping[k]: v for k, v in result["properties"].items()}
    if "required" in result:
        result["required"] = sorted(mapping[k] for k in result["required"])
    for key in ("dependentRequired", "dependentSchemas"):
        if key in result:
            raise ValueError("Dependent interface schemas need an explicit adapter")
    if any(
        k in result for k in ("$ref", "allOf", "anyOf", "oneOf", "patternProperties")
    ):
        raise ValueError("Composed interface schemas cannot prove role equivalence")
    return result


def rewrite_roles(value, inputs, outputs):
    if isinstance(value, list):
        return [rewrite_roles(v, inputs, outputs) for v in value]
    if not isinstance(value, dict):
        # Ordinary strings, including dollar-prefixed literals, are not roles.
        return deepcopy(value)
    if value.get("kind") == "constant" or "literal" in value:
        return deepcopy(value)
    result = {k: rewrite_roles(v, inputs, outputs) for k, v in value.items()}
    kind = value.get("kind")
    if kind in ("skill_input", "tool_input", "skill_output", "tool_output"):
        mapping = inputs if kind.endswith("input") else outputs
        role = value.get("source_role")
        if role not in mapping:
            raise ValueError(
                "Unknown semantic role reference: "
                + str(role)
                + "; available roles: "
                + ", ".join(sorted(mapping))
            )
        result["source_role"] = mapping[role]
    elif "source_role" in value:
        raise ValueError("Unsupported semantic binding kind")
    return result


def contract(asset, inputs=None, outputs=None):
    if "input_schema" not in asset or "output_schema" not in asset:
        raise ValueError("Explicit input/output contracts are required for identity")
    ins, outs = asset["input_schema"], asset["output_schema"]
    inputs = inputs if inputs is not None else {k: k for k in ins.get("properties", {})}
    outputs = (
        outputs if outputs is not None else {k: k for k in outs.get("properties", {})}
    )

    def predicates(key):
        result = []
        for p in asset.get(key, []):
            if not isinstance(p, dict) or not p.get("predicate"):
                raise ValueError("Semantic predicates need a name and typed arguments")
            item = rewrite_roles(p, inputs, outputs)
            item["predicate"] = item["predicate"].casefold()
            item.setdefault("args", {})
            item.setdefault("effect_domain", "world")
            item.setdefault("cardinality", 1)
            item.setdefault("distinct_by", "")
            if type(item["cardinality"]) is not int or item["cardinality"] < 1:
                raise ValueError("Invalid predicate cardinality")
            result.append(item)
        return sorted(result, key=encoded)

    return {
        "input_schema": role_schema(ins, inputs),
        "output_schema": role_schema(outs, outputs),
        "preconditions": predicates("preconditions"),
        "effects": predicates("effects"),
        "validator": rewrite_roles(asset.get("validator", {}), inputs, outputs),
        "entry_constraints": rewrite_roles(
            asset.get("entry_constraints", []), inputs, outputs
        ),
    }


def effects_cover(required, offered):
    """Conservative predicate entailment: identical terms, sufficient cardinality."""

    def fixed(value):
        return {k: v for k, v in value.items() if k != "cardinality"}

    return all(
        any(
            encoded(fixed(need)) == encoded(fixed(have))
            and have["cardinality"] >= need["cardinality"]
            for have in offered
        )
        for need in required
    )


def _satisfies(required, offered):
    # Retrieval is directional; merge identity remains strict equality.
    return (
        required["input_schema"] == offered["input_schema"]
        and required["output_schema"] == offered["output_schema"]
        and effects_cover(offered["preconditions"], required["preconditions"])
        and effects_cover(required["effects"], offered["effects"])
        and required["validator"] == offered["validator"]
        and required["entry_constraints"] == offered["entry_constraints"]
    )


def match_requirement(requirement, candidate, max_states=4096):
    """Prove role rebinding and coverage, never infer semantics from names."""
    return _align(requirement, candidate, max_states, directional=True)


def align(source, target, max_states=4096):
    return _align(source, target, max_states, directional=False)


def _align(source, target, max_states, directional):
    """Return exact/different/unknown plus independently verifiable role maps."""
    if max_states < 1:
        raise ValueError("Identity search budget must be positive")
    count = 0
    try:
        expected = contract(target)
        contract(source)
        groups = []
        for boundary in ("input", "output"):
            a = source[boundary + "_schema"].get("properties", {})
            b = target[boundary + "_schema"].get("properties", {})
            if len(a) != len(b):
                return {
                    "status": "different",
                    "reason": "interface arity",
                    "states": count,
                }
            groups.append((a, b))

        # Enumerate bounded bijections, using complete property contracts as colors.
        def maps(a, b):
            names = sorted(a)
            for values in permutations(sorted(b)):
                nonlocal count
                count += 1
                if count > max_states:
                    raise OverflowError("identity search exhausted")
                if all(encoded(a[k]) == encoded(b[v]) for k, v in zip(names, values)):
                    yield dict(zip(names, values))

        for ins in maps(*groups[0]):
            for outs in maps(*groups[1]):
                normalized = contract(source, ins, outs)
                compatible = (
                    _satisfies(normalized, expected)
                    if directional
                    else encoded(normalized) == encoded(expected)
                )
                if compatible:
                    proof = {
                        "input_map": ins,
                        "output_map": outs,
                        "source_hash": digest(contract(source)),
                        "target_hash": digest(contract(target)),
                        "contract_hash": digest(expected),
                    }
                    if directional:
                        proof["relation"] = "candidate_satisfies_requirement"
                    return {"status": "exact", "proof": proof, "states": count}
        return {"status": "different", "reason": "contract mismatch", "states": count}
    except (ValueError, KeyError, TypeError, OverflowError) as exc:
        return {"status": "unknown", "reason": str(exc), "states": count}


def verify(source, target, proof):
    try:
        if proof["source_hash"] != digest(contract(source)) or proof[
            "target_hash"
        ] != digest(contract(target)):
            return False
        for boundary in ("input", "output"):
            mapping = proof[boundary + "_map"]
            a = source[boundary + "_schema"].get("properties", {})
            b = target[boundary + "_schema"].get("properties", {})
            if (
                set(mapping) != set(a)
                or set(mapping.values()) != set(b)
                or len(set(mapping.values())) != len(mapping)
            ):
                return False
        normalized = contract(source, proof["input_map"], proof["output_map"])
        return (
            encoded(normalized) == encoded(contract(target))
            and digest(normalized) == proof["contract_hash"]
        )
    except (KeyError, ValueError, TypeError):
        return False


def implementations(skill):
    """Legacy program_ids are identity bindings, not a preferred-first policy."""
    return deepcopy(skill.get("implementations", [])) + [
        {"program_id": p, "input_map": {}, "output_map": {}}
        for p in skill.get("program_ids", [])
        if p not in {i["program_id"] for i in skill.get("implementations", [])}
    ]


def implementation_maps(skill, program, binding):
    ins = binding.get("input_map") or {
        k: k for k in program["input_schema"].get("properties", {})
    }
    outs = binding.get("output_map") or {
        k: k
        for k in skill.get("output_schema", program["output_schema"]).get(
            "properties", {}
        )
    }
    if len(set(ins.values())) != len(ins) or len(set(outs.values())) != len(outs):
        raise ValueError("Implementation role maps must be bijective")
    if "input_schema" in skill and "output_schema" in skill:
        skill_inputs = role_schema(
            skill["input_schema"],
            {k: k for k in skill["input_schema"].get("properties", {})},
        )
        skill_outputs = role_schema(
            skill["output_schema"],
            {k: k for k in skill["output_schema"].get("properties", {})},
        )
        if encoded(role_schema(program["input_schema"], ins)) != encoded(skill_inputs):
            raise ValueError("Implementation input contract mismatch")
        reverse = {v: k for k, v in outs.items()}
        if len(reverse) != len(outs) or encoded(
            role_schema(program["output_schema"], reverse)
        ) != encoded(skill_outputs):
            raise ValueError("Implementation output contract mismatch")
    return ins, outs


def executable_equal(source, target):
    """No claim of general Python program equivalence: compare executable bodies."""
    ignored = {"id", "name", "description"}
    return encoded({k: v for k, v in source.items() if k not in ignored}) == encoded(
        {k: v for k, v in target.items() if k not in ignored}
    )
