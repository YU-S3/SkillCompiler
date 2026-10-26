"""Stage, prove, rebind and atomically commit deployment assets.

Training receipts retain parent hashes and proofs. The deployment Bank contains
only executable interfaces and guidance. Failed edits never mutate their parent.
"""

from copy import deepcopy
from .contracts import digest
from .semantic import (
    align,
    verify,
    implementations,
    implementation_maps,
    executable_equal,
    role_schema,
)


def _skill_proof(source, target):
    if not source.get("effects") and not source.get("validator"):
        raise ValueError("Skill identity needs an effect or validator contract")
    result = align(source, target)
    if result["status"] != "exact" or not verify(source, target, result["proof"]):
        raise ValueError("Semantic alignment lacks exact proof: " + result["status"])
    return result["proof"]


def _transfer(source, target, programs, proof):
    target.setdefault("implementations", implementations(target))
    target.pop("program_ids", None)
    for binding in implementations(source):
        program = programs[binding["program_id"]]
        ins, outs = implementation_maps(source, program, binding)
        item = {
            "program_id": binding["program_id"],
            "input_map": {p: proof["input_map"][s] for p, s in ins.items()},
            "output_map": {proof["output_map"][s]: p for s, p in outs.items()},
        }
        implementation_maps(target, program, item)
        if item not in target["implementations"]:
            target["implementations"].append(item)


def _rewire_skeleton(skeleton, substitutions):
    """Rewrite typed bindings only; preserve literals and nested field suffixes."""
    output_maps = {}
    for node in skeleton["nodes"]:
        old = node.get("skill_id")
        if old in substitutions:
            new, proof = substitutions[old]
            node["skill_id"] = new
            node["args"] = {
                proof["input_map"][k]: v for k, v in node.get("args", {}).items()
            }
            output_maps[node["id"]] = proof["output_map"]
            if "output_schema" in node:
                node["output_schema"] = role_schema(
                    node["output_schema"], proof["output_map"]
                )
            # An explicit Program remains in its original ABI. Use the aligned
            # Skill binding instead of bypassing the boundary with old args.
            if node.get("program_id"):
                node["preferred_program_id"] = node.pop("program_id")
        node["reference_skill_ids"] = (
            list(
                dict.fromkeys(
                    substitutions[s][0] if s in substitutions else s
                    for s in node.get("reference_skill_ids", [])
                )
            )
            if "reference_skill_ids" in node
            else node.get("reference_skill_ids", [])
        )
        if not node["reference_skill_ids"]:
            node.pop("reference_skill_ids", None)

    def rewrite(value):
        if isinstance(value, list):
            return [rewrite(v) for v in value]
        if not isinstance(value, dict) or "literal" in value:
            return value
        out = {k: rewrite(v) for k, v in value.items()}
        if value.get("from") in output_maps and value.get("field"):
            root, *tail = value["field"].split(".")
            mapping = output_maps[value["from"]]
            if root not in mapping:
                raise ValueError("Consumer references an unknown output role")
            out["field"] = ".".join([mapping[root], *tail])
        return out

    for node in skeleton["nodes"]:
        node["args"] = rewrite(node.get("args", {}))
    skeleton["outputs"] = rewrite(skeleton.get("outputs", {}))


def skeleton_shape(asset):
    """Alpha-normalize node IDs while retaining order, data and dependency edges."""
    mapping = {n["id"]: "node_" + str(i) for i, n in enumerate(asset["nodes"])}

    def rewrite(value):
        if isinstance(value, list):
            return [rewrite(v) for v in value]
        if not isinstance(value, dict) or "literal" in value:
            return deepcopy(value)
        result = {k: rewrite(v) for k, v in value.items()}
        if "from" in value:
            result["from"] = mapping[value["from"]]
        return result

    result = {
        k: deepcopy(v)
        for k, v in asset.items()
        if k not in {"id", "name", "description", "nodes", "outputs"}
    }
    result["nodes"] = []
    for node in asset["nodes"]:
        n = rewrite(node)
        n["id"] = mapping[node["id"]]
        n["after"] = sorted(mapping[x] for x in node.get("after", []))
        result["nodes"].append(n)
    result["outputs"] = rewrite(asset.get("outputs", {}))
    return result


def _program_references(asset, replacements):
    if "program_ids" in asset:
        asset["program_ids"] = list(
            dict.fromkeys(replacements.get(p, p) for p in asset["program_ids"])
        )
    for binding in asset.get("implementations", []):
        binding["program_id"] = replacements.get(
            binding["program_id"], binding["program_id"]
        )
    for node in asset.get("nodes", []):
        for field in ("program_id", "preferred_program_id"):
            if field in node:
                node[field] = replacements.get(node[field], node[field])


def _composed(substitutions):
    result = {}
    for source, (target, original) in substitutions.items():
        proof, visited = deepcopy(original), {source}
        while target in substitutions:
            if target in visited:
                raise ValueError("Cyclic alignment substitutions")
            visited.add(target)
            destination, next_proof = substitutions[target]
            for boundary in ("input_map", "output_map"):
                proof[boundary] = {
                    k: next_proof[boundary][v] for k, v in proof[boundary].items()
                }
            target = destination
        result[source] = (target, proof)
    return result


def integrate(bank, additions, merges=(), removals=(), *, refine=False, evidence=()):
    """All checks and rewrites happen on private copies before Bank construction."""
    from .bank import Bank

    data = bank.data()
    indexes = {k: {a["id"]: a for a in v} for k, v in data.items()}
    before = {k: deepcopy(v) for k, v in indexes.items()}
    receipts, substitutions = [], {}
    program_substitutions = {}
    independent = len({r["episode"] for r in evidence}) >= 2
    explicit_targets = {m["target_id"] for m in merges}
    if removals:
        raise ValueError("Unqualified deletion is disabled; use a proved merge")
    for kind in ("programs", "skills", "skeletons"):
        seen = set()
        for value in additions.get(kind, []):
            asset = deepcopy(value)
            if kind != "programs":
                _program_references(asset, program_substitutions)
            aid = asset["id"]
            if aid in seen:
                raise ValueError("Duplicate proposed asset ID")
            seen.add(aid)
            prior = indexes[kind].get(aid)
            if kind == "programs" and prior is None and aid not in explicit_targets:
                equivalent = next(
                    (p for p in indexes[kind].values() if executable_equal(asset, p)),
                    None,
                )
                if equivalent is not None:
                    program_substitutions[aid] = equivalent["id"]
                    receipts.append(
                        {
                            "operation": "align",
                            "kind": kind,
                            "source": aid,
                            "target": equivalent["id"],
                            "source_hash": digest(asset),
                            "target_hash": digest(equivalent),
                            "proof": "exact_executable_and_interface",
                        }
                    )
                    continue
            if prior is not None and digest(prior) != digest(asset):
                if not refine or not independent:
                    raise ValueError(
                        "Revision needs refinement and independent training evidence"
                    )
                if kind == "skills":
                    proof = _skill_proof(prior, asset)
                    if any(
                        k != v
                        for boundary in ("input_map", "output_map")
                        for k, v in proof[boundary].items()
                    ):
                        raise ValueError(
                            "In-place revision must preserve role names; submit a new operation for alignment"
                        )
                    if implementations(prior) != implementations(asset):
                        raise ValueError(
                            "Implementation replacement must use the replay-qualified Program revision path"
                        )
                elif kind == "programs":
                    # Code may change after replay; its callers' ABI may not.
                    for field in ("input_schema", "output_schema", "allowed_tools"):
                        if prior[field] != asset[field]:
                            raise ValueError(
                                "Replacement changed a public Program interface"
                            )
                    if align(prior, asset)["status"] != "exact":
                        raise ValueError(
                            "Replacement changed the Program semantic contract"
                        )
                elif skeleton_shape(prior) != skeleton_shape(asset):
                    raise ValueError(
                        "Structural skeleton revisions require a new candidate ID"
                    )
                receipts.append(
                    {
                        "operation": "revise",
                        "kind": kind,
                        "id": aid,
                        "parents": [digest(prior)],
                        "version": digest(asset),
                    }
                )
            # Conservative auto-alignment of a newly extracted operation.
            if kind == "skills" and prior is None and aid not in explicit_targets:
                for existing in list(indexes[kind].values()):
                    try:
                        proof = _skill_proof(asset, existing)
                    except ValueError:
                        continue
                    _transfer(asset, existing, indexes["programs"], proof)
                    substitutions[aid] = (existing["id"], proof)
                    receipts.append(
                        {
                            "operation": "align",
                            "kind": kind,
                            "source": aid,
                            "target": existing["id"],
                            "proof": proof,
                        }
                    )
                    break
                if aid in substitutions:
                    continue
            indexes[kind][aid] = asset
    consumed = set()
    for merge in sorted(
        merges,
        key=lambda m: {"skills": 0, "programs": 1, "skeletons": 2}.get(m["kind"], 3),
    ):
        kind, sources, target_id = (
            merge["kind"],
            merge["source_ids"],
            merge["target_id"],
        )
        if not refine or not independent:
            raise ValueError("Merge needs refinement and independent training evidence")
        if (
            kind not in indexes
            or len(set(sources)) < 2
            or len(set(sources)) != len(sources)
        ):
            raise ValueError("Merge requires two distinct existing sources")
        if target_id in sources or target_id in before[kind] or consumed & set(sources):
            raise ValueError("Merge needs a new target and non-overlapping source sets")
        if (
            not all(s in before[kind] for s in sources)
            or target_id not in indexes[kind]
        ):
            raise ValueError("Unknown merge source or replacement")
        target = indexes[kind][target_id]
        proposed_bindings = implementations(target) if kind == "skills" else []
        if kind == "skills":
            target.pop("program_ids", None)
            target["implementations"] = []
        proofs = []
        for sid in sources:
            source = before[kind][sid]
            if kind == "skills":
                proof = _skill_proof(source, target)
                _transfer(source, target, indexes["programs"], proof)
                substitutions[sid] = (target_id, proof)
                proofs.append(proof)
            elif kind == "programs":
                if not executable_equal(source, target):
                    raise ValueError("Program merge requires exact executable identity")
                program_substitutions[sid] = target_id
                for skill in indexes["skills"].values():
                    skill["program_ids"] = list(
                        dict.fromkeys(
                            target_id if p == sid else p
                            for p in skill.get("program_ids", [])
                        )
                    )
                    for impl in skill.get("implementations", []):
                        if impl["program_id"] == sid:
                            impl["program_id"] = target_id
                for skeleton in indexes["skeletons"].values():
                    for node in skeleton["nodes"]:
                        for field in ("program_id", "preferred_program_id"):
                            if node.get(field) == sid:
                                node[field] = target_id
            else:
                left, right = deepcopy(source), deepcopy(target)
                for value in (left, right):
                    _program_references(value, program_substitutions)
                    if substitutions:
                        _rewire_skeleton(value, _composed(substitutions))
                if skeleton_shape(left) != skeleton_shape(right):
                    raise ValueError(
                        "Skeleton merge changed dependencies, operations or bindings"
                    )
            del indexes[kind][sid]
        if kind == "skills":
            for binding in proposed_bindings:
                ins, outs = implementation_maps(
                    target, indexes["programs"][binding["program_id"]], binding
                )
                resolved = {
                    "program_id": binding["program_id"],
                    "input_map": ins,
                    "output_map": outs,
                }
                if resolved not in target["implementations"]:
                    raise ValueError(
                        "Merge cannot introduce an unproved implementation mapping"
                    )
        consumed.update(sources)
        receipts.append(
            {
                "operation": "merge",
                "kind": kind,
                "sources": sources,
                "target": target_id,
                "parents": [digest(before[kind][s]) for s in sources],
                "version": digest(target),
                "proofs": proofs,
            }
        )
    if substitutions:
        substitutions = _composed(substitutions)
        for skeleton in indexes["skeletons"].values():
            _rewire_skeleton(skeleton, substitutions)
    # Complete skeleton identity is checked after all operation bindings are
    # retargeted. Existing assets are not silently retired by this discovery path.
    new_skeletons = (
        {s["id"] for s in additions.get("skeletons", [])}
        - set(before["skeletons"])
        - explicit_targets
    )
    for sid in sorted(new_skeletons):
        if sid not in indexes["skeletons"]:
            continue
        source = indexes["skeletons"][sid]
        target = next(
            (
                s
                for s in indexes["skeletons"].values()
                if s["id"] != sid and skeleton_shape(source) == skeleton_shape(s)
            ),
            None,
        )
        if target:
            receipts.append(
                {
                    "operation": "align",
                    "kind": "skeletons",
                    "source": sid,
                    "target": target["id"],
                    "structure_hash": digest(skeleton_shape(target)),
                }
            )
            del indexes["skeletons"][sid]
    candidate = Bank(**{k: list(v.values()) for k, v in indexes.items()})
    return candidate, receipts
