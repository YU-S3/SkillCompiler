"""Portable deployment content.

Programs carry compact certificates (hashes binding replay evidence to a body and
tool interface); full run logs and learning receipts live elsewhere.
"""

from copy import deepcopy
from pathlib import Path
import ast
import re
from .contracts import digest, read, save, validate


class Bank:
    def __init__(self, skills=(), programs=(), skeletons=()):
        skills, programs, skeletons = list(skills), list(programs), list(skeletons)
        self.skills = {x["id"]: deepcopy(x) for x in skills}
        self.programs = {x["id"]: deepcopy(x) for x in programs}
        self.skeletons = {x["id"]: deepcopy(x) for x in skeletons}
        if any(
            len(getattr(self, k)) != len(v)
            for k, v in (
                ("skills", skills),
                ("programs", programs),
                ("skeletons", skeletons),
            )
        ):
            raise ValueError("Duplicate asset ID")
        self.validate()

    def data(self):
        return deepcopy(
            {
                "skills": list(self.skills.values()),
                "programs": list(self.programs.values()),
                "skeletons": list(self.skeletons.values()),
            }
        )

    @property
    def hash(self):
        return digest(self.data())

    def clone(self):
        return Bank(**self.data())

    @classmethod
    def load(cls, folder):
        folder = Path(folder)
        if folder.is_file() and folder.name == "bank.json":
            folder = folder.parent
        data = read(folder / "bank.json")
        for program in data["programs"]:
            if "source_file" in program:
                path = (folder / program.pop("source_file")).resolve()
                if not path.is_relative_to(folder.resolve()):
                    raise ValueError("Source path escapes Bank")
                program["source"] = path.read_text(encoding="utf-8")
        return cls(**data)

    def write(self, folder):
        folder = Path(folder)
        data = self.data()
        for p in data["programs"]:
            if p.get("format", "python") == "python":
                path = Path("programs") / (p["id"] + ".py")
                (folder / path).parent.mkdir(parents=True, exist_ok=True)
                (folder / path).write_text(p.pop("source"), encoding="utf-8")
                p["source_file"] = path.as_posix()
        save(folder / "bank.json", data)

    def validate(self):
        ids = set()
        for kind in (self.skills, self.programs, self.skeletons):
            for key in kind:
                if key in ids or not re.fullmatch(r"[a-zA-Z0-9_-]+", key):
                    raise ValueError("Invalid or duplicate asset ID")
                ids.add(key)
        for s in self.skills.values():
            if not s.get("guidance") or not s.get("name"):
                raise ValueError("Skill needs name and guidance")
            if "usage" in s:
                usage = s["usage"]
                if (
                    not isinstance(usage, dict)
                    or set(usage) != {"attempts", "successes"}
                    or not all(type(v) is int and v >= 0 for v in usage.values())
                    or usage["successes"] > usage["attempts"]
                ):
                    raise ValueError("Skill usage needs attempt/success counts")
            if "utility" in s and not (
                isinstance(s["utility"], (int, float))
                and not isinstance(s["utility"], bool)
                and 0 <= s["utility"] <= 1
            ):
                raise ValueError("Skill utility must lie in [0, 1]")
            if "active" in s and not isinstance(s["active"], bool):
                raise ValueError("Skill active flag must be boolean")
            for p in s.get("program_ids", []):
                if p not in self.programs:
                    raise ValueError("Dangling Program reference")
            from .semantic import implementation_maps

            for binding in s.get("implementations", []):
                if binding.get("program_id") not in self.programs:
                    raise ValueError("Dangling implementation reference")
                implementation_maps(s, self.programs[binding["program_id"]], binding)
        for p in self.programs.values():
            if not isinstance(p.get("allowed_tools"), list):
                raise ValueError("Program needs explicit allowed_tools")
            if "qualification" in p:
                from .qualification import validate_record

                validate_record(p["qualification"])
            for k in ("input_schema", "output_schema"):
                from jsonschema import Draft202012Validator

                Draft202012Validator.check_schema(p[k])
            if p.get("format", "python") == "python":
                tree = ast.parse(p["source"])
                if not any(
                    isinstance(n, ast.FunctionDef) and n.name == "run"
                    for n in tree.body
                ):
                    raise ValueError("Missing run(ctx, inputs)")
            elif p["format"] != "ir":
                raise ValueError("Unknown Program format")
        for s in self.skeletons.values():
            known = set()
            for node in s["nodes"]:
                if node["id"] in known or set(node.get("after", [])) - known:
                    raise ValueError("Skeleton is cyclic or unordered")
                if node.get("skill_id") and node["skill_id"] not in self.skills:
                    raise ValueError("Dangling Skill reference")
                if node.get("execution_mode", "dynamic") not in {
                    "dynamic",
                    "programmatic",
                }:
                    raise ValueError("Unknown node execution_mode")
                if "allowed_tools" in node and not (
                    isinstance(node["allowed_tools"], list)
                    and all(isinstance(t, str) and t for t in node["allowed_tools"])
                ):
                    raise ValueError("Node allowed_tools must be tool names")
                if node.get("program_id") and node["program_id"] not in self.programs:
                    raise ValueError("Dangling Program reference")
                if (
                    node.get("preferred_program_id")
                    and node["preferred_program_id"] not in self.programs
                ):
                    raise ValueError("Dangling preferred implementation")
                for sid in node.get("reference_skill_ids", []):
                    if sid not in self.skills:
                        raise ValueError("Dangling node guidance reference")

                def check_refs(value):
                    if isinstance(value, dict):
                        if "from" in value and value["from"] not in known:
                            raise ValueError("Forward or unknown node output reference")
                        for child in value.values():
                            check_refs(child)
                    elif isinstance(value, list):
                        for child in value:
                            check_refs(child)

                check_refs(node.get("args", {}))
                known.add(node["id"])
            check_refs(s.get("outputs", {})) if s["nodes"] else None
            if "contract" in s:
                from .planning import skeleton_analysis, validate_analysis

                validate_analysis(skeleton_analysis(s, self))

    def retrieve(self, text, limit=8, requirement=None):
        """Eq. 3 for operations after the contract filter; skeletons rank lexically."""
        from .semantic import lexical_similarity, match_requirement

        def similarity(asset):
            return lexical_similarity(
                text,
                " ".join(
                    str(asset.get(k, "")) for k in ("name", "guidance", "applicability")
                ),
            )

        def score(skill):
            return (
                similarity(skill)
                + 0.1 * skill.get("utility", 0.0)
                + 0.05 * bool(skill.get("active", True))
            )

        return {
            "skills": sorted(
                (
                    s
                    for s in self.skills.values()
                    if requirement is None
                    or match_requirement(requirement, s)["status"] == "exact"
                ),
                key=lambda s: (-score(s), s["id"]),
            )[:limit],
            "skeletons": sorted(
                self.skeletons.values(), key=lambda s: (-similarity(s), s["id"])
            )[:limit],
        }
