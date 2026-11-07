"""Program reuse eligibility (Eq. 4): environment compatibility and certificates.

A certificate binds replay/sandbox evidence to one executable body and to the
tool interfaces it was validated against. Any change to either makes the
record inapplicable; the Program then needs fresh validation before reuse.
"""

from .contracts import digest

# Fields that identify neither behavior nor interface.
_NON_EXECUTABLE = {"id", "name", "description", "qualification"}


class Ineligible(ValueError):
    """Bind/Pre/Env/Cert failed before execution; the operation stays agent-guided."""


def program_digest(program):
    return digest({k: v for k, v in program.items() if k not in _NON_EXECUTABLE})


def interface_hash(tools, allowed_tools):
    named = {t["name"]: t for t in tools}
    return digest(
        [
            (
                {k: named[n].get(k) for k in ("name", "input_schema", "output_schema")}
                if n in named
                else {"name": n, "missing": True}
            )
            for n in sorted(set(allowed_tools) - {"read_result"})
        ]
    )


def environment_problem(program, ctx):
    """Env(p, x): every permitted interface exists in the current environment."""
    names = {t["name"] for t in ctx.available_tools()}
    missing = [
        n for n in program.get("allowed_tools", []) if n != "read_result" and n not in names
    ]
    if "read_result" in program.get("allowed_tools", []) and not hasattr(
        ctx.environment, "read_result"
    ):
        missing.append("read_result")
    return "Environment lacks permitted tools: " + ", ".join(missing) if missing else None


def certificate(program, tools, trials, scope="persistent"):
    if not trials or not all(t.get("passed") for t in trials):
        raise ValueError("A certificate needs passing validation evidence")
    return {
        "scope": scope,
        "program_digest": program_digest(program),
        "interface_hash": interface_hash(tools, program["allowed_tools"]),
        "trials": [
            {
                k: t[k]
                for k in ("episode", "inputs_hash", "reference_hash", "result_hash")
                if k in t
            }
            for t in trials
        ],
    }


def certificate_problem(program, ctx):
    """Cert(p, x): the stored validation record still applies to this body and interface."""
    record = program.get("qualification")
    if not isinstance(record, dict) or not record.get("trials"):
        return "Program has no validation certificate"
    if record.get("program_digest") != program_digest(program):
        return "Certificate does not match the current Program body"
    if record.get("interface_hash") != interface_hash(
        ctx.available_tools(), program.get("allowed_tools", [])
    ):
        return "Certificate was issued for different tool interfaces"
    return None


def validate_record(record):
    if not isinstance(record, dict) or record.get("scope") not in {
        "persistent",
        "task_local",
    }:
        raise ValueError("Invalid Program certificate scope")
    for key in ("program_digest", "interface_hash"):
        if not isinstance(record.get(key), str):
            raise ValueError("Program certificate needs " + key)
    if not isinstance(record.get("trials"), list) or not record["trials"]:
        raise ValueError("Program certificate needs validation trials")
