"""Shared, task-independent wire contracts for model-generated assets."""

ROLE = {
    "oneOf": [
        {
            "type": "object",
            "properties": {
                "kind": {
                    "enum": ["skill_input", "skill_output", "tool_input", "tool_output"]
                },
                "source_role": {
                    "type": "string",
                    "description": "Exact property name in this operation's input_schema or output_schema, never a placeholder.",
                },
            },
            "required": ["kind", "source_role"],
            "additionalProperties": False,
        },
        {
            "type": "object",
            "properties": {"kind": {"const": "constant"}, "value": {}},
            "required": ["kind", "value"],
            "additionalProperties": False,
        },
    ]
}
PREDICATE = {
    "type": "object",
    "properties": {
        "predicate": {"type": "string", "minLength": 1},
        "args": {"type": "object", "additionalProperties": ROLE},
        "effect_domain": {"type": "string"},
        "cardinality": {"type": "integer", "minimum": 1, "default": 1},
        "distinct_by": {"type": "string", "default": ""},
    },
    "required": ["predicate"],
    "additionalProperties": False,
}
INTERFACE = {
    "type": "object",
    "properties": {
        "type": {"const": "object"},
        "properties": {"type": "object", "additionalProperties": {"type": "object"}},
        "required": {"type": "array", "items": {"type": "string"}, "uniqueItems": True},
        "additionalProperties": {"const": False},
    },
    "required": ["type", "properties", "required", "additionalProperties"],
}
CONTRACT = {
    "type": "object",
    "properties": {
        "input_schema": INTERFACE,
        "output_schema": INTERFACE,
        "preconditions": {"type": "array", "items": PREDICATE},
        "effects": {"type": "array", "minItems": 1, "items": PREDICATE},
        "validator": {
            "type": "object",
            "description": "Adapter semantic validator specification, not JSON Schema. Omit unless the adapter declares support.",
        },
        "entry_constraints": {"type": "array", "items": PREDICATE},
    },
    "required": ["input_schema", "output_schema", "preconditions", "effects"],
    "additionalProperties": False,
}
BINDING = {
    "oneOf": [
        {
            "type": "object",
            "properties": {"task": {"type": "string"}},
            "required": ["task"],
            "additionalProperties": False,
        },
        {
            "type": "object",
            "properties": {"from": {"type": "string"}, "field": {"type": "string"}},
            "required": ["from", "field"],
            "additionalProperties": False,
        },
        {
            "type": "object",
            "properties": {"literal": {}},
            "required": ["literal"],
            "additionalProperties": False,
        },
    ]
}
TRIAL = {
    "type": "object",
    "properties": {
        "episode": {"type": "string"},
        "inputs": {"type": "object"},
        "prefix": {
            "type": "array",
            "items": {"type": "integer", "minimum": 0},
            "description": "Executed tool indices strictly BEFORE expected_from.event; never include that event.",
        },
        "expected_from": {
            "type": "object",
            "properties": {
                "event": {"type": "integer", "minimum": 0},
                "path": {"type": "array", "items": {"type": ["string", "integer"]}},
            },
            "required": ["event", "path"],
            "additionalProperties": False,
        },
    },
    "required": ["episode", "inputs", "expected_from"],
    "additionalProperties": False,
}
