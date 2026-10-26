"""Public task interfaces. Evaluator state stays behind TaskSuite.score()."""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol
import hashlib
import json
import os
from uuid import uuid4
from jsonschema import Draft202012Validator


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    temp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temp, path)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate(value, schema):
    Draft202012Validator.check_schema(schema)
    errors = sorted(
        Draft202012Validator(schema).iter_errors(value), key=lambda e: str(list(e.path))
    )
    if errors:
        raise ValueError("; ".join(f"{list(e.path)}: {e.message}" for e in errors))


@dataclass(frozen=True)
class Task:
    id: str
    instruction: str
    inputs: dict = field(default_factory=dict)

    def public(self):
        return asdict(self)


@dataclass(frozen=True)
class Feedback:
    score: float
    success: bool
    detail: dict = field(default_factory=dict)


class Environment(Protocol):
    """Each open() must return an isolated, reset environment for this task."""

    def observe(self) -> dict: ...
    def tools(self) -> list[dict]: ...
    def call(self, name: str, arguments: dict) -> Any: ...
    def close(self) -> None: ...


class TaskSuite(Protocol):
    def tasks(self, split: str) -> list[Task]: ...
    def open(self, task: Task) -> Environment: ...
    def score(self, task: Task, answer: Any, environment: Environment) -> Feedback: ...


OBJECT = {"type": "object"}
