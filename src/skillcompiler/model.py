"""OpenAI-compatible JSON requests, durable receipts, and conservative admission."""

from pathlib import Path
from threading import RLock
import os
import time
import requests
from .contracts import digest, read, save, validate


class BudgetExceeded(RuntimeError):
    pass


class ResponseContractError(ValueError):
    def __init__(self, message, response):
        super().__init__(message)
        self.response = response


class Ledger:
    def __init__(self, path, token_limit, request_limit):
        self.path = Path(path)
        self.lock = RLock()
        limits = {"tokens": token_limit, "requests": request_limit}
        self.data = (
            read(path) if self.path.exists() else {"limits": limits, "attempts": {}}
        )
        if self.data["limits"] != limits:
            raise ValueError("Budget identity changed")

    def reserve(self, key, amount):
        with self.lock:
            attempts = self.data["attempts"]
            if key in attempts:
                raise ValueError("Repeated physical request")
            used = sum(v.get("tokens", v["reserved"]) for v in attempts.values())
            if (
                used + amount > self.data["limits"]["tokens"]
                or len(attempts) >= self.data["limits"]["requests"]
            ):
                raise BudgetExceeded("Request does not fit the remaining run budget")
            attempts[key] = {
                "reserved": amount,
                "status": "pending",
                "started_at": time.time(),
            }
            save(self.path, self.data)

    def complete(self, key, usage=None, error=None):
        with self.lock:
            row = self.data["attempts"][key]
            if (
                usage
                and isinstance(usage.get("total_tokens"), int)
                and usage["total_tokens"] >= 0
            ):
                row.update(
                    status="reported", tokens=usage["total_tokens"], raw_usage=usage
                )
            else:
                row.update(status="unknown")
            row.update(ended_at=time.time(), error=error)
            save(self.path, self.data)

    def summary(self):
        return {
            "requests": len(self.data["attempts"]),
            "known_tokens": sum(
                v.get("tokens", 0) for v in self.data["attempts"].values()
            ),
            "unknown_reserved": sum(
                v["reserved"]
                for v in self.data["attempts"].values()
                if "tokens" not in v
            ),
        }


class ChatModel:
    def __init__(self, config, root, ledger):
        self.config = dict(config)
        self.root = Path(root)
        self.ledger = ledger
        if {
            "messages",
            "model",
            "max_tokens",
            "stream",
            "tools",
            "tool_choice",
        } & self.config.get("parameters", {}).keys():
            raise ValueError(
                "Generation parameters cannot override request identity or transport"
            )
        if not 1 <= self.config.get("max_attempts", 3) <= 10:
            raise ValueError("Invalid finite attempt limit")

    def ask(self, key, prompt, material, schema):
        import json

        path = self.root / key
        if not path.resolve().is_relative_to(self.root.resolve()):
            raise ValueError("Request key escapes output directory")
        messages = [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": json.dumps(
                    {"input": material, "response_schema": schema}, ensure_ascii=False
                ),
            },
        ]
        payload = {
            "model": self.config["model"],
            "messages": messages,
            "max_tokens": self.config.get("max_tokens", 4096),
            **self.config.get("parameters", {}),
        }
        if self.config.get("json_mode", True):
            payload["response_format"] = {"type": "json_object"}
        # Model-specific generation parameters are explicit configuration, never task dispatch.
        identity = digest([self.config, payload, schema])
        receipt = path / "response.json"
        if receipt.exists():
            value = read(receipt)
            if value["identity"] != identity:
                raise ValueError("Cached request identity changed")
            validate(value["result"], schema)
            return value["result"]
        request = path / "request.json"
        if request.exists() and read(request)["identity"] != identity:
            raise ValueError("Request identity changed")
        save(request, {"identity": identity, "payload": payload, "schema": schema})
        api_key = os.environ.get(self.config.get("api_key_env", "MODEL_API_KEY"))
        if not api_key:
            raise ValueError("Configured API key environment variable is empty")
        attempts = self.config.get("max_attempts", 3)
        for index in range(attempts):
            physical = path / f"attempt_{index}.json"
            ledger_key = digest([str(path.resolve()), index])
            if physical.exists():
                saved = read(physical)
                if "response" in saved:
                    return self._parse(saved["response"], schema, receipt, identity)
                if not saved.get("retryable", False):
                    raise RuntimeError("Prior request failed; inspect its receipt")
                continue
            if ledger_key in self.ledger.data["attempts"]:
                # A crash after admission retains its full reservation; never replay that physical ID.
                save(
                    physical,
                    {"error": "interrupted request; usage unknown", "retryable": True},
                )
                continue
            amount = (
                len(json.dumps(payload, ensure_ascii=False).encode())
                + payload["max_tokens"]
            )
            self.ledger.reserve(ledger_key, amount)
            try:
                response = requests.post(
                    self.config["base_url"].rstrip("/") + "/chat/completions",
                    headers={"Authorization": "Bearer " + api_key},
                    json=payload,
                    timeout=(
                        self.config.get("connect_timeout", 15),
                        self.config.get("timeout", 300),
                    ),
                )
                if not response.ok:
                    retryable = response.status_code in {408, 429, 500, 502, 503, 504}
                    self.ledger.complete(
                        ledger_key, error=f"HTTP {response.status_code}"
                    )
                    save(
                        physical,
                        {"status": response.status_code, "retryable": retryable},
                    )
                    if not retryable:
                        raise RuntimeError(f"Provider HTTP {response.status_code}")
                else:
                    raw = response.json()
                    self.ledger.complete(ledger_key, raw.get("usage"))
                    save(physical, {"response": raw, "received_at": time.time()})
                    return self._parse(raw, schema, receipt, identity)
            except (requests.RequestException, ValueError) as exc:
                if physical.exists():
                    raise
                self.ledger.complete(ledger_key, error=type(exc).__name__)
                save(physical, {"error": type(exc).__name__, "retryable": True})
            if index + 1 < attempts:
                time.sleep(min(2**index, 30))
        raise RuntimeError(
            "Finite request attempts exhausted; receipts and reserved costs are retained"
        )

    @staticmethod
    def _parse(raw, schema, receipt, identity):
        import json

        try:
            result = json.loads(raw["choices"][0]["message"]["content"])
            validate(result, schema)
        except (ValueError, KeyError, TypeError, IndexError) as exc:
            raise ResponseContractError(str(exc), raw) from exc
        save(receipt, {"identity": identity, "result": result, "response": raw})
        return result
