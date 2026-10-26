"""Network-isolated Python worker with bounded, host-validated JSON RPC."""

import json
import os
from pathlib import Path
import selectors
import shutil
import subprocess
import sys
import tempfile
import time
from uuid import uuid4


class Context:
    def __init__(self, rpc):
        self._rpc = rpc

    def observe(self):
        return self._rpc("observe")

    def available_tools(self):
        return self._rpc("available_tools")

    def remaining_calls(self):
        return self._rpc("remaining_calls")

    def call(self, name, arguments):
        return self._rpc("call", name=name, arguments=arguments)

    def read_result(self, result_id, offset=0, limit=None, path=None):
        return self._rpc(
            "read_result", result_id=result_id, offset=offset, limit=limit, path=path
        )


def worker_main():
    output = sys.stdout
    sys.stdout = sys.stderr
    sequence = 0

    def send(data):
        output.write(json.dumps(data, allow_nan=False) + "\n")
        output.flush()

    def rpc(method, **args):
        nonlocal sequence
        sequence += 1
        send({"kind": "rpc", "sequence": sequence, "method": method, "args": args})
        raw = sys.stdin.buffer.readline(1048577)
        if len(raw) > 1048576:
            raise ValueError("RPC response too large")
        result = json.loads(raw)
        if "error" in result:
            raise RuntimeError(result["error"])
        return result["value"]

    try:
        namespace = {"__name__": "skill_program"}
        exec(
            compile(
                Path("/program/source.py").read_text(), "/program/source.py", "exec"
            ),
            namespace,
        )
        result = namespace["run"](
            Context(rpc), json.loads(Path("/program/inputs.json").read_text())
        )
        send({"kind": "done", "result": result})
    except BaseException as exc:
        send({"kind": "error", "error": str(exc)[:4096]})


class ProgramWorker:
    def __init__(self, image, timeout=60, memory_mb=1024, max_rpc_bytes=1048576):
        if not image or not (image.startswith("sha256:") or "@sha256:" in image):
            raise ValueError("Pin a Docker image digest")
        self.image, self.timeout, self.memory_mb, self.max_rpc_bytes = (
            image,
            timeout,
            memory_mb,
            max_rpc_bytes,
        )

    def execute(self, program, inputs, ctx):
        from .contracts import validate

        validate(inputs, program["input_schema"])
        if sys.platform != "linux" or not shutil.which("docker"):
            raise RuntimeError(
                "Docker on Linux is required; host execution is disabled"
            )
        container = "skillcompiler-" + uuid4().hex
        result = None
        start = time.monotonic()
        sequence = 0
        with tempfile.TemporaryDirectory(prefix="skillcompiler-") as d:
            root = Path(d)
            root.chmod(0o755)
            (root / "source.py").write_text(program["source"])
            (root / "inputs.json").write_text(json.dumps(inputs))
            shutil.copyfile(__file__, root / "worker.py")
            for p in root.iterdir():
                p.chmod(0o444)
            command = [
                "docker",
                "run",
                "--rm",
                "-i",
                "--pull=never",
                "--name",
                container,
                "--network=none",
                "--read-only",
                "--user=1000:1000",
                "--cap-drop=ALL",
                "--security-opt=no-new-privileges",
                "--memory",
                str(self.memory_mb) + "m",
                "--memory-swap",
                str(self.memory_mb) + "m",
                "--pids-limit",
                "64",
                "--cpus",
                "1",
                "--tmpfs",
                "/tmp:rw,nosuid,nodev,size=64m",
                "--mount",
                f"type=bind,src={root},dst=/program,readonly",
                self.image,
                "python",
                "-I",
                "-u",
                "/program/worker.py",
            ]
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            selector = selectors.DefaultSelector()
            selector.register(process.stdout, selectors.EVENT_READ, "rpc")
            selector.register(process.stderr, selectors.EVENT_READ, "stderr")
            buffer = b""
            diagnostics = bytearray()
            try:
                while selector.get_map() and result is None:
                    if time.monotonic() - start > self.timeout:
                        raise TimeoutError("Program wall timeout")
                    for key, _ in selector.select(0.2):
                        chunk = os.read(key.fileobj.fileno(), 65536)
                        if not chunk:
                            selector.unregister(key.fileobj)
                            continue
                        if key.data == "stderr":
                            diagnostics.extend(chunk[: max(0, 4096 - len(diagnostics))])
                            continue
                        buffer += chunk
                        if len(buffer) > self.max_rpc_bytes:
                            raise ValueError("RPC message too large")
                        while b"\n" in buffer:
                            line, buffer = buffer.split(b"\n", 1)
                            message = json.loads(line)
                            if message["kind"] == "error":
                                raise RuntimeError(message["error"])
                            if message["kind"] == "done":
                                result = message["result"]
                                break
                            if message.get("sequence") != sequence + 1:
                                raise ValueError("Invalid RPC sequence")
                            sequence += 1
                            method = message["method"]
                            args = message.get("args", {})
                            try:
                                if method not in {
                                    "observe",
                                    "available_tools",
                                    "remaining_calls",
                                    "call",
                                    "read_result",
                                }:
                                    raise ValueError("Unknown RPC method")
                                if (
                                    method == "call"
                                    and args.get("name") not in program["allowed_tools"]
                                ):
                                    raise PermissionError(
                                        "Tool is outside Program permissions"
                                    )
                                if (
                                    method == "read_result"
                                    and "read_result" not in program["allowed_tools"]
                                ):
                                    raise PermissionError(
                                        "Result reader is outside Program permissions"
                                    )
                                value = getattr(ctx, method)(**args)
                                reply = {"value": value}
                            except (ValueError, PermissionError, RuntimeError) as exc:
                                reply = {"error": str(exc)}
                            wire = json.dumps(reply, allow_nan=False).encode() + b"\n"
                            if len(wire) > self.max_rpc_bytes:
                                raise ValueError("Tool result exceeds RPC capacity")
                            process.stdin.write(wire)
                            process.stdin.flush()
                if result is None:
                    raise RuntimeError(
                        "Worker exited without output: "
                        + diagnostics.decode(errors="replace")
                    )
            finally:
                selector.close()
                subprocess.run(
                    ["docker", "rm", "-f", container],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=15,
                )
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=15)
                for stream in [process.stdin, process.stdout, process.stderr]:
                    stream.close()
        if result.get("status") not in {"ok", "not_found", "needs_input", "blocked"}:
            raise ValueError("Invalid Program status")
        if not isinstance(result.get("outputs"), dict):
            raise ValueError("Program outputs must be an object")
        if result["status"] == "ok":
            validate(result["outputs"], program["output_schema"])
        return result


if __name__ == "__main__":
    worker_main()
