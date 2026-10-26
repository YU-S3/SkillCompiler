"""An OS-held run lock: released on process exit, including a crash."""

from contextlib import contextmanager
from pathlib import Path
import os


@contextmanager
def run_lock(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    handle = (folder / "run.lock").open("a+b")
    try:
        if os.name == "nt":
            import msvcrt

            handle.write(b"0")
            handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        handle.close()
