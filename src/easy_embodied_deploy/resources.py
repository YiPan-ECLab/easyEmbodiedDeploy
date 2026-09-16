"""Cross-process ownership of physical hardware; acquiring a lock never moves it."""

import fcntl
import os
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def hardware_lease(name="piper"):
    directory = Path(
        os.environ.get("XDG_RUNTIME_DIR", f"/tmp/easy-embodied-{os.getuid()}")
    )
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    path = directory / f"easy-embodied-{name}.lock"
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                "Piper 设备已被另一采集或部署进程占用，请先在对应页面停止"
            ) from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)
