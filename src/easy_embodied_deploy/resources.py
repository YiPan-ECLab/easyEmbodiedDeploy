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
                "Piper 设备已被采集、标定、工作流或部署进程占用；"
                "请先在占用页面断开设备或停止硬件服务。工作流执行结束/停止不等于断开设备"
            ) from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)
