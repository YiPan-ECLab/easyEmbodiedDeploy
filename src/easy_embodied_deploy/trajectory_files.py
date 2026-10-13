"""Shared reversible HDF5 deletion and conversion read leases."""

import fcntl
import hashlib
from contextlib import ExitStack, contextmanager
from pathlib import Path

from .files import FileStore, active_references

REPO = Path(__file__).resolve().parents[2]


def revision(path):
    st = Path(path).stat()
    return hashlib.sha256(f"{st.st_dev}:{st.st_ino}:{st.st_size}:{st.st_mtime_ns}".encode()).hexdigest()


def _store(path):
    path = Path(path).expanduser().absolute()
    if path.suffix.lower() not in (".hdf5", ".h5") or path.is_symlink():
        raise ValueError("只能删除普通 HDF5 轨迹文件")
    root = REPO / "datasets"
    if not path.is_relative_to(root):
        root = path.parent

    def references():
        for used, pid in active_references(REPO):
            # Recorder writes new episodes atomically via *.tmp. An older,
            # completed episode can be removed while collection continues.
            try:
                args = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
                recording = any(Path(arg.decode()).name == "record.py" for arg in args if arg)
            except OSError:
                recording = False
            if recording and used.is_dir():
                continue
            yield used, pid

    store = FileStore(REPO, references=references)
    store.roots["datasets"] = root
    relative = path.relative_to(root).as_posix()
    store.path("datasets", relative)  # Reject symlink traversal before mutation.
    return store, relative


def trash_trajectory(raw_path, expected_revision):
    path = Path(raw_path).expanduser().absolute()
    store, relative = _store(path)
    if not path.is_file():
        raise ValueError("轨迹文件不存在")
    with path.open("rb") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("轨迹正在读取、转换或写入，请稍后再试") from exc
        if not expected_revision or revision(path) != expected_revision:
            raise ValueError("文件已变化，请重新检查后再删除")
        return {**store.trash("datasets", relative), "original_path": str(path),
                "message": "已移入回收站，可撤销删除"}


def restore_trajectory(raw_path, identifier):
    store, relative = _store(raw_path)
    _, meta = store.bucket("datasets", identifier)
    if meta["path"] != relative:
        raise ValueError("回收站条目与原路径不匹配")
    return store.restore("datasets", identifier)


@contextmanager
def hold_trajectories(files):
    """Keep all planned sources protected against UI deletion until completion."""
    with ExitStack() as stack:
        for path, expected in files:
            handle = stack.enter_context(Path(path).open("rb"))
            try:
                fcntl.flock(handle, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(f"文件正在修改：{path}") from exc
            if expected and revision(path) != expected:
                raise ValueError(f"文件已变化，请重新检查：{path}")
        yield
