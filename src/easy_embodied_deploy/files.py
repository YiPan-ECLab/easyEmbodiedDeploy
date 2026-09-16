"""Manage only the selected model/dataset tree, with reversible deletion."""

import json
import os
import shutil
import stat
import threading
import time
import uuid
from pathlib import Path, PurePosixPath

TRASH = ".workbench-trash"
JOB_NAMES = {
    "record.py",
    "hardware_bridge.py",
    "LerobotDeployer.py",
    "OpenpiDeployer.py",
    "trainer.py",
    "train_jax.py",
    "lerobot-train",
    "DataProcessor.py",
    "DataAnalysis.py",
}


def active_references(repo):
    """Return only path arguments, never environment variables or credentials."""
    found = []
    for process in Path("/proc").iterdir():
        if not process.name.isdigit() or int(process.name) == os.getpid():
            continue
        try:
            args = (
                (process / "cmdline").read_bytes().decode(errors="replace").split("\0")
            )
            if not any(
                Path(arg).name in JOB_NAMES
                for arg in args
                if arg and not any(c.isspace() for c in arg)
            ):
                continue
            cwd = (process / "cwd").resolve(strict=True)
            for arg in args:
                value = arg.split("=", 1)[-1]
                if not value or value.startswith("-") or "\n" in value:
                    continue
                candidate = Path(value).expanduser()
                candidate = (
                    candidate if candidate.is_absolute() else cwd / candidate
                ).resolve()
                if candidate.exists() and any(
                    candidate.is_relative_to(repo / tree)
                    for tree in ("models", "datasets")
                ):
                    found.append((candidate, int(process.name)))
        except (OSError, ValueError):
            continue
    return found


class FileStore:
    def __init__(self, repo, references=None):
        self.repo = Path(repo).resolve()
        self.roots = {name: self.repo / name for name in ("models", "datasets")}
        self.lock = threading.RLock()
        self.references = references or (lambda: active_references(self.repo))

    def root(self, name):
        if not isinstance(name, str) or name not in self.roots:
            raise ValueError("仅支持 models 和 datasets")
        root = self.roots[name]
        if root.is_symlink():
            raise ValueError("管理根目录不能是符号链接")
        root.mkdir(exist_ok=True)
        return root

    def path(self, root_name, relative="", *, entry=False):
        root = self.root(root_name)
        if not isinstance(relative, str) or "\0" in relative or "\\" in relative:
            raise ValueError("无效路径")
        path = PurePosixPath(relative)
        if path.is_absolute() or ".." in path.parts or TRASH in path.parts:
            raise ValueError("路径必须位于所选目录内")
        candidate = root.joinpath(*path.parts)
        parent = candidate.parent if entry and candidate != root else candidate
        if not parent.resolve().is_relative_to(root):
            raise ValueError("符号链接指向管理目录以外")
        # Do not traverse links even when they point inside the tree: rename and
        # delete always refer to the visible entry, never an indirect target.
        current = root
        for part in path.parts[:-1] if entry else path.parts:
            current /= part
            if current.is_symlink():
                raise ValueError("不进入符号链接；可复制其路径或管理链接本身")
        return candidate

    @staticmethod
    def name(value):
        if (
            not isinstance(value, str)
            or value in ("", ".", "..", TRASH)
            or any(c in value for c in ("/", "\\"))
            or any(ord(c) < 32 or ord(c) == 127 for c in value)
        ):
            raise ValueError("名称不能为空，不能包含路径分隔符或控制字符")
        if len(value.encode()) > 255:
            raise ValueError("名称过长")
        return value

    def idle(self, path):
        for used, pid in self.references():
            if path == used or used.is_relative_to(path) or path.is_relative_to(used):
                raise RuntimeError(f"该路径正在被任务 PID {pid} 使用，请先停止对应任务")

    def describe(self, root, path):
        info = path.lstat()
        link = stat.S_ISLNK(info.st_mode)
        return dict(
            name=path.name,
            path=path.relative_to(root).as_posix(),
            absolute=str(path),
            relative=path.relative_to(self.repo).as_posix(),
            kind="link"
            if link
            else "directory"
            if stat.S_ISDIR(info.st_mode)
            else "file",
            size=None if stat.S_ISDIR(info.st_mode) else info.st_size,
            modified=info.st_mtime,
            link_target=os.readlink(path) if link else None,
        )

    def listing(self, root_name, relative=""):
        with self.lock:
            root = self.root(root_name)
            directory = self.path(root_name, relative)
            if not directory.is_dir():
                raise ValueError("目录不存在")
            items = []
            for entry in directory.iterdir():
                if entry.name == TRASH:
                    continue
                try:
                    items.append(self.describe(root, entry))
                except FileNotFoundError:
                    continue
            items.sort(
                key=lambda row: (row["kind"] != "directory", row["name"].casefold())
            )
            usage = shutil.disk_usage(root)
            return dict(
                root=root_name,
                path=directory.relative_to(root).as_posix(),
                absolute=str(directory),
                relative=directory.relative_to(self.repo).as_posix(),
                items=items,
                free_bytes=usage.free,
            )

    def mkdir(self, root_name, relative, name):
        with self.lock:
            target = self.path(root_name, relative) / self.name(name)
            self.idle(target)
            target.mkdir()
            return self.describe(self.root(root_name), target)

    def rename(self, root_name, relative, name):
        with self.lock:
            root = self.root(root_name)
            source = self.path(root_name, relative, entry=True)
            if source == root:
                raise ValueError("不能重命名根目录")
            target = source.with_name(self.name(name))
            if target.exists() or target.is_symlink():
                raise ValueError("同名项目已存在，不会覆盖")
            self.idle(source)
            self.idle(target)
            source.rename(target)
            return self.describe(root, target)

    def trash(self, root_name, relative):
        with self.lock:
            root = self.root(root_name)
            source = self.path(root_name, relative, entry=True)
            if source == root:
                raise ValueError("不能删除根目录")
            source.lstat()
            self.idle(source)
            trash = root / TRASH
            if trash.is_symlink():
                raise ValueError("无效回收站")
            trash.mkdir(exist_ok=True)
            bucket = trash / uuid.uuid4().hex
            bucket.mkdir()
            meta = dict(
                id=bucket.name,
                path=source.relative_to(root).as_posix(),
                name=source.name,
                deleted_at=time.time(),
            )
            try:
                (bucket / "metadata.json").write_text(
                    json.dumps(meta, ensure_ascii=False)
                )
                source.rename(bucket / "item")
            except BaseException:
                shutil.rmtree(bucket)
                raise
            return meta

    def bucket(self, root_name, identifier):
        if (
            not isinstance(identifier, str)
            or len(identifier) != 32
            or any(c not in "0123456789abcdef" for c in identifier)
        ):
            raise ValueError("无效回收站条目")
        trash = self.root(root_name) / TRASH
        bucket = trash / identifier
        if trash.is_symlink() or bucket.is_symlink() or not bucket.is_dir():
            raise ValueError("回收站条目不存在")
        meta_path = bucket / "metadata.json"
        if meta_path.is_symlink():
            raise ValueError("无效回收站元数据")
        return bucket, json.loads(meta_path.read_text())

    def trash_list(self, root_name):
        with self.lock:
            trash = self.root(root_name) / TRASH
            if trash.is_symlink():
                raise ValueError("无效回收站")
            items = []
            for path in trash.iterdir() if trash.exists() else []:
                try:
                    bucket, metadata = self.bucket(root_name, path.name)
                    if (bucket / "item").exists() or (bucket / "item").is_symlink():
                        items.append(metadata)
                except (ValueError, OSError):
                    continue
            return sorted(items, key=lambda x: x["deleted_at"], reverse=True)

    def restore(self, root_name, identifier):
        with self.lock:
            bucket, metadata = self.bucket(root_name, identifier)
            target = self.path(root_name, metadata["path"], entry=True)
            if target == self.root(root_name) or not target.parent.is_dir():
                raise ValueError("原父目录不存在，请先恢复父目录")
            if target.exists() or target.is_symlink():
                raise ValueError("原位置已有同名项目，不会覆盖")
            self.idle(target)
            (bucket / "item").rename(target)
            shutil.rmtree(bucket)
            return metadata

    def purge(self, root_name, identifier, confirmation):
        with self.lock:
            bucket, metadata = self.bucket(root_name, identifier)
            if confirmation != metadata["name"]:
                raise ValueError("请输入完整名称确认永久删除")
            self.idle(bucket)
            # shutil.rmtree uses symlink-safe fd operations on this platform;
            # symlink entries inside the trash are unlinked, never followed.
            shutil.rmtree(bucket)
            return metadata
