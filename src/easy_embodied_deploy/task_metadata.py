"""Validate LeRobot task references without assuming a single language task."""
import numpy as np


def validate_tasks(root, info, task_indices, episodes, expected_tasks=None):
    import pyarrow.parquet as pq
    errors = []
    path = root / "meta/tasks.parquet"
    if not path.is_file():
        return ["缺少 meta/tasks.parquet"], {}
    rows = pq.read_table(path).to_pydict()
    indices, texts = rows.get("task_index", []), rows.get("task", [])
    count = int(info.get("total_tasks", 0))
    if (count < 1 or len(indices) != count or len(texts) != count
            or sorted(indices) != list(range(count))
            or any(not isinstance(t, str) or not t.strip() for t in texts)):
        return ["tasks.parquet 的任务数量、连续索引或非空指令与 info 不一致"], {}
    if len(set(texts)) != count:
        errors.append("tasks.parquet 中同一指令重复分配了索引")
    mapping = dict(zip(indices, texts))
    if not set(np.unique(task_indices)).issubset(mapping):
        errors.append("数据 task_index 引用了 tasks.parquet 中不存在的指令")
    if expected_tasks is not None:
        if len(expected_tasks) != int(info["total_episodes"]):
            errors.append("处理计划的任务指令数量与 episode 数不一致")
        for ep, text in enumerate(expected_tasks):
            used = np.unique(task_indices[episodes == ep])
            if text is not None and (len(used) != 1 or mapping.get(int(used[0])) != text):
                errors.append(f"episode {ep} 的任务指令与处理计划不一致")
    return errors, mapping
