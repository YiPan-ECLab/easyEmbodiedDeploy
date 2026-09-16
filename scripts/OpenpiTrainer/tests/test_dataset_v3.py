import json
import sys
import tempfile
import unittest
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset_v3 import LeRobotV3Dataset, _fixed_vectors


class DatasetBoundaryTests(unittest.TestCase):
    def fixture(self, root, episodes):
        for sub in ("meta", "meta/episodes/chunk-000", "data/chunk-000"):
            (root / sub).mkdir(parents=True, exist_ok=True)
        (root / "meta/info.json").write_text(
            json.dumps(
                dict(
                    codebase_version="v3.0",
                    fps=30,
                    total_frames=4,
                    total_episodes=len(episodes),
                )
            )
        )
        pq.write_table(
            pa.table(
                {
                    "observation.state": [[0.0] * 14] * 4,
                    "action": [[float(i)] * 14 for i in range(4)],
                    "timestamp": [0.0, 1 / 30, 0.0, 1 / 30],
                    "episode_index": [0, 0, 1, 1],
                    "task_index": [0] * 4,
                }
            ),
            root / "data/chunk-000/file-000.parquet",
        )
        pq.write_table(
            pa.table({"task_index": [0], "task": ["test"]}), root / "meta/tasks.parquet"
        )
        pq.write_table(
            pa.Table.from_pylist(episodes),
            root / "meta/episodes/chunk-000/file-000.parquet",
        )

    def test_episode_tail_padding_does_not_cross_episode(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.fixture(
                root,
                [
                    dict(episode_index=0, dataset_from_index=0, dataset_to_index=2),
                    dict(episode_index=1, dataset_from_index=2, dataset_to_index=4),
                ],
            )
            dataset = LeRobotV3Dataset(root, 3)
            dataset._decode_image = lambda *args: None
            self.assertEqual(dataset[1]["action"][:, 0].tolist(), [1.0, 1.0, 1.0])
            self.assertEqual(dataset[2]["action"][:, 0].tolist(), [2.0, 3.0, 3.0])

    def test_missing_overlap_and_mismatched_episode_ranges_are_rejected(self):
        for start, end in ((3, 4), (1, 4), (2, 3)):
            with (
                self.subTest(start=start, end=end),
                tempfile.TemporaryDirectory() as temp,
            ):
                root = Path(temp)
                self.fixture(
                    root,
                    [
                        dict(episode_index=0, dataset_from_index=0, dataset_to_index=2),
                        dict(
                            episode_index=1,
                            dataset_from_index=start,
                            dataset_to_index=end,
                        ),
                    ],
                )
                with self.assertRaises(ValueError):
                    LeRobotV3Dataset(root, 3)

    def test_variable_length_vectors_cannot_be_silently_reshaped(self):
        table = pa.table({"action": [[0.0] * 13, [0.0] * 15]})
        with self.assertRaises(ValueError):
            _fixed_vectors(table, "action", 14)


if __name__ == "__main__":
    unittest.main()
