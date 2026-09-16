import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openpi_config import write_norm_stats


class NormalizationAssetTests(unittest.TestCase):
    def test_same_repo_id_with_different_stats_cannot_overwrite_other_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for index in (0, 1):
                meta = root / str(index) / "meta"
                meta.mkdir(parents=True)
                stats = {
                    key: {
                        name: [float(index)] * 14
                        for name in ("mean", "std", "q01", "q99")
                    }
                    for key in ("action", "observation.state")
                }
                (meta / "stats.json").write_text(json.dumps(stats))
                paths.append(
                    write_norm_stats(
                        meta.parent, root / "runs", "local/shared", "quantiles"
                    )
                )
            self.assertEqual(paths[0][1], paths[1][1])
            self.assertNotEqual(paths[0][0], paths[1][0])
            first = paths[0][0] / paths[0][1] / "norm_stats.json"
            second = paths[1][0] / paths[1][1] / "norm_stats.json"
            self.assertNotEqual(first.read_bytes(), second.read_bytes())


if __name__ == "__main__":
    unittest.main()
