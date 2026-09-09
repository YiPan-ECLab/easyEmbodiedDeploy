#!/home/py/main/easyEmbodiedDeploy/thirdParty/.venvs/piper-home/bin/python
"""Launch the local Piper HDF5 data viewer."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from piper_viewer.server import run_server


DEFAULT_DIRECTORY = Path(
    "/home/py/main/easyEmbodiedDeploy/datasets/hdf5/pyagxarm_support"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--directory",
        type=Path,
        default=DEFAULT_DIRECTORY,
        help="initial directory containing .hdf5/.h5 files",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    return run_server(args.host, args.port, args.directory)


if __name__ == "__main__":
    raise SystemExit(main())
