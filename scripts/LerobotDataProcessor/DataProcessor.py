#!/usr/bin/env python3
"""Launch the local DataProcessor web application."""

from __future__ import annotations

import argparse

from app import run_server


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--port", type=int, default=8766,
        help="listen port; use 0 to select a free port automatically",
    )
    args = parser.parse_args()
    run_server(args.host, args.port)


if __name__ == "__main__":
    main()
