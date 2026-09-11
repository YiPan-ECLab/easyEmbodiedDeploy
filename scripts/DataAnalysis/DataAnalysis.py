#!/usr/bin/env python3
"""Launch the read-only DataAnalysis web application."""

import argparse

from analysis_app import run_server


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8767, help="use 0 for a free port")
    args = parser.parse_args()
    run_server(args.host, args.port)


if __name__ == "__main__":
    main()
