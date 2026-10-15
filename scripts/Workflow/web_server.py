#!/usr/bin/env python3
"""Start the workflow editor; no hardware is opened on startup."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from easy_embodied_deploy.workflow.console import main

if __name__ == "__main__":
    main()
