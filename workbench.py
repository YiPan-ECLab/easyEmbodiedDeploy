#!/usr/bin/env python3
"""One local entry point for all easyEmbodiedDeploy modules."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from easy_embodied_deploy.workbench import main

if __name__ == "__main__":
    main()
