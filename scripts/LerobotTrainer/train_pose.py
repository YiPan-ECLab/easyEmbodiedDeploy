"""Run native LeRobot training with frozen-anchor pose chunks and loss masks."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from easy_embodied_deploy.pose_training_torch import main

if __name__ == "__main__":
    main()
