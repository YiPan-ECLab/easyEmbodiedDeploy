import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import unittest
from copy import deepcopy

import numpy as np

from deployer import (
    PiperActionAdapter,
    features_for_rollout,
    piper_feature_contract,
    validate_checkpoint_contract,
    validate_feature_contract,
)
from hardware_bridge import JOINT_LIMITS, low_pass_absolute_target


class ActionModeFilterContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.state = np.asarray(
            [
                0.0,
                0.5,
                -0.5,
                0.0,
                0.0,
                -0.5,
                0.01,
                0.0,
                0.5,
                -0.5,
                0.0,
                0.0,
                0.5,
                0.02,
            ],
            dtype=np.float64,
        )
        self.desired = np.asarray(
            [
                0.2,
                0.7,
                -0.8,
                0.1,
                -0.1,
                -0.3,
                0.03,
                -0.2,
                0.8,
                -0.7,
                -0.1,
                0.1,
                0.3,
                0.04,
            ],
            dtype=np.float64,
        )

    @staticmethod
    def adapter(mode: str) -> PiperActionAdapter:
        return PiperActionAdapter(
            mode,
            joint_limits=JOINT_LIMITS,
            gripper_limits=(0.0, 0.1),
        )

    def test_delta_and_absolute_reach_the_same_absolute_filter_input(self) -> None:
        delta_adapter = self.adapter("delta_joint")
        absolute_adapter = self.adapter("absolute_joint")
        delta_adapter.reset(self.state)
        absolute_adapter.reset(self.state)

        from_delta = delta_adapter.to_absolute(
            self.desired - self.state, self.state
        )
        from_absolute = absolute_adapter.to_absolute(self.desired, self.state)

        np.testing.assert_allclose(from_delta, self.desired)
        np.testing.assert_allclose(from_absolute, self.desired)
        np.testing.assert_allclose(
            low_pass_absolute_target(
                self.state,
                from_delta,
                control_hz=30.0,
                time_constant_s=0.1,
            ),
            low_pass_absolute_target(
                self.state,
                from_absolute,
                control_hz=30.0,
                time_constant_s=0.1,
            ),
        )

    def test_delta_accumulates_while_absolute_replaces(self) -> None:
        delta_adapter = self.adapter("delta_joint")
        absolute_adapter = self.adapter("absolute_joint")
        delta_adapter.reset(self.state)
        absolute_adapter.reset(self.state)

        delta = (self.desired - self.state) / 2.0
        delta_adapter.to_absolute(delta, self.state)
        np.testing.assert_allclose(
            delta_adapter.to_absolute(delta, self.state), self.desired
        )

        absolute_adapter.to_absolute(self.desired, self.state)
        replacement = self.state.copy()
        replacement[0] = -0.1
        np.testing.assert_allclose(
            absolute_adapter.to_absolute(replacement, self.state), replacement
        )

    def test_zero_time_constant_disables_filter(self) -> None:
        np.testing.assert_array_equal(
            low_pass_absolute_target(
                self.state,
                self.desired,
                control_hz=30.0,
                time_constant_s=0.0,
            ),
            self.desired,
        )


class CheckpointFeatureContractTest(unittest.TestCase):
    @staticmethod
    def valid_checkpoint_config() -> dict:
        cameras = ("agentview", "left_eye_in_hand", "right_eye_in_hand")
        return {
            "input_features": {
                "observation.state": {"type": "STATE", "shape": [14]},
                **{
                    f"observation.images.{camera}": {
                        "type": "VISUAL",
                        "shape": [3, 224, 224],
                    }
                    for camera in cameras
                },
            },
            "output_features": {"action": {"type": "ACTION", "shape": [14]}},
        }

    def test_runtime_features_are_complete_without_a_dataset(self) -> None:
        features = features_for_rollout(piper_feature_contract())
        validate_feature_contract(features)
        self.assertEqual(tuple(features["observation.state"]["shape"]), (14,))
        self.assertEqual(tuple(features["action"]["names"]), tuple(
            f"{side}_{axis}"
            for side in ("left", "right")
            for axis in ("j1", "j2", "j3", "j4", "j5", "j6", "gripper")
        ))

    def test_checkpoint_features_match_piper_and_bad_shape_is_rejected(self) -> None:
        config = self.valid_checkpoint_config()
        validate_checkpoint_contract(config)

        invalid = deepcopy(config)
        invalid["output_features"]["action"]["shape"] = [12]
        with self.assertRaisesRegex(ValueError, r"ACTION\(14,\)"):
            validate_checkpoint_contract(invalid)


if __name__ == "__main__":
    unittest.main()
