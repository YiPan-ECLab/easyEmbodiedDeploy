import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from easy_embodied_deploy.pose_training_torch import masked_loss
from easy_embodied_deploy.target_runtime import TorchPolicyAdapter


class MaskTests(unittest.TestCase):
    def test_guided_inference_can_locally_enable_action_gradients(self):
        import numpy as np
        from unittest.mock import patch
        class GuidedPolicy:
            def predict_action_chunk(self, batch, **kwargs):
                with torch.enable_grad():
                    action = torch.ones(1, 50, 14, requires_grad=True)
                    return torch.autograd.grad((action**2).sum(), action)[0]
        adapter = TorchPolicyAdapter(GuidedPolicy(), lambda x: x, lambda x: x, "cpu")
        with patch("lerobot.policies.utils.prepare_observation_for_inference", side_effect=lambda obs, *args: obs):
            result = adapter.infer({"state": np.zeros(14), "images": {}, "prompt": "test"})
        np.testing.assert_array_equal(result["actions"], np.full((50, 14), 2.))

    def test_padding_and_rtc_prefix_never_contribute(self):
        loss = torch.full((2, 50, 14), 1e6, requires_grad=True)
        valid = torch.zeros(2, 50, dtype=torch.bool)
        valid[0, :3] = True; valid[1, :1] = True
        prefix = torch.zeros_like(valid); prefix[0, :2] = True
        with torch.no_grad():
            loss[0, 2] = 2; loss[1, 0] = 4
        result = masked_loss(loss, valid, prefix)
        self.assertEqual(result.item(), 3)
        result.backward()
        self.assertEqual(torch.count_nonzero(loss.grad).item(), 28)
        self.assertEqual(loss.grad[0, :2].abs().sum().item(), 0)
        torch.testing.assert_close(masked_loss(loss, valid, prefix, "none"), torch.tensor([2., 4.]))

    def test_samples_have_equal_weight_despite_different_valid_lengths(self):
        loss = torch.tensor([2., 4.])[:, None, None].expand(2, 50, 14)
        valid = torch.zeros(2, 50, dtype=torch.bool)
        valid[0, :10] = True
        valid[1, :1] = True
        self.assertEqual(masked_loss(loss, valid).item(), 3.)


if __name__ == "__main__":
    unittest.main()
