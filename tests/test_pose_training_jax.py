import os
import sys
import unittest
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"
import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from easy_embodied_deploy.pose_training_jax import masked_loss


class MaskTests(unittest.TestCase):
    def test_padding_has_zero_gradient_and_each_sample_keeps_scale(self):
        loss = jnp.ones((2, 50))*1e6
        loss = loss.at[0, 0].set(2).at[1, :3].set(4)
        valid = jnp.arange(50)[None, :] < jnp.array([[1], [3]])
        result = masked_loss(loss, valid)
        np.testing.assert_allclose(result.mean(axis=1), [2, 4])
        gradient = jax.grad(lambda values: masked_loss(values, valid).mean())(loss)
        self.assertEqual(np.count_nonzero(gradient), 4)


if __name__ == "__main__":
    unittest.main()
