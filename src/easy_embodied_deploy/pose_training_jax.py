"""OpenPI pose training retains the native model and masks invalid targets."""

import dataclasses

import flax.nnx as nnx
from flax import struct
import jax
import jax.numpy as jnp
from openpi.models import model as model_module, pi0, pi0_config


@struct.dataclass
class PoseObservation(model_module.Observation):
    action_valid: object = None


def masked_loss(loss, valid):
    count = jnp.maximum(jnp.sum(valid, axis=-1), 1)
    return loss*valid*(loss.shape[-1]/count)[..., None]


class PosePi0(pi0.Pi0):
    def compute_loss(self, rng, observation, actions, *, train=False):
        valid = observation.action_valid
        if valid is None or valid.shape != actions.shape[:2]:
            raise ValueError("位姿训练缺少正确形状的 action_valid")
        preprocess_rng, noise_rng, time_rng, delay_rng = jax.random.split(rng, 4)
        observation = model_module.preprocess_observation(preprocess_rng, observation, train=train)
        shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, shape)*.999+.001
        xt = time[..., None, None]*noise+(1-time[..., None, None])*actions
        ut = noise-actions
        delay_max = min(int(self.train_time_rtc_max_delay), self.action_horizon-1)
        keep = valid
        if train and delay_max > 0:
            delay = jax.random.randint(delay_rng, shape, 0, delay_max+1)
            delay = jnp.minimum(delay, jnp.maximum(jnp.sum(valid, axis=-1)-1, 0))
            prefix = jnp.arange(self.action_horizon)[None] < delay[..., None]
            time = jnp.where(prefix, 0., time[..., None])
            xt = jnp.where(prefix[..., None], actions, xt)
            keep = valid & ~prefix
        prefix_tokens, prefix_mask, prefix_ar = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar, condition = self.embed_suffix(observation, xt, time)
        inputs = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar = jnp.concatenate([prefix_ar, suffix_ar], axis=0)
        (_, suffix), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=pi0.make_attn_mask(inputs, ar),
            positions=jnp.cumsum(inputs, axis=1)-1, adarms_cond=[None, condition])
        vt = self.action_out_proj(suffix[:, -self.action_horizon:])
        # Internal PI05 padding remains 32; only the 14 physical action axes
        # are supervised. The user's state and action contracts remain 14-D.
        loss = jnp.mean(jnp.square(vt[..., :14]-ut[..., :14]), axis=-1)
        return masked_loss(loss, keep)


@dataclasses.dataclass(frozen=True)
class PosePi0Config(pi0_config.Pi0Config):
    def create(self, rng):
        return PosePi0(self, rngs=nnx.Rngs(rng))


def install_loader():
    from openpi.training import data_loader

    class PoseDataLoader(data_loader.DataLoaderImpl):
        def __iter__(self):
            for batch in self._data_loader:
                observation = model_module.Observation.from_dict(batch)
                if "action_valid" in batch:
                    observation = PoseObservation(**vars(observation), action_valid=batch["action_valid"])
                yield observation, batch["actions"]

    data_loader.DataLoaderImpl = PoseDataLoader
