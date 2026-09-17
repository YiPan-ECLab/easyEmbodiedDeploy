"""Pose chunks and valid-target loss for pose and successive joint deltas."""

import torch

from .pose_dataset import PoseDataset, training_stats
from .action_contract import read_pose_contract


def masked_loss(losses, valid, prefix=None, reduction="mean"):
    if reduction not in ("mean", "none"):
        raise ValueError(f"Unsupported loss reduction: {reduction}")
    keep = valid.bool() if prefix is None else valid.bool() & ~prefix
    keep = keep.unsqueeze(-1).expand_as(losses)
    numerator, denominator = (losses*keep).sum((1, 2)), keep.sum((1, 2)).clamp(min=1)
    if reduction == "none":
        return numerator/denominator
    return (numerator/denominator).mean()


def pose_forward(self, batch, reduction="mean"):
    from lerobot.policies.pi05 import modeling_pi05 as native

    images, image_masks = self._preprocess_images(batch)
    states, state_masks = self._prepare_memory_states(batch)
    actions = self.prepare_action(batch)
    valid = ~batch["action_is_pad"].to(device=actions.device).bool()
    if valid.shape != actions.shape[:2]:
        raise ValueError("位姿 action mask 与训练 chunk 形状不一致")
    noise = self.model.sample_noise(actions.shape, actions.device)
    time = self.model.sample_time(actions.shape[0], actions.device)
    prefix = native._sample_training_rtc_prefix_mask(
        actions.shape[0], actions.shape[1], self.config.rtc_training_max_delay, actions.device)
    if prefix is not None:
        # Even a final anchor has one supervised future target. Do not hide it
        # inside an RTC clean prefix and turn the whole sample into zero loss.
        prefix = prefix & (torch.arange(actions.shape[1], device=actions.device)[None] < (valid.sum(1)-1)[:, None])
    losses = self.model.forward(images, image_masks, batch[native.OBS_LANGUAGE_TOKENS],
                               batch[native.OBS_LANGUAGE_ATTENTION_MASK], actions, noise, time,
                               prefix_mask=prefix, states=states, state_masks=state_masks)
    losses = losses[:, :, :14]
    keep = valid if prefix is None else valid & ~prefix
    loss = masked_loss(losses, valid, prefix, reduction)
    per_dim = (losses*keep[..., None]).sum((0, 1))/keep.sum().clamp(min=1)
    return loss, {"loss": loss.detach().mean().item(), "loss_per_dim": per_dim.detach().cpu().tolist()}


# A top-level class is required by LeRobot's spawn-based DataLoader workers.
from lerobot.datasets.lerobot_dataset import LeRobotDataset


class PoseLeRobotDataset(LeRobotDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pose_data = PoseDataset(self.root) if read_pose_contract(self.root) else None
        self.meta.stats = training_stats(self.root, self.meta.stats)

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        if self.pose_data is not None:
            actions, valid = self.pose_data.chunk(int(sample["index"]))
            sample["action"] = torch.from_numpy(actions)
            sample["action_is_pad"] = torch.from_numpy(~valid)
        return sample


def install():
    from lerobot.datasets import factory
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    factory.LeRobotDataset = PoseLeRobotDataset
    PI05Policy.forward = pose_forward


def main():
    install()
    from lerobot.scripts.lerobot_train import main as train
    train()
