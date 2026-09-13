from __future__ import annotations

import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np


MODULE_DIR = Path(__file__).resolve().parents[1]
TRAINER_DIR = MODULE_DIR.parent / "OpenpiTrainer"
sys.path.insert(0, str(MODULE_DIR))
sys.path.insert(0, str(TRAINER_DIR))

from openpi_config import asset_id  # noqa: E402
from runtime import (  # noqa: E402
    AsyncRTCActionEngine,
    PolicyChunkSampler,
    RTCSettings,
    SyncActionEngine,
    policy_observation,
    resolve_checkpoint,
)


def observation() -> dict[str, np.ndarray]:
    result = {"observation.state": np.zeros(14, dtype=np.float32)}
    for name in ("agentview", "left_eye_in_hand", "right_eye_in_hand"):
        result[f"observation.images.{name}"] = np.zeros((224, 224, 3), dtype=np.uint8)
    return result


class FakePolicy:
    def __init__(self, delay_s: float = 0.0) -> None:
        self.delay_s = delay_s
        self.requests: list[dict] = []
        self.calls = 0

    def infer(self, request: dict) -> dict:
        self.requests.append(request)
        self.calls += 1
        if self.delay_s:
            time.sleep(self.delay_s)
        values = np.arange(50 * 14, dtype=np.float32).reshape(50, 14)
        return {"actions": values + 1000 * self.calls}


class RuntimeTest(unittest.TestCase):
    def test_asset_id_is_portable(self) -> None:
        self.assertEqual(
            asset_id("local/piper", "quantiles"),
            asset_id("local/piper", "quantiles"),
        )
        self.assertNotEqual(
            asset_id("local/piper", "quantiles"),
            asset_id("local/piper", "mean_std"),
        )

    def test_resolve_latest_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for step in (1, 12):
                (root / str(step) / "params").mkdir(parents=True)
                (root / str(step) / "params/_METADATA").touch()
                (root / str(step) / "assets").mkdir()
            self.assertEqual(resolve_checkpoint(root), root / "12")
            self.assertEqual(resolve_checkpoint(root / "12/params"), root / "12")

    def test_policy_observation_rejects_wrong_camera_shape(self) -> None:
        value = observation()
        value["observation.images.agentview"] = np.zeros((3, 224, 224), dtype=np.uint8)
        with self.assertRaisesRegex(RuntimeError, "agentview"):
            policy_observation(value, "task")

    def test_guided_request_uses_explicit_leftover_and_masks_grippers(self) -> None:
        policy = FakePolicy()
        sampler = PolicyChunkSampler(policy, "task", RTCSettings(horizon=20))
        first = sampler.infer(observation())
        sampler.infer(observation(), prev_action_chunk=first[7:], prefix_start=3)
        request = policy.requests[-1]
        np.testing.assert_array_equal(request["prev_action_chunk"][:43], first[7:])
        np.testing.assert_array_equal(request["prev_action_chunk"][43:], 0)
        self.assertEqual(request["prefix_attention_start"], 3)
        self.assertEqual(request["prefix_attention_horizon"], 20)
        self.assertEqual(request["rtc_sample_mode"], "guidance")
        self.assertEqual(request["guidance_action_mask"][6], 0)
        self.assertEqual(request["guidance_action_mask"][13], 0)

    def test_trained_request_uses_direct_prefix(self) -> None:
        policy = FakePolicy()
        sampler = PolicyChunkSampler(policy, "task", RTCSettings(mode="trained", horizon=10))
        first = sampler.infer(observation())
        sampler.infer(observation(), prev_action_chunk=first[2:], prefix_start=4)
        self.assertEqual(policy.requests[-1]["rtc_sample_mode"], "train_time")
        self.assertEqual(policy.requests[-1]["prefix_attention_start"], 4)

    def test_sync_reuses_requested_number_of_actions(self) -> None:
        policy = FakePolicy(delay_s=0.01)
        sampler = PolicyChunkSampler(policy, "task", None)
        engine = SyncActionEngine(sampler, 3)
        outputs = [engine.get_action(observation()) for _ in range(4)]
        self.assertEqual(policy.calls, 2)
        self.assertTrue(all(output is not None for output in outputs))
        self.assertEqual(engine.n_action_steps, 3)
        np.testing.assert_array_equal(outputs[0], policy.requests and np.arange(14) + 1000)
        engine.close()

    def test_async_rtc_replaces_queue_and_skips_elapsed_frames(self) -> None:
        policy = FakePolicy(delay_s=0.015)
        settings = RTCSettings(horizon=20, queue_threshold=40, policy_hz=100.0)
        sampler = PolicyChunkSampler(policy, "task", settings)
        engine = AsyncRTCActionEngine(sampler, settings)
        try:
            self.assertIsNone(engine.get_action(observation()))
            self.assertTrue(engine.wait_for_action(1.0))
            while engine.queue_size > 40:
                self.assertIsNotNone(engine.get_action(observation()))
            engine.get_action(observation())
            deadline = time.monotonic() + 1.0
            while len(engine.enqueue_events) < 2 and time.monotonic() < deadline:
                time.sleep(0.005)
            self.assertGreaterEqual(len(engine.enqueue_events), 2)
            first_event = engine.enqueue_events[0]
            event = engine.enqueue_events[-1]
            self.assertEqual(event["start"], min(event["actual_infer_frames"], 50))
            self.assertEqual(event["pushed"], 50 - event["start"])
            self.assertGreaterEqual(event["conditioned_delay"], first_event["actual_infer_frames"])
            self.assertGreaterEqual(event["prev_action_steps"], 1)
            self.assertNotIn("prev_chunk_consumed", event)
            self.assertIsNotNone(event["guidance_horizon"])
        finally:
            engine.close()

    def test_trained_rtc_conditions_on_measured_latency_not_training_limit(self) -> None:
        policy = FakePolicy(delay_s=0.01)
        settings = RTCSettings(
            mode="trained",
            horizon=10,
            queue_threshold=40,
            policy_hz=100.0,
            training_max_delay=5,
        )
        sampler = PolicyChunkSampler(policy, "task", settings)
        engine = AsyncRTCActionEngine(sampler, settings)
        try:
            self.assertTrue(engine.prefill(observation(), timeout_s=1.0))
            while engine.queue_size > settings.queue_threshold:
                self.assertIsNotNone(engine.get_action(observation()))
            deadline = time.monotonic() + 1.0
            while len(engine.enqueue_events) < 2 and time.monotonic() < deadline:
                time.sleep(0.005)
            self.assertGreaterEqual(len(engine.enqueue_events), 2)
            event = engine.enqueue_events[-1]
            # 10 ms at 100 Hz is about one frame; it must not be replaced by
            # the checkpoint's train-time maximum of five frames.
            self.assertLess(event["conditioned_delay"], settings.training_max_delay)
            self.assertGreaterEqual(event["conditioned_delay"], 1)
        finally:
            engine.close()

    def test_async_prefill_starts_with_an_action_ready(self) -> None:
        policy = FakePolicy(delay_s=0.01)
        settings = RTCSettings(horizon=20, queue_threshold=40, policy_hz=100.0)
        sampler = PolicyChunkSampler(policy, "task", settings)
        engine = AsyncRTCActionEngine(sampler, settings)
        try:
            self.assertTrue(engine.prefill(observation(), timeout_s=1.0))
            self.assertIsNotNone(engine.get_action(observation()))
        finally:
            engine.close()

    def test_trained_rtc_preserves_old_queue_then_fails_after_persistent_delay(self) -> None:
        policy = FakePolicy(delay_s=0.03)
        settings = RTCSettings(
            mode="trained",
            horizon=5,
            queue_threshold=5,
            policy_hz=100.0,
            training_max_delay=1,
        )
        sampler = PolicyChunkSampler(policy, "task", settings)
        engine = AsyncRTCActionEngine(sampler, settings)
        try:
            self.assertTrue(engine.prefill(observation(), timeout_s=1.0))
            while engine.queue_size > settings.queue_threshold:
                self.assertIsNotNone(engine.get_action(observation()))
            preserved = engine.queue_size
            deadline = time.monotonic() + 1.0
            while not engine.failed and time.monotonic() < deadline:
                time.sleep(0.005)
            self.assertTrue(engine.failed)
            self.assertEqual(engine.queue_size, preserved)
            discarded = [event for event in engine.enqueue_events if event["discarded"]]
            self.assertEqual(len(discarded), 5)
        finally:
            engine.close()


if __name__ == "__main__":
    unittest.main()
