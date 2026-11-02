"""RTC timing in the same action-consumption clock used to splice queues."""

import math
import time
from collections import deque


class RTCTiming:
    def __init__(self, model, policy_hz, control_period):
        self.model = model
        self.hz = float(policy_hz)
        self.period = float(control_period)
        self.samples = deque(maxlen=20)

    def reserve(self, prefix_length):
        if not prefix_length:
            return 0
        config = self.model.config
        observed = max(self.model.latencies, default=0)
        limit = min(prefix_length, int(config.get('rtc_horizon', 20)))
        if config.get('rtc_mode') == 'trained':
            # Include camera/observation capture, executor/RPC dispatch and a
            # control-loop margin, not just GPU RPC time. Never exceed what the
            # checkpoint was trained to condition on.
            observed = max(observed, max(self.samples, default=0))
            estimate = math.ceil((observed+2*self.period)*self.hz)+1
            limit = min(limit, int(self.model.inference.get('training_max_delay', 0)))
        else:
            estimate = max(1, math.ceil(observed*self.hz))
        return max(0, min(estimate, limit))

    def can_consume(self, consumed, reserved, has_prefix):
        """Hold the last target once a trained request's known prefix is spent."""
        if self.model.config.get('rtc_mode') != 'trained' or not has_prefix:
            return True
        maximum = int(self.model.inference.get('training_max_delay', 0))
        return consumed < min(reserved, maximum)

    def complete(self, future, consumed, reserved, has_prefix, submitted_at):
        now = time.monotonic()
        timing = getattr(future, '_workflow_timing', None) or {}
        requested = timing.get('requested_at', submitted_at)
        finished = timing.get('finished_at', now)
        latency = max(0., finished-requested)
        self.samples.append(latency)
        inference_frames = max(0, math.ceil(latency*self.hz-1e-9))
        maximum = int(self.model.inference.get('training_max_delay', 0))
        trained = self.model.config.get('rtc_mode') == 'trained'
        reason = None
        if trained and has_prefix:
            # The consumer stops at the known prefix while inference is late.
            # Queue splicing therefore follows actual consumption, not elapsed
            # wall time. A slow valid response can resume from that boundary.
            if consumed > min(reserved, maximum):
                reason = 'prefix_exhausted'
        return dict(time=time.time(), model_id=self.model.config.get('id'),
                    model_name=self.model.config.get('name'), policy_hz=self.hz,
                    request_ms=round(latency*1000, 3),
                    observation_ms=round(max(0., timing.get('dispatched_at', requested)-requested)*1000, 3),
                    poll_delay_ms=round(max(0., now-finished)*1000, 3),
                    inference_frames=inference_frames,
                    wall_frames=max(0, math.ceil(max(0., now-requested)*self.hz-1e-9)),
                    consumed_frames=int(consumed), reserved_frames=int(reserved),
                    training_max_delay=maximum,
                    latency_exceeds_training_window=bool(trained and inference_frames > maximum),
                    discarded=reason is not None, reason=reason)


def failure_message(report, consecutive):
    reason = ('实际请求耗时超过训练支持范围' if report['reason'] == 'training_limit_exceeded'
              else '实际执行动作超过预留前缀')
    return (f"trained RTC 连续 {consecutive} 次动作前缀不可用：{reason}；"
            f"已执行 {report['consumed_frames']} 帧 / 预留 {report['reserved_frames']} 帧，"
            f"请求 {report['request_ms']:.1f} ms，检查结果延后 {report['poll_delay_ms']:.1f} ms，"
            f"训练补偿上限 {report['training_max_delay']} 帧。请检查推理负载或降低 policy Hz。")
