"""An asynchronous policy sampler; only the controller may command hardware."""

import time
from collections import deque

import numpy as np

from .rtc_timing import RTCTiming, failure_message


class ContinuousPolicy:
    def __init__(self, controller, config):
        self.controller = controller
        self.config = config
        self.model = controller.models.get(config['model'])
        self.queue = deque()
        self.pending = None
        self.generation = 0
        self.pending_generation = 0
        self.suppressed = False
        self.desired = None
        self.next_policy = 0
        self.actions = self.chunks = self.pending_consumed = self.discarded = 0
        self.conditioned_delay = 0
        self.pending_prefix = False
        self.submitted_at = 0
        self.waiting_for_inference = False
        self.timing = RTCTiming(self.model, self.model.config.get('policy_hz', 10),
                                1/float(self.model.config.get('control_hz', 50)))

    def reset(self):
        # A running request may complete, but its result cannot cross ownership
        # or pause boundaries. Do not queue another RPC behind that request.
        self.generation += 1
        self.queue.clear()
        self.desired = None
        self.next_policy = 0
        self.discarded = 0
        self.waiting_for_inference = False

    def suppress(self, value):
        self.suppressed = value
        self.reset()

    def tick(self):
        model, controller = self.model, self.controller
        if model.stage == 'failed':
            raise RuntimeError(f"持续 VLA 加载失败：{model.detail}")
        if model.stage != 'ready':
            return
        config = model.config
        hz = float(config.get('policy_hz', 10))
        rtc = config.get('inference_mode', 'rtc') == 'rtc'
        now = time.monotonic()
        if self.pending is not None and self.pending.done():
            if not self.pending.cancelled() and self.pending_generation == self.generation:
                chunk = self.pending.result()
                self.chunks += 1
                report = self.timing.complete(self.pending, self.pending_consumed,
                                              self.conditioned_delay, self.pending_prefix,
                                              self.submitted_at)
                if self.pending_generation == self.generation and not self.suppressed:
                    if rtc and report['discarded']:
                        self.discarded += 1
                    else:
                        self.discarded = 0
                        if not rtc:
                            chunk = chunk[:model.inference.get('action_steps', 50)]
                        self.queue = deque(chunk[min(self.pending_consumed, len(chunk)):])
                record = getattr(controller, '_record_rtc', None)
                if record:
                    record(report, self.discarded)
                if not self.suppressed and self.discarded >= 5:
                    raise RuntimeError(failure_message(report, self.discarded))
            self.pending = None
        threshold = int(config.get('rtc_queue_threshold', 40)) if rtc else 0
        if self.pending is None and (self.suppressed or len(self.queue) <= threshold):
            # Fixed-point control never claims that unexecuted VLA actions were
            # executed. Observe its current state with no stale RTC prefix.
            prefix = np.array(self.queue) if rtc and self.queue and not self.suppressed else None
            delay = self.timing.reserve(len(prefix)) if prefix is not None else 0
            self.submitted_at = now
            self.pending = controller._submit_model(model, self.config.get('task') or config.get('task', ''), prefix, delay)
            self.pending_generation = self.generation
            self.pending_consumed = 0
            self.pending_prefix = prefix is not None
            self.conditioned_delay = delay
        can_consume = (self.pending is None or
                       self.timing.can_consume(self.pending_consumed, self.conditioned_delay,
                                               self.pending_prefix))
        self.waiting_for_inference = not self.suppressed and not can_consume
        if not self.suppressed and can_consume and now >= self.next_policy and self.queue:
            self.desired = np.asarray(self.queue.popleft()).copy()
            self.actions += 1
            if self.pending is not None:
                self.pending_consumed += 1
            self.next_policy = now+1/hz
