"""Optional structured operator input for record.py; no robot control here."""

import json
import os
import queue
import sys
import threading
import time

EVENT_PREFIX = "PIPER_RECORDER_EVENT="


class WebOperator:
    def __init__(self):
        self.messages = queue.Queue(maxsize=32)
        self.disconnected = threading.Event()
        self.token = 0
        self.phase_name = "initializing"
        self.accepts = []
        self.last_progress = 0.0
        threading.Thread(target=self._read, daemon=True, name="operator-input").start()

    def _read(self):
        try:
            for line in sys.stdin:
                if len(line) > 16384:
                    continue
                try:
                    message = json.loads(line)
                    if isinstance(message, dict):
                        self.messages.put_nowait(message)
                except (ValueError, queue.Full):
                    continue
        finally:
            self.disconnected.set()

    def emit(self, **details):
        event = dict(
            phase=self.phase_name, token=self.token, accepts=self.accepts, **details
        )
        os.write(
            sys.stdout.fileno(),
            (
                "\n" + EVENT_PREFIX + json.dumps(event, ensure_ascii=False) + "\n"
            ).encode(),
        )

    def phase(self, name, *, accepts=(), **details):
        self.token += 1
        self.phase_name = name
        self.accepts = list(accepts)
        self.emit(**details)

    def update(self, name=None, **details):
        if name:
            self.phase_name = name
        now = time.monotonic()
        if name or now - self.last_progress >= 0.25:
            self.last_progress = now
            self.emit(**details)

    def _take(self):
        if self.disconnected.is_set():
            raise EOFError("Web 操作通道已关闭")
        while True:
            try:
                message = self.messages.get_nowait()
            except queue.Empty:
                return None
            if (
                message.get("token") == self.token
                and message.get("action") in self.accepts
            ):
                # Invalidate this phase before another queued pedal event is read.
                self.accepts = []
                return message

    def prompt(self, name, message):
        self.phase(name, accepts=("submit",), message=message)
        while True:
            command = self._take()
            if command is not None:
                return str(command.get("value", "")).strip()
            time.sleep(0.02)

    def stop_requested(self):
        return self._take() is not None
