"""Reliable ZeroMQ client for the Piper hardware process."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import zmq


PROTOCOL_VERSION = 3


class BridgeError(RuntimeError):
    """The hardware bridge rejected a request or returned an invalid reply."""


class BridgeTimeout(BridgeError):
    """The hardware bridge did not answer before the RPC deadline."""


class PiperBridge:
    """One-request/one-response RPC client with recovery after timeouts."""

    def __init__(
        self,
        endpoint: str = "tcp://127.0.0.1:6007",
        *,
        timeout_s: float = 5.0,
    ) -> None:
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        self.endpoint = endpoint
        self.timeout_ms = max(1, round(timeout_s * 1000))
        self._context = zmq.Context()
        self._socket: zmq.Socket | None = None
        self._request_id = 0
        self._open_socket()

    def _open_socket(self) -> None:
        if self._socket is not None:
            self._socket.close(linger=0)
        socket = self._context.socket(zmq.REQ)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        socket.connect(self.endpoint)
        self._socket = socket

    def _request(self, operation: str, **payload: Any) -> dict[str, Any]:
        self._request_id += 1
        request_id = self._request_id
        request = {
            "protocol": PROTOCOL_VERSION,
            "request_id": request_id,
            "operation": operation,
            **payload,
        }
        try:
            assert self._socket is not None
            self._socket.send_pyobj(request)
            reply = self._socket.recv_pyobj()
        except zmq.Again as exc:
            self._open_socket()
            raise BridgeTimeout(
                f"hardware bridge timed out during {operation!r} at {self.endpoint}"
            ) from exc
        except zmq.ZMQError as exc:
            self._open_socket()
            raise BridgeError(
                f"hardware bridge transport failed during {operation!r}: {exc}"
            ) from exc

        if not isinstance(reply, dict):
            raise BridgeError(f"invalid {operation!r} reply type: {type(reply).__name__}")
        if reply.get("protocol") != PROTOCOL_VERSION:
            raise BridgeError(
                f"bridge protocol mismatch: client={PROTOCOL_VERSION}, "
                f"server={reply.get('protocol')!r}; restart hardware_bridge.py"
            )
        if reply.get("request_id") != request_id:
            raise BridgeError(
                f"bridge response id mismatch: sent={request_id}, "
                f"received={reply.get('request_id')!r}"
            )
        if not reply.get("ok", False):
            raise BridgeError(str(reply.get("error", f"{operation!r} failed")))
        return reply

    def ping(self) -> dict[str, Any]:
        return self._request("ping")

    def configure(
        self,
        *,
        control_hz: float,
        smoothing_time_constant_s: float,
        watchdog_timeout_s: float,
    ) -> dict[str, Any]:
        return self._request(
            "configure",
            control_hz=float(control_hz),
            smoothing_time_constant_s=float(smoothing_time_constant_s),
            watchdog_timeout_s=float(watchdog_timeout_s),
        )

    def status(self) -> dict[str, Any]:
        return self._request("status")

    def observation(self) -> dict[str, Any]:
        reply = self._request("observe")
        observation = reply.get("observation")
        if not isinstance(observation, dict):
            raise BridgeError("bridge observe reply has no observation mapping")
        return observation

    def action(self, target: np.ndarray) -> dict[str, Any]:
        values = np.asarray(target, dtype=np.float64).reshape(-1)
        if values.shape != (14,) or not np.isfinite(values).all():
            raise ValueError("Piper target must contain 14 finite values")
        return self._request("act", target=values.tolist())

    def stop(self) -> None:
        try:
            self._request("stop")
        finally:
            self.close()

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close(linger=0)
            self._socket = None
        if not self._context.closed:
            self._context.term()

    def __enter__(self) -> "PiperBridge":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
