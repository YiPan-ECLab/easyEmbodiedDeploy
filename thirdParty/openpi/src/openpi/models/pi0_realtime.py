"""Real-time chunking for π0 (Black et al., 2024).

Adds prefix-attention guidance to ``Pi0.sample_actions``: during the
flow-matching denoising loop, the new chunk's prefix is softly pulled
toward a previously generated chunk's tail so that the trajectory does
not jump at chunk boundaries.

Algorithm reference: ``real-time-chunking-kinetix/src/model.py`` →
``Model.realtime_action`` (lines ~219–286). This file ports that logic
to the π0 / π0.5 flow-matching loop in :mod:`openpi.models.pi0`.

Time conventions
----------------
- kinetix model: ``t=0`` is noise, ``t=1`` is clean, ``dt = +1/N``.
- π0 model:      ``t=1`` is noise, ``t=0`` is clean, ``dt = -1/N``.

The denoiser (one-shot prediction of the clean target from ``x_t`` at
time ``t``) and the paper's two scalars therefore become, in π0
convention:

    x_clean = x_t + v_t * (0 - t) = x_t - v_t * t
    c       = t / (1 - t)
    inv_r2  = (t² + (1-t)²) / t²

(both obtained by substituting ``t_kinetix = 1 - t`` into the kinetix
formulae). Kinetix / paper convention uses a positive-time sampler
``x <- x + dt * v`` with ``dt > 0``, so the correction is written as
``v + guidance * vjp(error)``. In π0 inference we integrate from noise
``t=1`` down to clean ``t=0`` with ``dt < 0``. Keeping the same pull on
the clean prediction therefore appears in code as::

    v' = v - min(c * inv_r2, max_guidance_weight) * vjp(denoiser)(error)

where ``error = (prev_action_chunk - x_clean) * weights[:, None]``.

Backward compatibility
----------------------
:func:`realtime_sample_actions` accepts the same kwargs as the original
``Pi0.sample_actions`` plus a few optional ones. When
``prev_action_chunk is None`` it simply runs the original sampling loop,
so installing the patch is safe for callers that never pass realtime
fields.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import einops
import jax
import jax.numpy as jnp

from openpi.models import model as _model
from openpi.models import pi0 as _pi0
from openpi.models.pi0 import make_attn_mask

logger = logging.getLogger("openpi")

PrefixAttentionSchedule = str  # one of: "exp" | "linear" | "ones" | "zeros"


def _dim_weights_from_env(action_dim: int) -> jax.Array:
    """Optional per-action-dim RTC guidance multipliers.

    ``OPENPI_RT_GUIDANCE_DIM_WEIGHTS`` is a comma-separated list. Missing
    values are padded with 1.0; extra values are ignored. This multiplies the
    per-frame prefix weights after the 0/1 action mask, so masked dimensions
    remain disabled.
    """
    raw = os.environ.get("OPENPI_RT_GUIDANCE_DIM_WEIGHTS", "").strip()
    if not raw:
        return jnp.ones((action_dim,), dtype=jnp.float32)
    try:
        vals = [float(x.strip()) for x in raw.split(",") if x.strip()]
    except ValueError:
        logger.warning(
            "[pi0_realtime] invalid OPENPI_RT_GUIDANCE_DIM_WEIGHTS=%r; "
            "using all-ones",
            raw,
        )
        return jnp.ones((action_dim,), dtype=jnp.float32)
    if len(vals) < action_dim:
        vals.extend([1.0] * (action_dim - len(vals)))
    vals = vals[:action_dim]
    return jnp.asarray(vals, dtype=jnp.float32)


def _error_clip_from_env() -> Optional[float]:
    """Optional robust clipping threshold for RTC guidance residuals.

    ``OPENPI_RT_GUIDANCE_ERROR_CLIP`` clips ``prev_action_chunk - x_clean``
    before applying frame/dim weights. This prevents one large joint outlier
    from dominating the VJP correction while preserving ordinary small-error
    guidance. Empty, zero, or negative values disable clipping.
    """
    raw = os.environ.get("OPENPI_RT_GUIDANCE_ERROR_CLIP", "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        logger.warning(
            "[pi0_realtime] invalid OPENPI_RT_GUIDANCE_ERROR_CLIP=%r; "
            "disabling error clipping",
            raw,
        )
        return None
    if value <= 0:
        return None
    return value


def _tt_rtc_mask_gripper_from_env() -> bool:
    return os.environ.get("OPENPI_TT_RTC_MASK_GRIPPER", "").lower() in (
        "1", "true", "yes", "on",
    )


def _tt_rtc_arm_merge_mask(action_dim: int) -> jax.Array:
    """Dims to keep from the TT-RTC prefixed sample when grippers are masked.

    Gripper indices come from ``OPENPI_GRIPPER_INDICES`` so the same realtime
    implementation supports both 14-D ARX/Aloha and 16-D Nero layouts.
    Any model-internal dims beyond the robot action remain on the prefixed path.
    """
    mask = jnp.ones((action_dim,), dtype=jnp.bool_)
    raw_indices = os.environ.get("OPENPI_GRIPPER_INDICES", "6,13")
    try:
        gripper_indices = [
            int(value.strip()) for value in raw_indices.split(",") if value.strip()
        ]
    except ValueError:
        logger.warning(
            "[pi0_realtime] invalid OPENPI_GRIPPER_INDICES=%r; using 6,13",
            raw_indices,
        )
        gripper_indices = [6, 13]
    for index in gripper_indices:
        if 0 <= index < action_dim:
            mask = mask.at[index].set(False)
    return mask


def get_prefix_weights(start: int, end: int, total: int,
                       schedule: PrefixAttentionSchedule) -> jax.Array:
    """Per-frame guidance weights over the action chunk.

    Direct port of ``real-time-chunking-kinetix/src/model.py``
    :func:`get_prefix_weights`. With ``start=2, end=6, total=10`` the
    output is::

        1  1  4/5 3/5 2/5 1/5 0  0  0  0
              ^               ^
            start            end

    ``start`` (inclusive) is where the chunk begins to be allowed to
    change. ``end`` (exclusive) is where the chunk stops paying any
    attention to the prefix. ``end`` takes precedence over ``start``;
    if ``end < start`` then ``start`` is pulled down to ``end``.
    """
    start = jnp.minimum(start, end)
    if schedule == "ones":
        w = jnp.ones(total)
    elif schedule == "zeros":
        w = (jnp.arange(total) < start).astype(jnp.float32)
    elif schedule == "linear" or schedule == "exp":
        w = jnp.clip((start - 1 - jnp.arange(total)) / (end - start + 1) + 1, 0, 1)
        if schedule == "exp":
            w = w * jnp.expm1(w) / (jnp.e - 1)
    else:
        raise ValueError(f"Invalid schedule: {schedule!r}")
    return jnp.where(jnp.arange(total) >= end, 0, w)


def realtime_sample_actions(
    self: _pi0.Pi0,
    rng: jax.Array,
    observation: _model.Observation,
    *,
    num_steps: int = 10,
    noise: Optional[jax.Array] = None,
    prev_action_chunk: Optional[jax.Array] = None,
    prefix_attention_start: int = 0,
    prefix_attention_horizon: Optional[int] = None,
    prefix_attention_schedule: str = "exp",
    max_guidance_weight: float = 5.0,
    guidance_action_dim: Optional[int] = None,
    guidance_action_mask: Optional[jax.Array] = None,
    rtc_sample_mode: Optional[str] = None,
) -> jax.Array:
    """Drop-in replacement for :meth:`Pi0.sample_actions` with optional guidance.

    All kwargs except ``num_steps`` and ``noise`` are realtime-specific
    and optional. When ``prev_action_chunk is None`` the function
    behaves identically to the upstream implementation (which lets us
    install this as a monkey-patch without breaking existing callers).

    Args:
        prev_action_chunk: Previous chunk to guide toward, shape
            ``(b, action_horizon, action_dim)``. Must match the model's
            chunk size; pad with zeros if your client only knows about a
            shorter prefix (zero rows are ignored thanks to ``weights``).
        prefix_attention_start: First frame allowed to deviate. The
            kinetix paper calls this ``inference_delay``; for offline /
            synchronous chunking just leave at 0.
        prefix_attention_horizon: Frame index past which guidance is
            switched off. Defaults to the full chunk size.
        prefix_attention_schedule: One of ``"exp" | "linear" | "ones" |
            "zeros"``. ``"exp"`` matches the paper's default.
        max_guidance_weight: Upper bound on ``c * inv_r2``. Paper uses
            5.0.
        guidance_action_dim: If provided, only the first this many action
            dims are used to compute the guidance gradient (mask the rest
            to zero). Lets you guide a 32-dim π0.5 model with a 14-dim
            client target.
        guidance_action_mask: Optional per-action-dim mask. Values > 0
            participate in the guidance loss; zeros are ignored. This is
            useful for excluding gripper dims while still guiding all arm
            joints.
    """
    if rtc_sample_mode is None:
        rtc_sample_mode = os.environ.get("OPENPI_RTC_SAMPLE_MODE", "guidance")
    rtc_sample_mode = rtc_sample_mode.lower()
    if rtc_sample_mode in ("off", "none", "disabled", "disable"):
        prev_action_chunk = None

    # Backward-compat fast path: behave exactly like the original.
    if prev_action_chunk is None:
        return _pi0_original_sample_actions(
            self, rng, observation, num_steps=num_steps, noise=noise,
        )

    if rtc_sample_mode in ("train_time", "training_time", "tt_rtc", "direct"):
        mask_gripper = _tt_rtc_mask_gripper_from_env()
        logger.info(
            "[pi0_tt_rtc] direct prefix conditioning prefix_len=%s mode=%s "
            "mask_gripper=%s",
            prefix_attention_start,
            rtc_sample_mode,
            mask_gripper,
        )
        prefixed = _pi0_original_sample_actions(
            self,
            rng,
            observation,
            num_steps=num_steps,
            noise=noise,
            action_prefix=prev_action_chunk,
            prefix_length=prefix_attention_start,
        )
        if not mask_gripper:
            return prefixed
        raw = _pi0_original_sample_actions(
            self,
            rng,
            observation,
            num_steps=num_steps,
            noise=noise,
        )
        dim_mask = _tt_rtc_arm_merge_mask(self.action_dim)
        return jnp.where(dim_mask[None, None, :], prefixed, raw)
    if rtc_sample_mode != "guidance":
        logger.warning(
            "[pi0_realtime] unknown OPENPI_RTC_SAMPLE_MODE=%r; falling back to guidance",
            rtc_sample_mode,
        )

    observation = _model.preprocess_observation(None, observation, train=False)
    dt = -1.0 / num_steps
    batch_size = observation.state.shape[0]
    if noise is None:
        noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

    # Pad client-side prev chunk along the action-dim axis if it's narrower
    # than the model's full action_dim (e.g. ce_eda sends 14 dims for a
    # 32-dim π0.5 model). The padded zeros are ignored thanks to the
    # `guidance_action_dim` mask further below.
    prev_dim = prev_action_chunk.shape[-1]
    if prev_dim < self.action_dim:
        pad_width = ((0, 0), (0, 0), (0, self.action_dim - prev_dim))
        prev_action_chunk = jnp.pad(prev_action_chunk, pad_width)
    elif prev_dim > self.action_dim:
        prev_action_chunk = prev_action_chunk[..., : self.action_dim]

    # Fill KV cache with one prefix forward pass.
    prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
    prefix_attn_mask_kv = make_attn_mask(prefix_mask, prefix_ar_mask)
    positions_pref = jnp.cumsum(prefix_mask, axis=1) - 1
    _, kv_cache = self.PaliGemma.llm(
        [prefix_tokens, None], mask=prefix_attn_mask_kv, positions=positions_pref,
    )

    horizon = self.action_horizon
    if prefix_attention_horizon is None:
        prefix_attention_horizon = horizon

    weights = get_prefix_weights(
        prefix_attention_start,
        prefix_attention_horizon,
        horizon,
        prefix_attention_schedule,
    )  # [horizon]

    if guidance_action_mask is not None:
        dim_mask = jnp.asarray(guidance_action_mask, dtype=jnp.float32)
        if dim_mask.ndim != 1:
            dim_mask = jnp.reshape(dim_mask, (-1,))
        mask_dim = dim_mask.shape[0]
        if mask_dim < self.action_dim:
            dim_mask = jnp.pad(dim_mask, (0, self.action_dim - mask_dim))
        elif mask_dim > self.action_dim:
            dim_mask = dim_mask[: self.action_dim]
        dim_mask = (dim_mask > 0).astype(jnp.float32)
    elif guidance_action_dim is not None and int(guidance_action_dim) < self.action_dim:
        dim_mask = (jnp.arange(self.action_dim) < int(guidance_action_dim)).astype(jnp.float32)
    else:
        dim_mask = jnp.ones(self.action_dim, dtype=jnp.float32)
    dim_weights = _dim_weights_from_env(self.action_dim)
    error_clip = _error_clip_from_env()
    eff_weights = weights[:, None] * dim_mask[None, :] * dim_weights[None, :]
    debug_enabled = os.environ.get("OPENPI_RT_GUIDANCE_DEBUG", "").lower() in (
        "1", "true", "yes", "on",
    )
    hard_project_locked = os.environ.get(
        "OPENPI_RT_HARD_PROJECT_LOCKED_ARM", "",
    ).lower() in ("1", "true", "yes", "on")
    try:
        debug_dim = int(os.environ.get("OPENPI_RT_GUIDANCE_DEBUG_DIM", "12"))
    except ValueError:
        debug_dim = 12
    debug_dim = max(0, min(debug_dim, self.action_dim - 1))
    locked_mask = (
        (weights[:, None] >= (1.0 - 1e-6)) &
        (dim_mask[None, :] > 0)
    )
    guided_mask = eff_weights > 0

    if debug_enabled:
        logger.info(
            "[pi0_rt_dbg] enabled start=%s horizon=%s schedule=%s "
            "max_guidance=%s guidance_action_dim=%s guidance_action_mask=%s "
            "hard_project_locked=%s debug_dim=%s dim_weights=%s error_clip=%s "
            "rtc_sample_mode=%s",
            prefix_attention_start,
            prefix_attention_horizon,
            prefix_attention_schedule,
            max_guidance_weight,
            guidance_action_dim,
            "provided" if guidance_action_mask is not None else "none",
            hard_project_locked,
            debug_dim,
            os.environ.get("OPENPI_RT_GUIDANCE_DIM_WEIGHTS", ""),
            error_clip,
            rtc_sample_mode,
        )
        jax.debug.print(
            "[pi0_rt_dbg] weights start={start} horizon={horizon} "
            "locked_frames={locked} guided_frames={guided} mask_dims={mask_dims} "
            "dim_weight_min={dw_min} dim_weight_max={dw_max} "
            "first={first} mid={mid} last_guided={last}",
            start=prefix_attention_start,
            horizon=prefix_attention_horizon,
            locked=jnp.sum(weights >= (1.0 - 1e-6)),
            guided=jnp.sum(weights > 0),
            mask_dims=jnp.sum(dim_mask),
            dw_min=jnp.min(dim_weights),
            dw_max=jnp.max(dim_weights),
            first=weights[0],
            mid=weights[jnp.minimum(prefix_attention_start, horizon - 1)],
            last=weights[jnp.minimum(prefix_attention_horizon - 1, horizon - 1)],
        )

    def compute_v_t(x_t: jax.Array, time: jax.Array) -> jax.Array:
        """Velocity at time ``t`` for state ``x_t`` (single batched forward)."""
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
            observation, x_t, jnp.broadcast_to(time, batch_size),
        )
        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_attn_mask = einops.repeat(
            prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1],
        )
        full_attn_mask = jnp.concatenate(
            [prefix_attn_mask, suffix_attn_mask], axis=-1,
        )
        positions = (
            jnp.sum(prefix_mask, axis=-1)[:, None]
            + jnp.cumsum(suffix_mask, axis=-1) - 1
        )
        (_, suffix_out), _ = self.PaliGemma.llm(
            [None, suffix_tokens],
            mask=full_attn_mask,
            positions=positions,
            kv_cache=kv_cache,
            adarms_cond=[None, adarms_cond],
        )
        return self.action_out_proj(suffix_out[:, -horizon:])

    def step(carry):
        x_t, time = carry

        def denoiser(x_t_inner):
            v_t = compute_v_t(x_t_inner, time)
            # In π0 convention t=0 is clean, so x_clean = x_t - v_t * t.
            x_clean = x_t_inner - v_t * time
            return x_clean, v_t

        x_clean, vjp_fun, v_t = jax.vjp(denoiser, x_t, has_aux=True)
        # error: pull predicted clean trajectory toward prev chunk on
        # frames where weights > 0; broadcast over batch dim. Optional clipping
        # makes the guidance objective robust to a single joint outlier.
        raw_error = prev_action_chunk - x_clean
        if error_clip is not None:
            clipped_error = jnp.clip(raw_error, -error_clip, error_clip)
        else:
            clipped_error = raw_error
        error = clipped_error * eff_weights[None, :, :]
        (pinv_correction,) = vjp_fun(error)
        # Paper constants in π0 convention:
        eps = 1e-8
        inv_r2 = (time * time + (1.0 - time) ** 2) / (time * time + eps)
        c = jnp.nan_to_num(time / (1.0 - time + eps), posinf=max_guidance_weight)
        guidance = jnp.minimum(c * inv_r2, max_guidance_weight)

        v_corrected = v_t - guidance * pinv_correction
        if debug_enabled:
            clean_delta = x_clean - prev_action_chunk
            prefix_clean_err = jnp.max(jnp.where(
                locked_mask[None, :, :], jnp.abs(clean_delta), 0.0,
            ))
            guided_clean_err = jnp.max(jnp.where(
                guided_mask[None, :, :], jnp.abs(clean_delta), 0.0,
            ))
            dim_prefix_err = jnp.max(jnp.where(
                locked_mask[:, debug_dim],
                jnp.abs(clean_delta[0, :, debug_dim]),
                0.0,
            ))
            prefix_corr_step = jnp.max(jnp.where(
                locked_mask[None, :, :],
                jnp.abs(dt * guidance * pinv_correction),
                0.0,
            ))
            prefix_model_step = jnp.max(jnp.where(
                locked_mask[None, :, :],
                jnp.abs(dt * v_t),
                0.0,
            ))
            pinv_norm = jnp.sqrt(jnp.sum(jnp.square(
                jnp.where(locked_mask[None, :, :], pinv_correction, 0.0)
            )))
            error_norm = jnp.sqrt(jnp.sum(jnp.square(error)))
            if error_clip is not None:
                clipped_count = jnp.sum(
                    (jnp.abs(raw_error) > error_clip) &
                    (guided_mask[None, :, :] > 0),
                )
                clipped_max = jnp.max(jnp.where(
                    guided_mask[None, :, :],
                    jnp.maximum(jnp.abs(raw_error) - error_clip, 0.0),
                    0.0,
                ))
            else:
                clipped_count = jnp.asarray(0)
                clipped_max = jnp.asarray(0.0)
            jax.debug.print(
                "[pi0_rt_dbg] step t={t} guidance={g} "
                "prefix_clean_err_max={pe} guided_clean_err_max={ge} "
                "dim{d}_prefix_err_max={de} "
                "prefix_corr_step_max={cs} prefix_model_step_max={ms} "
                "pinv_norm={pn} error_norm={en} "
                "clipped_count={cc} clipped_excess_max={cm}",
                t=time,
                g=guidance,
                pe=prefix_clean_err,
                ge=guided_clean_err,
                d=debug_dim,
                de=dim_prefix_err,
                cs=prefix_corr_step,
                ms=prefix_model_step,
                pn=pinv_norm,
                en=error_norm,
                cc=clipped_count,
                cm=clipped_max,
            )
        return x_t + dt * v_corrected, time + dt

    def cond(carry):
        _, time = carry
        return time >= -dt / 2

    x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
    if hard_project_locked:
        x_0 = jnp.where(locked_mask[None, :, :], prev_action_chunk, x_0)
    if debug_enabled:
        final_delta = x_0 - prev_action_chunk
        final_prefix_err = jnp.max(jnp.where(
            locked_mask[None, :, :], jnp.abs(final_delta), 0.0,
        ))
        final_guided_err = jnp.max(jnp.where(
            guided_mask[None, :, :], jnp.abs(final_delta), 0.0,
        ))
        final_dim_prefix_err = jnp.max(jnp.where(
            locked_mask[:, debug_dim],
            jnp.abs(final_delta[0, :, debug_dim]),
            0.0,
        ))
        jax.debug.print(
            "[pi0_rt_dbg] final prefix_err_max={pe} "
            "guided_err_max={ge} dim{d}_prefix_err_max={de}",
            pe=final_prefix_err,
            ge=final_guided_err,
            d=debug_dim,
            de=final_dim_prefix_err,
        )
    return x_0


# Will be populated by :func:`install_realtime_patch`.
_pi0_original_sample_actions = None


def install_realtime_patch() -> None:
    """Replace :meth:`openpi.models.pi0.Pi0.sample_actions` in-place.

    Idempotent: calling twice is a no-op. The replacement falls back to
    the original behavior whenever ``prev_action_chunk`` is not passed
    in, so it's safe to install unconditionally at server startup.
    """
    global _pi0_original_sample_actions
    if getattr(_pi0.Pi0.sample_actions, "_realtime_patched", False):
        return
    _pi0_original_sample_actions = _pi0.Pi0.sample_actions

    def patched(self, rng, observation, *, num_steps=10, noise=None, **kwargs):
        return realtime_sample_actions(
            self, rng, observation,
            num_steps=num_steps, noise=noise, **kwargs,
        )

    patched._realtime_patched = True  # type: ignore[attr-defined]
    _pi0.Pi0.sample_actions = patched  # type: ignore[assignment]
    logger.info("[pi0_realtime] installed Pi0.sample_actions patch (realtime chunking available)")
