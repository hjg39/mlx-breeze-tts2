"""Metal-independent validation for public generation arguments."""

from __future__ import annotations

import math
from numbers import Integral, Real


def _finite_real(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number.")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number.")
    return number


def _integer(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer.")
    return int(value)


def validate_generation_args(
    *,
    text,
    voice,
    instruct,
    cfg_scale,
    max_tokens,
    temperature,
    top_p,
    top_k,
    repetition_penalty,
    seed,
    stream,
    streaming_interval,
    fast_depth,
) -> None:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string.")
    if voice is not None and not isinstance(voice, str):
        raise ValueError("voice must be a string or None.")
    if instruct is not None and not isinstance(instruct, str):
        raise ValueError("instruct must be a string or None.")
    if seed is not None:
        seed = _integer(seed, "seed")
        if seed < 0:
            raise ValueError("seed must be non-negative or None.")
    max_tokens = _integer(max_tokens, "max_tokens")
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive.")
    top_k = _integer(top_k, "top_k")
    if top_k < 0:
        raise ValueError("top_k must be non-negative.")
    temperature = _finite_real(temperature, "temperature")
    if temperature < 0:
        raise ValueError("temperature must be non-negative.")
    top_p = _finite_real(top_p, "top_p")
    if not 0 <= top_p <= 1:
        raise ValueError("top_p must be between 0 and 1.")
    repetition_penalty = _finite_real(repetition_penalty, "repetition_penalty")
    if repetition_penalty <= 0:
        raise ValueError("repetition_penalty must be positive.")
    if cfg_scale is not None:
        cfg_scale = _finite_real(cfg_scale, "cfg_scale")
        if cfg_scale <= 0:
            raise ValueError("cfg_scale must be positive.")
    if not isinstance(stream, bool):
        raise ValueError("stream must be a boolean.")
    if not isinstance(fast_depth, bool):
        raise ValueError("fast_depth must be a boolean.")
    streaming_interval = _finite_real(streaming_interval, "streaming_interval")
    if streaming_interval <= 0:
        raise ValueError("streaming_interval must be positive.")
