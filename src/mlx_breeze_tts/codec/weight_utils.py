"""Checkpoint-layout helpers for the vendored Qwen3 speech codec."""

from __future__ import annotations

from typing import Protocol


class _Shaped(Protocol):
    shape: tuple[int, ...]


def conv1d_weight_is_mlx_layout(array: _Shaped) -> bool:
    """Return whether a 3-D Conv1d weight uses MLX's (out, kernel, in) layout.

    Qwen3 tokenizer checkpoints can contain either PyTorch ``(out, in, kernel)``
    or already-converted MLX weights. This is the upstream Qwen3 codec heuristic,
    kept local so the Breeze runtime does not depend on ``mlx_audio``.
    """

    shape = array.shape
    if len(shape) != 3:
        return False

    _, dim2, dim3 = shape
    if dim2 == 1:
        return dim3 > 64
    if dim3 == 1:
        return dim2 <= 64
    return dim2 < dim3
