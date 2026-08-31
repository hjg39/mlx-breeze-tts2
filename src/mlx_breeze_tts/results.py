"""Public result and configuration primitives."""

import inspect
from dataclasses import dataclass

import mlx.core as mx


@dataclass
class BaseModelArgs:
    @classmethod
    def from_dict(cls, params):
        return cls(
            **{
                key: value
                for key, value in params.items()
                if key in inspect.signature(cls).parameters
            }
        )


@dataclass
class GenerationResult:
    audio: mx.array
    samples: int
    sample_rate: int
    segment_idx: int
    token_count: int
    audio_duration: str
    real_time_factor: float
    prompt: dict
    audio_samples: dict
    processing_time_seconds: float
    peak_memory_usage: float
    is_streaming_chunk: bool = False
    is_final_chunk: bool = False
