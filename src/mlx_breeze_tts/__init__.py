"""Standalone MLX implementation of Breeze TTS 2."""

from .audio import load_audio, write_audio
from .config import ModelConfig
from .loader import DEFAULT_MODEL, load
from .model import Model
from .results import GenerationResult

__all__ = [
    "DEFAULT_MODEL",
    "GenerationResult",
    "Model",
    "ModelConfig",
    "load",
    "load_audio",
    "write_audio",
]

__version__ = "0.1.0"
