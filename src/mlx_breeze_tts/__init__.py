"""Standalone MLX implementation of Breeze TTS 2.

Public objects are imported lazily so package metadata, CLI help, and HTTP
contract tests do not initialize Metal before inference is requested.
"""

from importlib import import_module

__version__ = "0.1.0"

_EXPORTS = {
    "BreezeEngine": (".engine", "BreezeEngine"),
    "DEFAULT_MODEL": (".loader", "DEFAULT_MODEL"),
    "GenerationResult": (".results", "GenerationResult"),
    "Model": (".model", "Model"),
    "ModelConfig": (".config", "ModelConfig"),
    "load": (".loader", "load"),
    "load_audio": (".audio", "load_audio"),
    "load_model": (".engine", "load_model"),
    "write_audio": (".audio", "write_audio"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, object_name = _EXPORTS[name]
    value = getattr(import_module(module_name, __name__), object_name)
    globals()[name] = value
    return value
