"""Vendored Qwen3 speech codec required by Breeze TTS 2."""

__all__ = ["Qwen3TTSSpeechTokenizer"]


def __getattr__(name: str):
    if name != "Qwen3TTSSpeechTokenizer":
        raise AttributeError(name)
    from .speech_tokenizer import Qwen3TTSSpeechTokenizer

    globals()[name] = Qwen3TTSSpeechTokenizer
    return Qwen3TTSSpeechTokenizer
