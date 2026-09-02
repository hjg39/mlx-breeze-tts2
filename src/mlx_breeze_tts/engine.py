"""Stable long-lived engine API above the model generator."""

from pathlib import Path

from .loader import DEFAULT_MODEL, load
from .model import Model


class BreezeEngine:
    def __init__(self, model: Model):
        self.model = model

    @classmethod
    def from_pretrained(
        cls,
        model: str | Path = DEFAULT_MODEL,
        *,
        revision: str | None = None,
        lazy: bool = False,
    ) -> "BreezeEngine":
        return cls(load(model, revision=revision, lazy=lazy))

    @property
    def sample_rate(self) -> int:
        return self.model.sample_rate

    def generate(
        self,
        text: str,
        *,
        instruction: str | None = None,
        ref_audio=None,
        ref_text: str | None = None,
        speaker: str = "S0",
        seed: int | None = 42,
        cfg_scale: float = 1.0,
        temperature: float = 0.9,
        top_p: float = 1.0,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        max_new_tokens: int = 1500,
        stream: bool = False,
        streaming_interval: float = 2.0,
        fast_depth: bool = False,
    ):
        return self.model.generate(
            text=text,
            voice=speaker,
            instruct=instruction,
            ref_audio=ref_audio,
            ref_text=ref_text,
            seed=seed,
            cfg_scale=cfg_scale,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            max_tokens=max_new_tokens,
            stream=stream,
            streaming_interval=streaming_interval,
            fast_depth=fast_depth,
        )


def load_model(
    model: str | Path = DEFAULT_MODEL,
    *,
    revision: str | None = None,
    lazy: bool = False,
) -> BreezeEngine:
    return BreezeEngine.from_pretrained(model, revision=revision, lazy=lazy)
