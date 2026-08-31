"""Small, standalone audio I/O helpers."""

from pathlib import Path

import mlx.core as mx
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def load_audio(path: str | Path, sample_rate: int = 24_000) -> mx.array:
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"Audio file not found: {source}")
    samples, source_rate = sf.read(source, dtype="float32", always_2d=True)
    samples = samples.mean(axis=1)
    if source_rate != sample_rate:
        divisor = np.gcd(source_rate, sample_rate)
        samples = resample_poly(
            samples, sample_rate // divisor, source_rate // divisor
        ).astype(np.float32)
    return mx.array(samples, dtype=mx.float32)


def write_audio(path: str | Path, audio: mx.array, sample_rate: int = 24_000) -> Path:
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    sf.write(destination, np.asarray(audio, dtype=np.float32), sample_rate)
    return destination


def write_audio_chunks(path: str | Path, chunks, sample_rate: int = 24_000) -> Path:
    """Write generated chunks incrementally as a mono PCM16 WAV file."""
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with sf.SoundFile(
        destination,
        mode="w",
        samplerate=sample_rate,
        channels=1,
        subtype="PCM_16",
    ) as output:
        for chunk in chunks:
            output.write(np.asarray(chunk.audio, dtype=np.float32))
    return destination
