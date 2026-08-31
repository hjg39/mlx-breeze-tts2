import wave

import numpy as np
import pytest
import soundfile as sf

from mlx_breeze_tts.audio import AudioInputError, load_audio


def test_load_audio_rejects_undecodable_input(tmp_path):
    source = tmp_path / "broken.wav"
    source.write_bytes(b"not a wave file")
    with pytest.raises(AudioInputError, match="could not be decoded"):
        load_audio(source)


def test_load_audio_rejects_empty_wave(tmp_path):
    source = tmp_path / "empty.wav"
    with wave.open(str(source), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(24_000)
        output.writeframes(b"")
    with pytest.raises(AudioInputError, match="must not be empty"):
        load_audio(source)


def test_load_audio_rejects_nonfinite_float_wave(tmp_path):
    source = tmp_path / "nonfinite.wav"
    sf.write(source, np.array([np.nan], dtype=np.float32), 24_000, subtype="FLOAT")
    with pytest.raises(AudioInputError, match="finite"):
        load_audio(source)
