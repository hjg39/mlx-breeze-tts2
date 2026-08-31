import json
from pathlib import Path

import numpy as np

from mlx_breeze_tts import benchmark


class _FakeMX:
    float32 = np.float32

    @staticmethod
    def concatenate(arrays):
        return np.concatenate(arrays)

    @staticmethod
    def zeros(shape, dtype):
        return np.zeros(shape, dtype=dtype)

    @staticmethod
    def array(value):
        return np.asarray(value)

    @staticmethod
    def get_peak_memory():
        return 2_000_000_000


class _Chunk:
    def __init__(self, value=0.1):
        self.audio = np.full((12,), value, dtype=np.float32)
        self.time_to_first_audio_seconds = 0.01


class _Model:
    sample_rate = 12

    def generate(self, **kwargs):
        value = 0.2 if kwargs.get("temperature") == 0.7 else 0.1
        yield _Chunk(value)
        if kwargs.get("stream"):
            yield _Chunk(value)


def test_http_probe_exercises_real_fastapi_contract():
    result = benchmark._probe_http(_Model(), 42)
    assert result["status"] == "pass"
    assert result["speech_status_code"] == 200
    assert all(result["checks"].values())


def test_waveform_diagnostics_detects_repeated_tail_and_stream_jump():
    clean = np.linspace(-0.2, 0.2, 120, dtype=np.float32)
    clean_report = benchmark._waveform_diagnostics(clean, 120, [60, 60])
    assert clean_report["nonfinite_count"] == 0
    assert clean_report["repeated_tail"] is False
    assert clean_report["stream_discontinuity"] is False

    repeated = np.tile(np.linspace(-0.1, 0.1, 60, dtype=np.float32), 2)
    repeated_report = benchmark._waveform_diagnostics(repeated, 120)
    assert repeated_report["repeated_tail"] is True

    jumped = np.concatenate([np.full(60, -0.4), np.full(60, 0.4)])
    jump_report = benchmark._waveform_diagnostics(jumped, 120, [60, 60])
    assert jump_report["stream_discontinuity"] is True


def test_full_benchmark_emits_complete_fail_closed_bundle(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark, "_mlx_core", lambda: _FakeMX)
    monkeypatch.setattr(benchmark, "_resolve_model_path", lambda _: tmp_path / "model")
    monkeypatch.setattr(benchmark, "_load_model", lambda _: _Model())
    monkeypatch.setattr(
        benchmark, "_probe_http", lambda model, seed: {"status": "pass"}
    )
    monkeypatch.setattr(
        benchmark,
        "_write_audio",
        lambda path, audio, sample_rate: Path(path).write_bytes(b"RIFF"),
    )
    summary_path = benchmark.run_benchmark(
        "fake-model",
        tmp_path / "report",
        ref_audio_en="en.wav",
        ref_text_en="Exact English transcript.",
        ref_audio_zh="zh.wav",
        ref_text_zh="精确的中文转写。",
        invoked_via_cli=True,
    )
    report = json.loads(summary_path.read_text())
    assert report["matrix_coverage"]["pass"] is True
    assert len(report["samples"]) == 23
    assert all(sample["samples"] > 0 for sample in report["samples"])
    assert report["interfaces"]["python"] == "pass"
    assert report["interfaces"]["streaming"] == "pass"
    assert report["interfaces"]["cli"] == "pass"
    assert report["interfaces"]["http"] == "pass"
    assert report["validation"]["seed_reproducibility"] == "pass"
    assert report["validation"]["sampling_path"] == "pass"
    assert report["validation"]["waveform_integrity"] == "pass"
    assert report["validation"]["pytorch_parity"] == "pending"
    assert (tmp_path / "report/report.md").is_file()
    assert (tmp_path / "report/index.html").is_file()
    assert (tmp_path / "report/objective_metrics.json").is_file()


def test_benchmark_records_missing_reference_cases(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark, "_mlx_core", lambda: _FakeMX)
    monkeypatch.setattr(benchmark, "_resolve_model_path", lambda _: tmp_path / "model")
    monkeypatch.setattr(benchmark, "_load_model", lambda _: _Model())
    monkeypatch.setattr(
        benchmark, "_probe_http", lambda model, seed: {"status": "pass"}
    )
    monkeypatch.setattr(
        benchmark,
        "_write_audio",
        lambda path, audio, sample_rate: Path(path).write_bytes(b"RIFF"),
    )
    summary_path = benchmark.run_benchmark("fake", tmp_path / "report")
    report = json.loads(summary_path.read_text())
    missing = [
        sample for sample in report["samples"] if sample["status"] == "missing_input"
    ]
    assert {sample["capability"] for sample in missing} == {
        "voice_clone_en",
        "voice_clone_zh",
        "cross_clone_en_to_zh",
        "cross_clone_zh_to_en",
        "voice_direction_en",
        "voice_direction_zh",
    }
