import json
import subprocess
from pathlib import Path

import pytest

from mlx_breeze_tts.objective import (
    apply_objective_metrics,
    character_error_rate,
    normalize_text,
    objective_template,
    reference_leakage_similarity,
)
from mlx_breeze_tts.objective_runner import evaluate_objective_metrics


def test_text_metrics_strip_events_and_match_skill_leakage_semantics():
    assert normalize_text("(laugh) Hello, 世界!", strip_events=True) == "hello世界"
    assert character_error_rate("[笑] 欢迎回来。", "欢迎回来") == 0.0
    assert character_error_rate("hello", "hallo") == pytest.approx(0.2)
    assert reference_leakage_similarity("Exact prompt", "exact prompt") == 1.0
    assert reference_leakage_similarity("Exact prompt", "different target") < 0.65


def _summary(path):
    samples = [
        {
            "capability": "voice_design_en",
            "text": "Hello world.",
            "expected_text": "Hello world.",
            "audio": "design.wav",
            "status": "audio_generated",
        },
        {
            "capability": "voice_clone_en",
            "text": "Target sentence.",
            "expected_text": "Target sentence.",
            "audio": "clone.wav",
            "status": "audio_generated",
            "ref_audio": "ref.wav",
            "ref_text": "Unrelated reference prompt.",
        },
    ]
    path.write_text(json.dumps({"samples": samples, "validation": {}}))


def test_template_resolves_audio_and_objective_merge_aggregates(tmp_path):
    summary = tmp_path / "summary.json"
    _summary(summary)
    template_path = objective_template(summary, tmp_path / "template.json")
    template = json.loads(template_path.read_text())
    assert template["rows"][0]["audio"] == str((tmp_path / "design.wav").resolve())
    for row in template["rows"]:
        row["asr_text"] = (
            "Hello world"
            if row["capability"] == "voice_design_en"
            else "Target sentence"
        )
        if row["capability"] == "voice_clone_en":
            row["speaker_cosine"] = 0.8
            row["speaker_p10"] = 0.7
            row["speaker_backend"] = "ecapa"
    template_path.write_text(json.dumps(template))
    output = apply_objective_metrics(
        summary, template_path, tmp_path / "summary.metrics.json"
    )
    merged = json.loads(output.read_text())
    validation = merged["validation"]
    assert validation["max_cer"] == 0.0
    assert validation["corpus_cer"] == 0.0
    assert validation["corpus_cer_edits"] == 0
    assert validation["clone_cosine_min"] == 0.8
    assert validation["clone_p10_min"] == 0.7
    assert validation["leakage_max"] < 0.65
    assert validation["objective_metrics"] == "complete"


def test_cer_gate_excludes_events_and_exploratory_cross_language(tmp_path):
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "capability": "voice_design_en",
                        "expected_text": "Standard text.",
                        "status": "audio_generated",
                    },
                    {
                        "capability": "event_en_laugh",
                        "expected_text": "(laugh) Event text.",
                        "status": "audio_generated",
                    },
                    {
                        "capability": "cross_clone_zh_to_en",
                        "expected_text": "Exploratory text.",
                        "status": "audio_generated",
                    },
                ],
                "validation": {},
            }
        )
    )
    metrics = tmp_path / "metrics.json"
    metrics.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "rows": [
                    {"capability": "voice_design_en", "asr_text": "Standard text"},
                    {"capability": "event_en_laugh", "asr_text": "wrong"},
                    {"capability": "cross_clone_zh_to_en", "asr_text": "wrong"},
                ],
            }
        )
    )

    output = apply_objective_metrics(summary, metrics, tmp_path / "output.json")
    validation = json.loads(output.read_text())["validation"]
    assert validation["max_cer"] == 0.0
    assert validation["corpus_cer"] == 0.0
    assert (
        validation["cer_scope"]
        == "standard_content_excluding_events_and_cross_language"
    )


def test_missing_or_invalid_metrics_fail_closed(tmp_path):
    summary = tmp_path / "summary.json"
    _summary(summary)
    metrics = tmp_path / "metrics.json"
    metrics.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "rows": [
                    {"capability": "voice_design_en", "asr_text": "Hello world"},
                    {
                        "capability": "voice_clone_en",
                        "asr_text": "Target sentence",
                        "speaker_cosine": 1.1,
                    },
                ],
            }
        )
    )
    with pytest.raises(ValueError, match="between 0 and 1"):
        apply_objective_metrics(summary, metrics, tmp_path / "output.json")
    document = json.loads(metrics.read_text())
    document["rows"][1]["speaker_cosine"] = None
    metrics.write_text(json.dumps(document))
    output = apply_objective_metrics(summary, metrics, tmp_path / "output.json")
    validation = json.loads(output.read_text())["validation"]
    assert validation["objective_metrics"] == "partial"
    assert validation["clone_cosine_min"] is None


def test_evaluator_fills_metrics_and_records_provenance(tmp_path):
    audio = tmp_path / "clone.wav"
    reference = tmp_path / "reference.wav"
    audio.write_bytes(b"generated-audio")
    reference.write_bytes(b"reference-audio")
    model = tmp_path / "whisper-snapshot"
    model.mkdir()
    (model / "weights.safetensors").write_bytes(b"whisper-weights")
    executable = tmp_path / "mlx_whisper"
    executable.write_text("#!/bin/sh\n")
    executable.chmod(0o755)
    python = tmp_path / "python"
    python.write_text("#!/bin/sh\n")
    python.chmod(0o755)
    metrics = tmp_path / "metrics.json"
    metrics.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "rows": [
                    {
                        "capability": "voice_clone_en",
                        "audio": str(audio),
                        "reference_audio": str(reference),
                        "expected_text": "Target sentence.",
                    }
                ],
            }
        )
    )

    def fake_run(command, **_kwargs):
        if "--output-dir" in command:
            output = command[command.index("--output-dir") + 1]
            Path(output, "result.json").write_text(
                json.dumps({"text": "Target sentence"})
            )
            return subprocess.CompletedProcess(command, 0, "", "")
        payload = {
            "cosine": 0.8,
            "p10_cosine": 0.7,
            "backend": "speechbrain_ecapa_voxceleb",
            "model_id": "speechbrain/spkrec-ecapa-voxceleb",
            "model_artifact_sha256": "a" * 64,
            "device": "cpu",
            "generated_segments": 2,
            "reference_segments": 2,
            "reference_consistency": 0.9,
        }
        return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")

    output, complete = evaluate_objective_metrics(
        metrics,
        tmp_path / "evaluated.json",
        whisper_executable=executable,
        whisper_model=model,
        speaker_python=python,
        run=fake_run,
    )
    document = json.loads(output.read_text())
    row = document["rows"][0]
    assert complete
    assert row["asr_text"] == "Target sentence"
    assert row["speaker_cosine"] == 0.8
    assert len(row["audio_sha256"]) == 64
    assert document["evaluation"]["status"] == "complete"
    assert document["evaluation"]["asr_model_revision"] == "whisper-snapshot"


def test_evaluator_retains_failures_and_fails_closed(tmp_path):
    audio = tmp_path / "design.wav"
    audio.write_bytes(b"audio")
    model = tmp_path / "snapshot"
    model.mkdir()
    (model / "weights.safetensors").write_bytes(b"weights")
    executable = tmp_path / "mlx_whisper"
    executable.write_text("#!/bin/sh\n")
    executable.chmod(0o755)
    python = tmp_path / "python"
    python.write_text("#!/bin/sh\n")
    python.chmod(0o755)
    metrics = tmp_path / "metrics.json"
    metrics.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "rows": [{"capability": "voice_design_en", "audio": str(audio)}],
            }
        )
    )

    def failing_run(command, **_kwargs):
        return subprocess.CompletedProcess(command, 2, "", "decoder failed")

    output, complete = evaluate_objective_metrics(
        metrics,
        tmp_path / "evaluated.json",
        whisper_executable=executable,
        whisper_model=model,
        speaker_python=python,
        run=failing_run,
    )
    document = json.loads(output.read_text())
    assert not complete
    assert document["evaluation"]["status"] == "partial"
    assert document["evaluation"]["failures"][0]["metric"] == "asr"
