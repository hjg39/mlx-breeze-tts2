"""Offline objective-metric runner with auditable ASR and ECAPA provenance."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


WHISPER_MODEL_ID = "mlx-community/whisper-large-v3-turbo"
SPEAKER_MODEL_ID = "speechbrain/spkrec-ecapa-voxceleb"
_WHISPER_EXECUTABLES = (
    Path("/Users/vanch/.codex/envs/bluestone-mlx-audio/bin/mlx_whisper"),
    Path("/Users/vanch/pinokio/bin/miniconda/envs/videolingo/bin/mlx_whisper"),
    Path("/Users/vanch/.local/pipx/venvs/whisperlivekit/bin/mlx_whisper"),
    Path("/Users/vanch/Library/Python/3.9/bin/mlx_whisper"),
)
_SPEAKER_PYTHONS = (
    Path("/Users/vanch/.codex/envs/bluestone-mlx-audio/bin/python"),
    Path("/Users/vanch/tts-test-project/.venv/bin/python"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cached_snapshot(model_id: str) -> Path | None:
    root = (
        Path.home()
        / ".cache/huggingface/hub"
        / f"models--{model_id.replace('/', '--')}"
        / "snapshots"
    )
    candidates = [path for path in root.glob("*") if path.is_dir()]
    candidates.sort(key=lambda path: path.stat().st_mtime_ns, reverse=True)
    for candidate in candidates:
        if any(candidate.glob("*.safetensors")):
            return candidate.resolve()
    return None


def _valid_executable(candidates: tuple[Path, ...]) -> Path | None:
    for candidate in candidates:
        if not os.access(candidate, os.X_OK):
            continue
        try:
            first_line = candidate.read_text(errors="ignore").splitlines()[0]
        except (IndexError, OSError, UnicodeDecodeError):
            return candidate
        if first_line.startswith("#!"):
            interpreter = Path(first_line[2:].split(" ", 1)[0])
            if not interpreter.exists():
                continue
        return candidate
    return None


def _language(row: dict[str, Any]) -> str:
    capability = str(row.get("capability") or "").lower()
    if capability.endswith("_zh"):
        return "zh"
    if capability.endswith("_en"):
        return "en"
    text = str(row.get("expected_text") or "")
    return (
        "zh" if any("\u4e00" <= character <= "\u9fff" for character in text) else "en"
    )


def _parse_asr_output(directory: Path, stdout: str) -> str:
    for path in sorted(directory.glob("*.json")):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            text = str(payload.get("text") or "").strip()
            if text:
                return text
    text = stdout.strip()
    if text and any(character.isalnum() for character in text):
        return text
    raise RuntimeError("ASR produced no linguistic text")


def _run_asr(
    row: dict[str, Any],
    *,
    executable: Path,
    model: Path,
    timeout: int,
    run: Callable[..., subprocess.CompletedProcess[str]],
) -> dict[str, Any]:
    audio = Path(str(row["audio"])).expanduser().resolve()
    with tempfile.TemporaryDirectory(prefix="mlx-breeze-asr-") as temporary:
        output = Path(temporary)
        command = [
            str(executable),
            str(audio),
            "--model",
            str(model),
            "--output-dir",
            str(output),
            "--output-format",
            "json",
            "--language",
            _language(row),
        ]
        result = run(
            command, text=True, capture_output=True, timeout=timeout, check=False
        )
        if result.returncode:
            detail = (result.stderr or result.stdout or "").strip()[:1000]
            raise RuntimeError(f"ASR exited {result.returncode}: {detail}")
        text = _parse_asr_output(output, result.stdout or "")
    weights = next(iter(sorted(model.glob("*.safetensors"))), None)
    return {
        "asr_text": text,
        "asr_backend": "mlx_whisper",
        "asr_model_id": WHISPER_MODEL_ID,
        "asr_model_revision": model.name,
        "asr_model_artifact_sha256": _sha256(weights) if weights else None,
        "asr_executable": str(executable),
    }


def _parse_worker_output(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise RuntimeError("ECAPA worker returned no JSON result")


def _run_speaker(
    row: dict[str, Any],
    *,
    python: Path,
    timeout: int,
    device: str,
    run: Callable[..., subprocess.CompletedProcess[str]],
) -> dict[str, Any]:
    worker = Path(__file__).with_name("speaker_worker.py")
    command = [
        str(python),
        str(worker),
        "--generated",
        str(Path(str(row["audio"])).expanduser().resolve()),
        "--reference",
        str(Path(str(row["reference_audio"])).expanduser().resolve()),
        "--device",
        device,
    ]
    result = run(command, text=True, capture_output=True, timeout=timeout, check=False)
    if result.returncode:
        detail = (result.stderr or result.stdout or "").strip()[:1000]
        raise RuntimeError(f"ECAPA exited {result.returncode}: {detail}")
    payload = _parse_worker_output(result.stdout or "")
    return {
        "speaker_cosine": payload["cosine"],
        "speaker_p10": payload["p10_cosine"],
        "speaker_backend": payload["backend"],
        "speaker_model_id": payload["model_id"],
        "speaker_model_artifact_sha256": payload["model_artifact_sha256"],
        "speaker_device": payload["device"],
        "speaker_segments": payload["generated_segments"],
        "speaker_reference_segments": payload["reference_segments"],
        "speaker_reference_consistency": payload["reference_consistency"],
        "speaker_python": str(python),
    }


def evaluate_objective_metrics(
    metrics_path: str | Path,
    output_path: str | Path,
    *,
    whisper_executable: str | Path | None = None,
    whisper_model: str | Path | None = None,
    speaker_python: str | Path | None = None,
    speaker_device: str = "auto",
    timeout: int = 600,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> tuple[Path, bool]:
    """Fill a schema-v1 objective worksheet and return `(path, complete)`."""

    source = Path(metrics_path).expanduser().resolve()
    destination = Path(output_path).expanduser().resolve()
    if source == destination:
        raise ValueError("Output must differ from the source metrics worksheet")
    document = json.loads(source.read_text())
    rows = document.get("rows")
    if document.get("schema_version") != 1 or not isinstance(rows, list):
        raise ValueError("Metrics document must use schema_version 1 and rows[]")

    executable = (
        Path(whisper_executable).expanduser().resolve()
        if whisper_executable
        else _valid_executable(_WHISPER_EXECUTABLES)
    )
    model = (
        Path(whisper_model).expanduser().resolve()
        if whisper_model
        else _cached_snapshot(WHISPER_MODEL_ID)
    )
    python = (
        Path(speaker_python).expanduser().resolve()
        if speaker_python
        else _valid_executable(_SPEAKER_PYTHONS)
    )
    if executable is None:
        raise RuntimeError("No usable mlx_whisper executable was found")
    if model is None or not model.is_dir():
        raise RuntimeError(f"No cached weights found for {WHISPER_MODEL_ID}")
    if python is None:
        raise RuntimeError(
            "No Python environment with the ECAPA dependencies was found"
        )
    if speaker_device not in {"auto", "cpu", "mps"}:
        raise ValueError("speaker_device must be auto, cpu, or mps")

    failures: list[dict[str, str]] = []
    for row in rows:
        capability = str(row.get("capability") or "unknown")
        audio = Path(str(row.get("audio") or "")).expanduser().resolve()
        if not audio.is_file():
            failures.append(
                {"capability": capability, "metric": "audio", "error": "missing audio"}
            )
            continue
        row["audio_sha256"] = _sha256(audio)
        try:
            row.update(
                _run_asr(
                    row,
                    executable=executable,
                    model=model,
                    timeout=timeout,
                    run=run,
                )
            )
            row["asr_error"] = None
        except Exception as exc:
            row["asr_error"] = str(exc)
            failures.append(
                {"capability": capability, "metric": "asr", "error": str(exc)}
            )

        reference = str(row.get("reference_audio") or "").strip()
        if not reference:
            continue
        reference_path = Path(reference).expanduser().resolve()
        if not reference_path.is_file():
            error = "missing reference audio"
            row["speaker_error"] = error
            failures.append(
                {"capability": capability, "metric": "speaker", "error": error}
            )
            continue
        row["reference_audio_sha256"] = _sha256(reference_path)
        try:
            row.update(
                _run_speaker(
                    row,
                    python=python,
                    timeout=timeout,
                    device=speaker_device,
                    run=run,
                )
            )
            row["speaker_error"] = None
        except Exception as exc:
            row["speaker_error"] = str(exc)
            failures.append(
                {"capability": capability, "metric": "speaker", "error": str(exc)}
            )

    document["evaluation"] = {
        "status": "complete" if not failures else "partial",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "source_metrics_sha256": _sha256(source),
        "asr_backend": "mlx_whisper",
        "asr_model_id": WHISPER_MODEL_ID,
        "asr_model_revision": model.name,
        "speaker_backend": "speechbrain_ecapa_voxceleb",
        "speaker_model_id": SPEAKER_MODEL_ID,
        "speaker_device_requested": speaker_device,
        "failures": failures,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(document, ensure_ascii=False, indent=2))
    return destination, not failures
