"""Validation for pinned local acceptance-matrix reference inputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import soundfile as sf


def _sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def validate_input_manifest(path: str | Path) -> dict:
    path = Path(path).expanduser().resolve()
    document = json.loads(path.read_text())
    issues: list[dict] = []
    checked = []
    references = document.get("references", {})
    for language in ("en", "zh"):
        item = references.get(language)
        if not isinstance(item, dict):
            issues.append({"language": language, "gate": "entry", "detail": "missing"})
            continue
        audio_path = Path(str(item.get("path", ""))).expanduser()
        if not audio_path.is_file():
            issues.append(
                {"language": language, "gate": "path", "detail": str(audio_path)}
            )
            continue
        checked.append(str(audio_path))
        if not str(item.get("transcript", "")).strip():
            issues.append(
                {"language": language, "gate": "transcript", "detail": "empty"}
            )
        if item.get("language") != language:
            issues.append(
                {
                    "language": language,
                    "gate": "language",
                    "detail": repr(item.get("language")),
                }
            )
        actual_hash = _sha256(audio_path)
        if actual_hash != item.get("sha256"):
            issues.append(
                {"language": language, "gate": "sha256", "detail": actual_hash}
            )
        info = sf.info(audio_path)
        expected = {
            "sample_rate": int(item.get("observed_sample_rate", -1)),
            "channels": int(item.get("observed_channels", -1)),
            "subtype": str(item.get("observed_subtype", "")),
        }
        actual = {
            "sample_rate": info.samplerate,
            "channels": info.channels,
            "subtype": info.subtype,
        }
        for field in expected:
            if expected[field] != actual[field]:
                issues.append(
                    {
                        "language": language,
                        "gate": field,
                        "detail": f"expected {expected[field]!r}, got {actual[field]!r}",
                    }
                )
    return {
        "schema_version": 1,
        "manifest": str(path),
        "checked": checked,
        "issues": issues,
        "pass": not issues,
    }
