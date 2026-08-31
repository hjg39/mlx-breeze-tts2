import hashlib
import json

import numpy as np
import soundfile as sf

from mlx_breeze_tts.inputs import validate_input_manifest


def _manifest(tmp_path):
    references = {}
    for language in ("en", "zh"):
        path = tmp_path / f"{language}.wav"
        sf.write(path, np.zeros(240, dtype=np.float32), 24_000, subtype="PCM_16")
        references[language] = {
            "path": str(path),
            "transcript": "Exact text" if language == "en" else "精确转写",
            "language": language,
            "observed_sample_rate": 24_000,
            "observed_channels": 1,
            "observed_subtype": "PCM_16",
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"references": references}))
    return manifest


def test_reference_manifest_validates_hash_and_audio_contract(tmp_path):
    report = validate_input_manifest(_manifest(tmp_path))
    assert report["pass"] is True
    assert len(report["checked"]) == 2


def test_reference_manifest_fails_closed_on_drift(tmp_path):
    manifest = _manifest(tmp_path)
    document = json.loads(manifest.read_text())
    document["references"]["zh"]["sha256"] = "wrong"
    document["references"]["en"]["transcript"] = ""
    manifest.write_text(json.dumps(document))
    report = validate_input_manifest(manifest)
    assert report["pass"] is False
    assert {issue["gate"] for issue in report["issues"]} == {"sha256", "transcript"}
