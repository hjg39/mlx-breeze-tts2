"""Evidence-backed selection between quantization policy candidates."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

_EVENTS = {
    "event_en_laugh",
    "event_en_cough",
    "event_en_clears_throat",
    "event_en_sigh",
    "event_zh_laugh",
    "event_zh_cough",
    "event_zh_clears_throat",
    "event_zh_sigh",
}
_PERFORMANCE = (
    "load_time_s",
    "load_peak_memory_gb",
    "peak_memory_gb",
    "steady_state_rtf",
    "long_text_rtf",
    "streaming_ttfa_s",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite(value) -> bool:
    try:
        return math.isfinite(float(value)) and float(value) >= 0
    except (TypeError, ValueError):
        return False


def _candidate(path: Path) -> dict:
    summary = json.loads(path.read_text())
    provenance = summary.get("model_provenance") or {}
    policy = provenance.get("quantization_policy") or {}
    validation = summary.get("validation") or {}
    issues = []
    name = policy.get("name")
    if name not in {"full", "sensitive-bf16"}:
        issues.append("missing supported quantization policy")
    if not summary.get("artifact_audit", {}).get("pass"):
        issues.append("artifact audit is not pass")
    if any(
        summary.get("interfaces", {}).get(name) != "pass"
        for name in ("python", "cli", "http", "streaming")
    ):
        issues.append("public interfaces are not all pass")
    thresholds = (
        (validation.get("corpus_cer"), lambda value: value <= 0.05, "corpus CER"),
        (
            validation.get("clone_cosine_min"),
            lambda value: value >= 0.25,
            "clone cosine",
        ),
        (
            validation.get("clone_p10_min"),
            lambda value: value >= 0.25,
            "clone P10",
        ),
        (
            validation.get("leakage_max"),
            lambda value: value < 0.65,
            "reference leakage",
        ),
    )
    for raw, predicate, label in thresholds:
        try:
            passed = math.isfinite(float(raw)) and predicate(float(raw))
        except (TypeError, ValueError):
            passed = False
        if not passed:
            issues.append(f"{label} threshold is not pass")
    for gate in (
        "seed_reproducibility",
        "sampling_path",
        "waveform_integrity",
        "pytorch_parity",
    ):
        if validation.get(gate) != "pass":
            issues.append(f"{gate} is not pass")
    samples = {
        sample.get("capability"): sample for sample in summary.get("samples", [])
    }
    if any(samples.get(name, {}).get("manual_event") != "audible" for name in _EVENTS):
        issues.append("all eight manual event verdicts are not audible")
    performance = summary.get("performance") or {}
    if any(not _finite(performance.get(metric)) for metric in _PERFORMANCE):
        issues.append("performance evidence is incomplete")
    artifact_bytes = provenance.get("artifact_bytes")
    if not _finite(artifact_bytes) or float(artifact_bytes) <= 0:
        issues.append("artifact size is missing")
    return {
        "summary": str(path),
        "sha256": _sha256(path),
        "policy": name,
        "bits": provenance.get("bits"),
        "model_revision": summary.get("model_revision"),
        "artifact_bytes": artifact_bytes,
        "steady_state_rtf": performance.get("steady_state_rtf"),
        "issues": issues,
        "pass": not issues,
    }


def compare_quantization_candidates(
    full_path: str | Path,
    sensitive_path: str | Path,
    output_path: str | Path,
) -> Path:
    full_path = Path(full_path).expanduser().resolve()
    sensitive_path = Path(sensitive_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    candidates = [_candidate(full_path), _candidate(sensitive_path)]
    issues = []
    by_policy = {candidate["policy"]: candidate for candidate in candidates}
    if set(by_policy) != {"full", "sensitive-bf16"}:
        issues.append("one full and one sensitive-bf16 candidate are required")
    if len({candidate["bits"] for candidate in candidates}) != 1 or candidates[0][
        "bits"
    ] not in {4, 8}:
        issues.append("candidates must use the same 4-bit or 8-bit precision")
    if len({candidate["model_revision"] for candidate in candidates}) != 1:
        issues.append("candidates must use the same immutable model revision")
    passing = [candidate for candidate in candidates if candidate["pass"]]
    selected = None
    if not passing:
        issues.append("neither candidate passes the approved quality gates")
    elif len(passing) == 1:
        selected = passing[0]
    else:
        selected = min(
            passing,
            key=lambda candidate: (
                float(candidate["artifact_bytes"]),
                float(candidate["steady_state_rtf"]),
                candidate["policy"] != "sensitive-bf16",
            ),
        )
    report = {
        "schema_version": 1,
        "candidates": candidates,
        "selected_policy": selected["policy"] if selected else None,
        "selected_summary": selected["summary"] if selected else None,
        "selected_summary_sha256": selected["sha256"] if selected else None,
        "bits": selected["bits"] if selected else None,
        "model_revision": selected["model_revision"] if selected else None,
        "selection_rule": "quality gates, then artifact bytes, steady-state RTF, sensitive tie-break",
        "issues": issues,
        "pass": not issues and selected is not None,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return output_path


def apply_quantization_evidence(
    summary_path: str | Path, evidence_path: str | Path, output_path: str | Path
) -> Path:
    summary_path = Path(summary_path).expanduser().resolve()
    evidence_path = Path(evidence_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    if output_path == summary_path:
        raise ValueError("Output must differ from the source summary")
    summary = json.loads(summary_path.read_text())
    evidence = json.loads(evidence_path.read_text())
    if evidence.get("schema_version") != 1 or evidence.get("pass") is not True:
        raise ValueError(
            "Quantization evidence must be a passing schema_version 1 report"
        )
    if _sha256(summary_path) != evidence.get("selected_summary_sha256"):
        raise ValueError("Summary is not the selected quantization candidate")
    provenance = summary.get("model_provenance") or {}
    policy = provenance.get("quantization_policy") or {}
    if policy.get("name") != evidence.get("selected_policy"):
        raise ValueError("Selected policy does not match summary provenance")
    if provenance.get("bits") != evidence.get("bits"):
        raise ValueError("Selected precision does not match summary provenance")
    if summary.get("model_revision") != evidence.get("model_revision"):
        raise ValueError("Selected revision does not match summary provenance")
    validation = summary.setdefault("validation", {})
    validation["quantization_ablation"] = "pass"
    validation["quantization_ablation_evidence"] = {
        "path": str(evidence_path),
        "sha256": _sha256(evidence_path),
        "selected_policy": evidence["selected_policy"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    return output_path
