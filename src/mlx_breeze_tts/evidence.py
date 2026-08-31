"""Fail-closed verifier for the approved completion gates."""

import hashlib
import json
from pathlib import Path

from .parity import EXACT_SECTIONS

REQUIRED_VARIANTS = ("bf16", "8bit", "4bit")
REQUIRED_CAPABILITIES = {
    "voice_design_en",
    "voice_design_zh",
    "voice_clone_en",
    "voice_clone_zh",
    "cross_clone_en_to_zh",
    "cross_clone_zh_to_en",
    "voice_direction_en",
    "voice_direction_zh",
    "nonstream_en",
    "streaming_en",
    "stream_cancel",
    "long_text",
    "steady_state",
}
REQUIRED_EVENTS = {
    "event_en_laugh",
    "event_en_cough",
    "event_en_clears_throat",
    "event_en_sigh",
    "event_zh_laugh",
    "event_zh_cough",
    "event_zh_clears_throat",
    "event_zh_sigh",
}
REQUIRED_INTERFACES = ("python", "cli", "http", "streaming")


def _issue(issues: list[dict], variant: str, gate: str, detail: str) -> None:
    issues.append({"variant": variant, "gate": gate, "detail": detail})


def _number(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parity_evidence_passes(validation: dict) -> bool:
    metadata = validation.get("pytorch_parity_evidence")
    if not isinstance(metadata, dict):
        return False
    path = Path(str(metadata.get("path", ""))).expanduser()
    if not path.is_file():
        return False
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != metadata.get("sha256"):
        return False
    try:
        report = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    required = set(EXACT_SECTIONS) | {
        "model_id",
        "model_revision",
        "case_id",
        "intermediate_tensors",
    }
    checks = report.get("checks", {})
    return (
        report.get("schema_version") == 1
        and report.get("pass") is True
        and all(checks.get(name, {}).get("status") == "pass" for name in required)
    )


def verify_evidence_bundle(root: str | Path) -> dict:
    root = Path(root).expanduser()
    issues: list[dict] = []
    checked = []
    for variant in REQUIRED_VARIANTS:
        candidates = (
            root / variant / "summary.final.json",
            root / variant / "summary.reviewed.json",
            root / variant / "summary.metrics.json",
            root / variant / "summary.json",
        )
        path = next(
            (candidate for candidate in candidates if candidate.is_file()),
            candidates[-1],
        )
        if not path.is_file():
            _issue(issues, variant, "summary", f"missing {path}")
            continue
        try:
            report = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            _issue(issues, variant, "summary", f"invalid JSON: {exc}")
            continue
        checked.append(str(path))

        audit = report.get("artifact_audit", {})
        if not audit.get("pass"):
            _issue(issues, variant, "artifact_audit", "strict load audit is not pass")
        if (
            audit.get("missing")
            or audit.get("unexpected")
            or audit.get("shape_mismatches")
        ):
            _issue(issues, variant, "artifact_audit", "weight audit is not clean")

        interfaces = report.get("interfaces", {})
        for name in REQUIRED_INTERFACES:
            if interfaces.get(name) != "pass":
                _issue(issues, variant, f"interface_{name}", "not pass")

        samples = {item.get("capability"): item for item in report.get("samples", [])}
        missing_capabilities = sorted(REQUIRED_CAPABILITIES - set(samples))
        if missing_capabilities:
            _issue(
                issues,
                variant,
                "capability_matrix",
                "missing: " + ", ".join(missing_capabilities),
            )
        missing_events = sorted(REQUIRED_EVENTS - set(samples))
        if missing_events:
            _issue(
                issues,
                variant,
                "event_matrix",
                "missing: " + ", ".join(missing_events),
            )

        for capability, item in samples.items():
            if item.get("empty_audio") is True or _number(item.get("samples"), 0) <= 0:
                _issue(issues, variant, capability, "empty audio")
            if _number(item.get("clipping_fraction"), 1.0) > 0:
                _issue(issues, variant, capability, "clipping detected")
            if capability in REQUIRED_EVENTS and item.get("manual_event") != "audible":
                _issue(
                    issues, variant, capability, "manual event verdict is not audible"
                )

        validation = report.get("validation", {})
        if _number(validation.get("max_cer"), 1.0) > 0.05:
            _issue(issues, variant, "content", "max CER exceeds 0.05 or is missing")
        if _number(validation.get("clone_cosine_min"), -1.0) < 0.25:
            _issue(issues, variant, "speaker", "clone cosine minimum below 0.25")
        if _number(validation.get("clone_p10_min"), -1.0) < 0.25:
            _issue(issues, variant, "speaker", "clone P10 below 0.25")
        if _number(validation.get("leakage_max"), 1.0) >= 0.65:
            _issue(issues, variant, "leakage", "leakage is >= 0.65 or missing")
        if validation.get("seed_reproducibility") != "pass":
            _issue(issues, variant, "seed", "fixed-seed reproducibility not pass")
        if validation.get("sampling_path") != "pass":
            _issue(issues, variant, "sampling", "sampling controls not proven")
        if validation.get("waveform_integrity") != "pass":
            _issue(
                issues,
                variant,
                "waveform_integrity",
                "non-finite/clipping/repeated-tail/stream continuity checks not pass",
            )
        if validation.get("pytorch_parity") != "pass" or not _parity_evidence_passes(
            validation
        ):
            _issue(
                issues,
                variant,
                "pytorch_parity",
                "PyTorch parity report, required checks, or SHA-256 is not pass",
            )

        performance = report.get("performance", {})
        for metric in (
            "load_time_s",
            "load_peak_memory_gb",
            "peak_memory_gb",
            "steady_state_rtf",
            "long_text_rtf",
            "streaming_ttfa_s",
        ):
            value = _number(performance.get(metric), -1.0)
            if value < 0 or not value < float("inf"):
                _issue(
                    issues,
                    variant,
                    "performance",
                    f"{metric} is missing or invalid",
                )

        for filename in ("report.md", "index.html"):
            if not (root / variant / filename).is_file():
                _issue(issues, variant, "deliverable", f"missing {filename}")
        if not any((root / variant).glob("*.wav")):
            _issue(issues, variant, "deliverable", "no WAV evidence")

    return {
        "schema_version": 1,
        "root": str(root),
        "required_variants": list(REQUIRED_VARIANTS),
        "checked": checked,
        "issue_count": len(issues),
        "issues": issues,
        "pass": not issues,
    }
