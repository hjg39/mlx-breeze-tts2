"""Fail-closed verifier for the approved completion gates."""

import json
from pathlib import Path

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


def verify_evidence_bundle(root: str | Path) -> dict:
    root = Path(root).expanduser()
    issues: list[dict] = []
    checked = []
    for variant in REQUIRED_VARIANTS:
        path = root / variant / "summary.json"
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
            if item.get("empty_audio") is True or item.get("samples", 0) <= 0:
                _issue(issues, variant, capability, "empty audio")
            if float(item.get("clipping_fraction", 1.0)) > 0:
                _issue(issues, variant, capability, "clipping detected")
            if capability in REQUIRED_EVENTS and item.get("manual_event") != "audible":
                _issue(
                    issues, variant, capability, "manual event verdict is not audible"
                )

        validation = report.get("validation", {})
        if float(validation.get("max_cer", 1.0)) > 0.05:
            _issue(issues, variant, "content", "max CER exceeds 0.05 or is missing")
        if float(validation.get("clone_cosine_min", -1.0)) < 0.25:
            _issue(issues, variant, "speaker", "clone cosine minimum below 0.25")
        if float(validation.get("clone_p10_min", -1.0)) < 0.25:
            _issue(issues, variant, "speaker", "clone P10 below 0.25")
        if float(validation.get("leakage_max", 1.0)) >= 0.65:
            _issue(issues, variant, "leakage", "leakage is >= 0.65 or missing")
        if validation.get("seed_reproducibility") != "pass":
            _issue(issues, variant, "seed", "fixed-seed reproducibility not pass")
        if validation.get("sampling_path") != "pass":
            _issue(issues, variant, "sampling", "sampling controls not proven")

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
