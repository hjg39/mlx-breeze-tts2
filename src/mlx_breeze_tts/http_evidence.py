"""Revision-bound capture and merge for the real HTTP inference contract."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from .loader import load, resolve_model_path
from .provenance import checkpoint_provenance

REQUIRED_HTTP_CHECKS = (
    "health_200",
    "health_sample_rate",
    "speech_200",
    "content_type_pcm",
    "sample_rate_header",
    "sample_format_header",
    "cache_control",
    "nonempty_even_pcm",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def http_report_passes(report: dict, model_revision: str | None = None) -> bool:
    probe = report.get("probe") or {}
    checks = probe.get("checks") or {}
    return (
        report.get("schema_version") == 1
        and (model_revision is None or report.get("model_revision") == model_revision)
        and probe.get("status") == "pass"
        and probe.get("speech_status_code") == 200
        and all(checks.get(name) is True for name in REQUIRED_HTTP_CHECKS)
    )


def capture_http_evidence(model_id: str | Path, output_path: str | Path) -> Path:
    from .benchmark import _probe_http

    output_path = Path(output_path).expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite HTTP evidence: {output_path}")
    resolved = resolve_model_path(model_id)
    provenance = checkpoint_provenance(resolved)
    model = load(resolved)
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": str(model_id),
        "resolved_model_path": str(resolved),
        "model_revision": provenance.get("artifact_revision")
        or provenance.get("upstream_revision"),
        "model_provenance": provenance,
        "probe": _probe_http(model, 42),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return output_path


def apply_http_evidence(
    summary_path: str | Path, evidence_path: str | Path, output_path: str | Path
) -> Path:
    summary_path = Path(summary_path).expanduser().resolve()
    evidence_path = Path(evidence_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    if output_path == summary_path:
        raise ValueError("Output must differ from the source summary")
    if output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite merged HTTP evidence: {output_path}"
        )
    summary = json.loads(summary_path.read_text())
    evidence = json.loads(evidence_path.read_text())
    if not http_report_passes(evidence, summary.get("model_revision")):
        raise ValueError(
            "HTTP evidence is not a matching passing schema_version 1 report"
        )
    summary.setdefault("interfaces", {})["http"] = "pass"
    summary.setdefault("interface_details", {})["http"] = evidence["probe"]
    summary.setdefault("validation", {})["http_evidence"] = {
        "path": str(evidence_path),
        "sha256": _sha256(evidence_path),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    return output_path
