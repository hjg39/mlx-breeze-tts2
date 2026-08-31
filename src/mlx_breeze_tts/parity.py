"""Fail-closed comparison and merge for paired PyTorch/MLX parity snapshots."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

EXACT_SECTIONS = (
    "template_render",
    "token_ids",
    "reference_audio_codes",
    "masks",
    "weight_shapes",
    "deterministic_tokens",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compare_parity_snapshots(
    pytorch_path: str | Path,
    mlx_path: str | Path,
    output_path: str | Path,
    *,
    atol: float = 1e-4,
    rtol: float = 1e-3,
) -> Path:
    if atol < 0 or rtol < 0 or not np.isfinite(atol) or not np.isfinite(rtol):
        raise ValueError("atol and rtol must be finite and non-negative")
    pytorch_path = Path(pytorch_path).expanduser().resolve()
    mlx_path = Path(mlx_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    source = json.loads(pytorch_path.read_text())
    candidate = json.loads(mlx_path.read_text())
    if source.get("schema_version") != 1 or candidate.get("schema_version") != 1:
        raise ValueError("Parity snapshots must use schema_version 1")

    checks = {}
    issues = []
    for field in ("model_id", "model_revision", "case_id"):
        passed = source.get(field) == candidate.get(field) and source.get(
            field
        ) not in (
            None,
            "",
        )
        checks[field] = {"status": "pass" if passed else "fail"}
        if not passed:
            issues.append(f"{field} differs or is missing")

    for section in EXACT_SECTIONS:
        expected = source.get(section)
        observed = candidate.get(section)
        passed = expected is not None and expected == observed
        checks[section] = {"status": "pass" if passed else "fail"}
        if not passed:
            issues.append(f"{section} differs or is missing")

    source_tensors = source.get("intermediate_tensors")
    candidate_tensors = candidate.get("intermediate_tensors")
    tensor_checks = {}
    if not isinstance(source_tensors, dict) or not isinstance(candidate_tensors, dict):
        issues.append("intermediate_tensors must be objects")
    else:
        for name in sorted(set(source_tensors) | set(candidate_tensors)):
            if name not in source_tensors or name not in candidate_tensors:
                tensor_checks[name] = {"status": "fail", "detail": "missing tensor"}
                issues.append(f"intermediate tensor set differs: {name}")
                continue
            expected = np.asarray(source_tensors[name], dtype=np.float64)
            observed = np.asarray(candidate_tensors[name], dtype=np.float64)
            shape_match = expected.shape == observed.shape
            finite = bool(
                np.all(np.isfinite(expected)) and np.all(np.isfinite(observed))
            )
            close = (
                shape_match
                and finite
                and bool(np.allclose(expected, observed, atol=atol, rtol=rtol))
            )
            delta = np.abs(expected - observed) if shape_match else np.asarray([])
            tensor_checks[name] = {
                "status": "pass" if close else "fail",
                "shape": list(observed.shape),
                "max_abs_error": float(delta.max()) if delta.size else None,
            }
            if not close:
                issues.append(f"intermediate tensor differs: {name}")
    if not tensor_checks:
        issues.append("no intermediate tensors compared")
    checks["intermediate_tensors"] = {
        "status": "pass"
        if tensor_checks
        and all(row["status"] == "pass" for row in tensor_checks.values())
        else "fail",
        "tensors": tensor_checks,
    }

    report = {
        "schema_version": 1,
        "pytorch_snapshot": str(pytorch_path),
        "pytorch_sha256": _sha256(pytorch_path),
        "mlx_snapshot": str(mlx_path),
        "mlx_sha256": _sha256(mlx_path),
        "atol": atol,
        "rtol": rtol,
        "checks": checks,
        "issues": issues,
        "status": "pass" if not issues else "fail",
        "pass": not issues,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return output_path


def apply_parity_evidence(
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
        raise ValueError("Parity evidence must be a passing schema_version 1 report")
    required = set(EXACT_SECTIONS) | {
        "model_id",
        "model_revision",
        "case_id",
        "intermediate_tensors",
    }
    checks = evidence.get("checks", {})
    if any(checks.get(name, {}).get("status") != "pass" for name in required):
        raise ValueError("Parity evidence is missing required passing checks")
    validation = summary.setdefault("validation", {})
    validation["pytorch_parity"] = "pass"
    validation["pytorch_parity_evidence"] = {
        "path": str(evidence_path),
        "sha256": _sha256(evidence_path),
        "atol": evidence.get("atol"),
        "rtol": evidence.get("rtol"),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    return output_path
