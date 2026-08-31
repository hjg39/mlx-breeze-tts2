"""Validated merge of manual listening verdicts into benchmark evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

_ALLOWED = {
    "manual_content": {"pending", "pass", "fail"},
    "manual_voice": {"pending", "pass", "fail", "n/a"},
    "manual_event": {"pending", "audible", "missing", "n/a"},
}


def apply_reviews(
    summary_path: str | Path,
    reviews_path: str | Path,
    output_path: str | Path,
) -> Path:
    summary_path = Path(summary_path).expanduser()
    reviews_path = Path(reviews_path).expanduser()
    output_path = Path(output_path).expanduser()
    if output_path.resolve() == summary_path.resolve():
        raise ValueError("Output must differ from the source summary")
    summary = json.loads(summary_path.read_text())
    document = json.loads(reviews_path.read_text())
    reviews = document.get("reviews")
    if document.get("schema_version") != 1 or not isinstance(reviews, list):
        raise ValueError("Review document must use schema_version 1 and reviews[]")
    review_revision = document.get("model_revision")
    summary_revision = summary.get("model_revision")
    if review_revision is not None and review_revision != summary_revision:
        raise ValueError("Review model_revision does not match the summary")

    samples = {
        sample.get("capability"): sample for sample in summary.get("samples", [])
    }
    seen: set[str] = set()
    for review in reviews:
        capability = review.get("capability")
        if not capability or capability not in samples:
            raise ValueError(f"Unknown review capability: {capability!r}")
        if capability in seen:
            raise ValueError(f"Duplicate review capability: {capability}")
        seen.add(capability)
        for field, allowed in _ALLOWED.items():
            value = review.get(field, "pending")
            if value not in allowed:
                raise ValueError(f"Invalid {field} verdict for {capability}: {value!r}")
            samples[capability][field] = value
        samples[capability]["manual_notes"] = str(review.get("manual_notes", ""))

    summary.setdefault("validation", {})["manual_listening"] = (
        "reviewed"
        if seen == set(samples)
        and all(
            sample.get("manual_content") != "pending" for sample in samples.values()
        )
        else "partial"
    )
    summary["validation"]["manual_listening_evidence"] = {
        "path": str(reviews_path.resolve()),
        "sha256": hashlib.sha256(reviews_path.read_bytes()).hexdigest(),
        "exported_at": document.get("exported_at"),
        "model_revision": review_revision,
        "review_count": len(reviews),
        "reviewed_capabilities": sorted(seen),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    return output_path
