"""Objective ASR, leakage, and speaker-metric evidence integration."""

from __future__ import annotations

import json
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

_EVENT_RE = re.compile(
    r"\((?:laugh|cough|clears throat|sigh)\)|\[(?:笑|咳嗽|清嗓子|叹气)\]",
    re.IGNORECASE,
)
_REFERENCE_CAPABILITIES = {
    "voice_clone_en",
    "voice_clone_zh",
    "cross_clone_en_to_zh",
    "cross_clone_zh_to_en",
    "voice_direction_en",
    "voice_direction_zh",
}


def _is_standard_content(capability: str) -> bool:
    """Exclude non-verbal events from the linguistic-content CER gate."""

    return not capability.startswith("event_")


def normalize_text(text: str, *, strip_events: bool = False) -> str:
    value = unicodedata.normalize("NFKC", text or "").lower()
    if strip_events:
        value = _EVENT_RE.sub("", value)
    return "".join(character for character in value if character.isalnum())


def character_error_rate(expected: str, observed: str) -> float:
    source = normalize_text(expected, strip_events=True)
    target = normalize_text(observed)
    edits = _character_edit_distance(source, target)
    return edits / len(source) if source else (0.0 if not target else 1.0)


def _character_edit_distance(source: str, target: str) -> int:
    if not source:
        return len(target)
    previous = list(range(len(target) + 1))
    for source_index, source_character in enumerate(source, 1):
        current = [source_index]
        for target_index, target_character in enumerate(target, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[target_index] + 1,
                    previous[target_index - 1] + (source_character != target_character),
                )
            )
        previous = current
    return previous[-1]


def reference_leakage_similarity(reference_text: str, asr_text: str) -> float:
    reference = normalize_text(reference_text)
    observed = normalize_text(asr_text)
    if not reference or not observed:
        return 0.0
    return SequenceMatcher(None, reference, observed).ratio()


def objective_template(summary_path: str | Path, output_path: str | Path) -> Path:
    summary_path = Path(summary_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    summary = json.loads(summary_path.read_text())
    rows = []
    for sample in summary.get("samples", []):
        audio = sample.get("audio", "")
        rows.append(
            {
                "capability": sample.get("capability"),
                "audio": str((summary_path.parent / audio).resolve()) if audio else "",
                "expected_text": sample.get("expected_text") or sample.get("text", ""),
                "reference_audio": (
                    str((summary_path.parent / sample["ref_audio"]).resolve())
                    if sample.get("ref_audio")
                    and not Path(str(sample["ref_audio"])).expanduser().is_absolute()
                    else sample.get("ref_audio")
                ),
                "reference_text": sample.get("ref_text"),
                "asr_text": None,
                "speaker_cosine": None,
                "speaker_p10": None,
                "asr_backend": None,
                "speaker_backend": None,
            }
        )
    document = {"schema_version": 1, "summary": str(summary_path), "rows": rows}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(document, ensure_ascii=False, indent=2))
    return output_path


def _score(value, name: str, capability: str) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {name} for {capability}: {value!r}") from exc
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} for {capability} must be between 0 and 1")
    return number


def apply_objective_metrics(
    summary_path: str | Path,
    metrics_path: str | Path,
    output_path: str | Path,
) -> Path:
    summary_path = Path(summary_path).expanduser().resolve()
    metrics_path = Path(metrics_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    if output_path == summary_path:
        raise ValueError("Output must differ from the source summary")
    summary = json.loads(summary_path.read_text())
    document = json.loads(metrics_path.read_text())
    rows = document.get("rows")
    if document.get("schema_version") != 1 or not isinstance(rows, list):
        raise ValueError("Metrics document must use schema_version 1 and rows[]")

    samples = {
        sample.get("capability"): sample for sample in summary.get("samples", [])
    }
    if document.get("evaluation"):
        summary["objective_provenance"] = document["evaluation"]
    seen: set[str] = set()
    for row in rows:
        capability = row.get("capability")
        if not capability or capability not in samples:
            raise ValueError(f"Unknown metric capability: {capability!r}")
        if capability in seen:
            raise ValueError(f"Duplicate metric capability: {capability}")
        seen.add(capability)
        sample = samples[capability]
        asr_text = str(row.get("asr_text") or "").strip()
        sample["asr_text"] = asr_text or None
        sample["asr_backend"] = row.get("asr_backend")
        if asr_text:
            expected = sample.get("expected_text") or sample.get("text", "")
            sample["cer"] = character_error_rate(expected, asr_text)
            if sample.get("ref_text"):
                sample["reference_leakage"] = reference_leakage_similarity(
                    sample["ref_text"], asr_text
                )
        cosine = _score(row.get("speaker_cosine"), "speaker_cosine", capability)
        p10 = _score(row.get("speaker_p10"), "speaker_p10", capability)
        sample["speaker_cosine"] = cosine
        sample["speaker_p10"] = p10
        sample["speaker_backend"] = row.get("speaker_backend")

    generated = [
        sample
        for sample in samples.values()
        if sample.get("status") == "audio_generated"
    ]
    standard_content = [
        sample
        for sample in generated
        if _is_standard_content(str(sample.get("capability") or ""))
    ]
    corpus_characters = 0
    corpus_edits = 0
    for sample in standard_content:
        expected = normalize_text(
            sample.get("expected_text") or sample.get("text", ""),
            strip_events=True,
        )
        observed = normalize_text(sample.get("asr_text") or "")
        corpus_characters += len(expected)
        corpus_edits += _character_edit_distance(expected, observed)
    missing_asr = sorted(
        sample["capability"] for sample in generated if sample.get("cer") is None
    )
    referenced = [
        samples[capability]
        for capability in sorted(_REFERENCE_CAPABILITIES & set(samples))
        if samples[capability].get("status") == "audio_generated"
    ]
    missing_speaker = sorted(
        sample["capability"]
        for sample in referenced
        if sample.get("speaker_cosine") is None or sample.get("speaker_p10") is None
    )
    missing_leakage = sorted(
        sample["capability"]
        for sample in referenced
        if sample.get("reference_leakage") is None
    )
    validation = summary.setdefault("validation", {})
    validation.update(
        {
            "max_cer": max((sample["cer"] for sample in standard_content), default=None)
            if not missing_asr
            else None,
            "corpus_cer": (
                corpus_edits / corpus_characters if corpus_characters else None
            )
            if not missing_asr
            else None,
            "corpus_cer_edits": corpus_edits if not missing_asr else None,
            "corpus_cer_characters": corpus_characters if not missing_asr else None,
            "cer_scope": "linguistic_content_excluding_events",
            "clone_cosine_min": min(
                (sample["speaker_cosine"] for sample in referenced), default=None
            )
            if referenced and not missing_speaker
            else None,
            "clone_p10_min": min(
                (sample["speaker_p10"] for sample in referenced), default=None
            )
            if referenced and not missing_speaker
            else None,
            "leakage_max": max(
                (sample["reference_leakage"] for sample in referenced), default=None
            )
            if referenced and not missing_leakage
            else None,
            "objective_metrics": "complete"
            if not missing_asr and not missing_speaker and not missing_leakage
            else "partial",
            "missing_asr": missing_asr,
            "missing_speaker": missing_speaker,
            "missing_leakage": missing_leakage,
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    return output_path
