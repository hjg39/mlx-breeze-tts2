"""Canonical Breeze TTS 2 acceptance matrix, independent of MLX runtime."""

from __future__ import annotations

from dataclasses import dataclass

from .evidence import REQUIRED_CAPABILITIES, REQUIRED_EVENTS


@dataclass(frozen=True)
class ReferencePair:
    audio: str
    text: str
    language: str

    def __post_init__(self) -> None:
        if not self.audio or not self.text:
            raise ValueError(
                "Reference audio and exact transcript must both be non-empty"
            )
        if self.language not in {"en", "zh"}:
            raise ValueError("Reference language must be en or zh")


def reference_pair(
    audio: str | None, text: str | None, language: str
) -> ReferencePair | None:
    if bool(audio) != bool(text):
        raise ValueError(
            f"{language} reference audio and exact transcript must be supplied together"
        )
    return ReferencePair(audio, text, language) if audio and text else None


def _clone_case(
    capability: str,
    text: str,
    reference: ReferencePair | None,
    *,
    instruct: str | None = None,
) -> dict:
    case = {"capability": capability, "text": text, "instruct": instruct}
    if reference:
        case.update(
            {
                "ref_audio": reference.audio,
                "ref_text": reference.text,
                "reference_language": reference.language,
            }
        )
    else:
        case["skip_reason"] = "required reference pair is missing"
    return case


def build_acceptance_matrix(
    reference_en: ReferencePair | None = None,
    reference_zh: ReferencePair | None = None,
) -> list[dict]:
    """Return every approved capability/event case in deterministic order."""

    cases = [
        {
            "capability": "voice_design_en",
            "text": "Welcome aboard. Your journey begins now.",
            "instruct": "A warm, thoughtful young woman with a clear voice.",
        },
        {
            "capability": "voice_design_zh",
            "text": "欢迎来到今晚的故事时间。",
            "instruct": "一位温柔自信的年轻女性，声音清晰，语气亲切。",
        },
        _clone_case(
            "voice_clone_en",
            "Please read this sentence in the reference voice.",
            reference_en,
        ),
        _clone_case(
            "voice_clone_zh",
            "请使用参考音色清晰自然地朗读这段内容，保持语速稳定并准确表达完整意思。",
            reference_zh,
        ),
        _clone_case(
            "cross_clone_en_to_zh",
            "这是一次从英文参考音色到中文正文的跨语言测试。",
            reference_en,
        ),
        _clone_case(
            "cross_clone_zh_to_en",
            "This is a cross-language test from a Chinese reference voice.",
            reference_zh,
        ),
        _clone_case(
            "voice_direction_en",
            "We need to discuss what happened last night.",
            reference_en,
            instruct="Speak slowly with a restrained, serious tone.",
        ),
        _clone_case(
            "voice_direction_zh",
            "我们需要认真讨论昨晚发生的事情。",
            reference_zh,
            instruct="语速缓慢，语气克制而严肃。",
        ),
        {
            "capability": "nonstream_en",
            "text": "This sentence validates complete non-streaming generation.",
        },
        {
            "capability": "streaming_en",
            "text": "This response is streamed incrementally as it is generated.",
            "stream": True,
        },
        {
            "capability": "stream_cancel",
            "text": "This longer response will be cancelled after its first audio chunk, then the decoder state will be tested again.",
            "post_cancel_text": "Decoder state was reset after cancellation.",
            "stream": True,
            "cancel_after_first_chunk": True,
        },
        {
            "capability": "long_text",
            "text": (
                "Apple Silicon combines high memory bandwidth with an efficient unified "
                "memory architecture. This longer passage verifies that the model can "
                "maintain intelligibility, pacing, and stable generation across several "
                "sentences without silently dropping the final clause."
            ),
        },
        {
            "capability": "steady_state",
            "text": "Steady state performance is measured after warmup.",
            "iterations": 5,
        },
        {
            "capability": "sampling_controls",
            "text": "This sentence exercises every public sampling control.",
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 25,
            "repetition_penalty": 1.05,
        },
        {
            "capability": "seed_reproducibility",
            "text": "A fixed seed should reproduce the same generated waveform.",
            "repetitions_for_equality": 2,
        },
    ]
    event_cases = [
        ("event_en_laugh", "(laugh) I did not expect that answer.", None),
        ("event_en_cough", "(cough) Please excuse me for a moment.", 7),
        (
            "event_en_clears_throat",
            "(clears throat) May I have your attention?",
            None,
        ),
        ("event_en_sigh", "(sigh) We have a long way to go.", 3),
        ("event_zh_laugh", "[笑] 这个答案真让人意外。", None),
        ("event_zh_cough", "[咳嗽] 不好意思，请稍等一下。", None),
        ("event_zh_clears_throat", "[清嗓子] 请大家注意。", 1),
        ("event_zh_sigh", "[叹气] 我们还有很长的路要走。", None),
    ]
    cases.extend(
        {
            "capability": capability,
            "text": text,
            "manual_event": "pending",
            **({"seed": case_seed} if case_seed is not None else {}),
        }
        for capability, text, case_seed in event_cases
    )
    return cases


def matrix_coverage(cases: list[dict]) -> dict:
    capabilities = {case["capability"] for case in cases}
    required = REQUIRED_CAPABILITIES | REQUIRED_EVENTS
    duplicates = sorted(
        capability
        for capability in capabilities
        if sum(case["capability"] == capability for case in cases) > 1
    )
    return {
        "required": sorted(required),
        "present": sorted(capabilities),
        "missing": sorted(required - capabilities),
        "duplicates": duplicates,
        "pass": not (required - capabilities) and not duplicates,
    }
