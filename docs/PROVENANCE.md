# Provenance and evidence ledger

## Code

The initial MLX implementation was mechanically isolated from the MIT-licensed
`Blaizzy/mlx-audio` working tree at overall commit
`512f6080727d78152349607358bfa8244dcdbee3`. The relevant files' latest commit
was `fdc2e92bc26a2daf5ecb11059245f10e36db7357` (2026-08-26). Imports, loading,
audio I/O, conversion, serving, tests, and documentation were adapted for this
standalone package.

Vendored transformer utilities identify their origin as Apple `mlx-lm`.
Vendored codec modules retain their original copyright headers. See `LICENSE`
and `THIRD_PARTY_NOTICES.md`.

## Models

| Artifact | Pinned revision | License | Status |
|---|---|---|---|
| `BreezeBlue/Breeze-TTS-2` | `c1c8ca18b70b30822735633991d9ebf4898e47d4` | BreezeBlue Research and Non-Commercial | official BF16 plus selected sensitive-BF16 8-bit/4-bit derivatives pass the standalone release verifier |
| `vanch007/Sirocco-MLX-BF16` | `4c7eec64281272a32376234075e955dce50c9252` | gated BreezeBlue Research and Non-Commercial | published verified BF16 artifact |
| `vanch007/Sirocco-MLX-8bit` | `45c58f3a91ddee3c4ce90cd10ec86ddf97e933a0` | gated BreezeBlue Research and Non-Commercial | published selected sensitive-BF16 8-bit artifact |
| `vanch007/Sirocco-MLX-4bit` | `0c4f095035f82e06574c41cdd1212ebada404638` | gated BreezeBlue Research and Non-Commercial | published selected sensitive-BF16 4-bit artifact |
| `LunaFox/Breeze-TTS-2-mlx-4bit` | `27be05f01bd8aad9628022c2bac6ded0119eef8a` | derivative subject to upstream model terms | precursor baseline only; not the selected release artifact |

## Verification status

| Claim | Status | Evidence |
|---|---|---|
| Production source contains no `mlx_audio` import | pass | `tests/test_isolation.py` |
| Package compiles and wheel/sdist build | pass | `compileall`; `uv build --no-build-isolation` |
| Built wheel installs and exposes static audit CLI | pass | isolated `/private/tmp` venv; `inspect-checkpoint` passed on pinned 4-bit snapshot |
| Required CLI and HTTP surfaces exist | pass | fake-model contract tests plus revision-bound BF16/8-bit/4-bit real HTTP probes under `reports/release-v2/*/http_evidence.json` |
| HTTP 400/409 behavior and temp cleanup | pass | `tests/test_public_surfaces.py` |
| Generation argument and audio-input failure contract | pass | Metal-independent finite/type/range validation; undecodable/empty/non-finite WAV checks; HTTP 400 mapping and cleanup regressions |
| 4-bit voice design / clone / direction / ZH event / streaming on M3 Max | pass on precursor implementation | `reports/baseline/summary.json` and WAVs |
| ASR content fidelity in five baseline cases | pass | Whisper large-v3-turbo, zero failures |
| Clone speaker similarity | pass | ECAPA cosine `0.775864`, threshold `0.25` |
| Standalone automated tests | pass | 132 passed, 0 failed on the project machine |
| Approved acceptance-matrix coverage | pass | 23 deterministic cases; complete required capability/event set; fake-runtime report and real FastAPI contract tests |
| Manual listening evidence workflow | pass | user confirmed all three variants; three revision/timestamp-bound documents and their SHA-256 hashes are stored under `reports/release-v2/*/manual_reviews.json`; all 24 event verdicts are audible |
| EN/ZH evaluation references | pass for local internal evaluation | `reports/full_matrix/input_manifest.json` and `input_validation.json`; files exist, mono 24 kHz PCM16, exact registry transcripts and SHA-256 verified |
| Objective metric integration | pass | validated ASR/ECAPA worksheet merge; event-aware CER, skill-compatible leakage similarity, min cosine/P10 aggregation, missing values fail closed |
| Automated objective backend | pass on precursor evidence | `evaluate-objective-metrics` records Whisper snapshot/artifact hashes and actual ECAPA device; `reports/baseline/objective_backend_check.json` reproduces cosine/P10 `0.775864`/`0.774540` with the standalone worker |
| Waveform/performance completion gates | pass | benchmark records load/RTF/TTFA/memory plus non-finite, clipping, repeated-tail, and stream-boundary diagnostics; verifier fails closed on missing evidence |
| PyTorch parity evidence workflow | pass | clean official source checkout and environment manifest; paired BF16/8-bit/4-bit captures under `reports/parity/`; exact template/token/codec/mask/shape/argmax checks plus recorded precision-specific tensor tolerances |
| Revision propagation and report identity | pass | HF snapshot revisions are resolved from paths; BF16→8/4-bit conversions inherit the original upstream identity; completion verifier rejects mutable/missing revisions |
| Explicit quantization policy and static consistency | pass | sensitive-BF16 v2 selected for both 8-bit and 4-bit; full candidates have hash-bound terminal parity failures and the selected candidates pass all gates |
| Low-disk auxiliary artifact storage | pass | quantized candidates hard-link immutable codec/tokenizer assets on the same volume, fall back to copies across volumes, and regenerate mutable config/audits |
| Quantization ablation selection workflow | pass | `reports/release-v2/{8bit,4bit}/quantization-comparison.json` selects sensitive-BF16 after complete pass/fail/pending-aware comparison |
| Objective metric baseline cross-check | pass on precursor evidence | `reports/baseline/objective_recalculation.json`; max CER 0.0, leakage 0.2194, ECAPA cosine/P10 0.7759/0.7745 |
| 4-bit safetensors/index static preflight | pass | `reports/baseline/checkpoint_static_audit.json`; 1,234 tensors, no index inconsistency |
| Standalone MLX architecture/model tests | pass | Metal-capable local run included in the 132-test suite and full real-device matrix |
| Standalone 4-bit real-device run | pass | `reports/release-v2/4bit/` |
| Official BF16 conversion and inference | pass | pinned download plus `reports/release-v2/bf16/` |
| 8-bit conversion and inference | pass | sensitive-BF16 policy v2 plus `reports/release-v2/8bit/` |
| PyTorch parity | pass | `reports/parity/comparison-*-v2-*.json` and BF16 2% report |
| Manual event audibility | pass | eight English/Chinese event verdicts per artifact, 24/24 audible, hash-bound in final summaries |

## Final release verification

`reports/release-v2/completion_audit.json` checks the immutable final BF16,
8-bit, and 4-bit summaries and reports `issue_count: 0`, `pass: true`. The
review documents preserve the user's explicit human verdict; no event result
was inferred from ASR or waveform metrics.
