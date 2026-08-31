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
| `BreezeBlue/Breeze-TTS-2` | `c1c8ca18b70b30822735633991d9ebf4898e47d4` | BreezeBlue Research and Non-Commercial | source identified; standalone BF16 conversion pending |
| `LunaFox/Breeze-TTS-2-mlx-4bit` | `27be05f01bd8aad9628022c2bac6ded0119eef8a` | derivative subject to upstream model terms | baseline real-device pass; standalone rerun pending |

## Verification status

| Claim | Status | Evidence |
|---|---|---|
| Production source contains no `mlx_audio` import | pass | `tests/test_isolation.py` |
| Package compiles and wheel/sdist build | pass | `compileall`; `uv build --no-build-isolation` |
| Built wheel installs and exposes static audit CLI | pass | isolated `/private/tmp` venv; `inspect-checkpoint` passed on pinned 4-bit snapshot |
| Required CLI and HTTP surfaces exist | pass | dynamic fake-model API tests |
| HTTP 400/409 behavior and temp cleanup | pass | `tests/test_public_surfaces.py` |
| 4-bit voice design / clone / direction / ZH event / streaming on M3 Max | pass on precursor implementation | `reports/baseline/summary.json` and WAVs |
| ASR content fidelity in five baseline cases | pass | Whisper large-v3-turbo, zero failures |
| Clone speaker similarity | pass | ECAPA cosine `0.775864`, threshold `0.25` |
| Metal-independent standalone tests | pass | 35 passed, 0 failed |
| Approved acceptance-matrix coverage | pass | 23 deterministic cases; complete required capability/event set; fake-runtime report and real FastAPI contract tests |
| Manual listening evidence workflow | pass | HTML JSON export plus validated, non-destructive CLI merge tests |
| EN/ZH evaluation references | pass for local internal evaluation | `reports/full_matrix/input_manifest.json` and `input_validation.json`; files exist, mono 24 kHz PCM16, exact registry transcripts and SHA-256 verified |
| Objective metric integration | pass | validated ASR/ECAPA worksheet merge; event-aware CER, skill-compatible leakage similarity, min cosine/P10 aggregation, missing values fail closed |
| Objective metric baseline cross-check | pass on precursor evidence | `reports/baseline/objective_recalculation.json`; max CER 0.0, leakage 0.2194, ECAPA cosine/P10 0.7759/0.7745 |
| 4-bit safetensors/index static preflight | pass | `reports/baseline/checkpoint_static_audit.json`; 1,234 tensors, no index inconsistency |
| Standalone MLX architecture/model tests | pending | 2 modules skipped because this execution sandbox cannot initialize Metal |
| Standalone 4-bit real-device run | pending | `reports/current_runtime_probe.json`; current execution sandbox cannot initialize Metal |
| Official BF16 conversion and inference | pending | source artifact not yet downloaded in this run |
| 8-bit conversion and inference | pending | depends on BF16 artifact |
| PyTorch parity | missing evidence | paired upstream run not yet captured |
| Manual event audibility | pending | listening review required |

Do not promote any pending row to pass without adding the machine-readable
artifact and exact command/revision used.

## Current execution blockers

- The execution sandbox reports `No Metal device available`; this prevents a
  fresh standalone model load even though the earlier precursor run used the
  same M3 Max successfully.
- The data volume currently has about 12 GiB free. A normal duplicate BF16
  conversion remains too close to the disk limit, so `link-bf16` now creates a
  same-volume hard-linked candidate without duplicating safetensors blocks.
- Browser security denied the official Hugging Face download, and the command
  sandbox currently cannot resolve Hugging Face/PyPI hosts. The pinned local
  snapshot or an approved working download channel is still required before
  BF16/8-bit conversion can run.
