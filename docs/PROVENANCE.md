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
| Package compiles | pass | Python `compileall` |
| Required CLI and HTTP surfaces exist | pass | static tests |
| 4-bit voice design / clone / direction / ZH event / streaming on M3 Max | pass on precursor implementation | `reports/baseline/summary.json` and WAVs |
| ASR content fidelity in five baseline cases | pass | Whisper large-v3-turbo, zero failures |
| Clone speaker similarity | pass | ECAPA cosine `0.775864`, threshold `0.25` |
| Standalone package full unit suite | pending | current execution sandbox cannot initialize Metal |
| Standalone 4-bit real-device run | pending | current execution sandbox cannot initialize Metal |
| Official BF16 conversion and inference | pending | source artifact not yet downloaded in this run |
| 8-bit conversion and inference | pending | depends on BF16 artifact |
| PyTorch parity | missing evidence | paired upstream run not yet captured |
| Manual event audibility | pending | listening review required |

Do not promote any pending row to pass without adding the machine-readable
artifact and exact command/revision used.
