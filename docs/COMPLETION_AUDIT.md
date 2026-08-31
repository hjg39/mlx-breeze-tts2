# Completion audit

Audit date: 2026-09-01. Scope is the approved specification in
`docs/design/2026-08-31-approved-spec.md`. This ledger separates evidence that
already passes from the human-only gate that is still pending; it is not a
release-complete claim.

| Approved requirement | Status | Authoritative evidence |
|---|---|---|
| Research new, visible TTS projects on GitHub and X and select one | pass | `docs/research/2026-08-31-hot-tts-selection.{md,json}` records dated GitHub/X snapshots, source URLs, conflicts, and the Breeze TTS 2 decision |
| Independent installable MLX project | pass | `pyproject.toml`, `uv.lock`, `tests/test_isolation.py`; `uv build` succeeds and production source has no `mlx_audio` dependency |
| Full MLX compute graph and Qwen3-TTS codec | pass | `src/mlx_breeze_tts/{model,codec,core}` plus strict BF16/8-bit/4-bit artifact audits embedded in `reports/release-v2/*/summary.http.json` |
| Voice design, clone, direction, English/Chinese events | pending | all 23 cases generated on Metal for all three precisions; objective gates pass, but the eight event cases per artifact still require exported human `audible` verdicts |
| Non-streaming and incremental streaming, cancellation and reset | pass | `nonstream_en`, `streaming_en`, and `stream_cancel` WAV/metrics in every release-v2 artifact; waveform continuity validation is `pass` |
| Sampling controls and deterministic seed | pass | `sampling_controls` and `seed_reproducibility` cases plus `validation.sampling_path` and `validation.seed_reproducibility` in each summary |
| Python API and CLI | pass | `src/mlx_breeze_tts/{engine,cli}.py`, public-surface tests, and per-artifact interface results |
| Real HTTP `/health` and streaming `/v1/audio/speech` contract | pass | revision-bound `reports/release-v2/*/http_evidence.json`; PCM headers/status/body checks pass for BF16, 8-bit, and 4-bit |
| Reproducible official BF16, 8-bit, and 4-bit conversion | pass | pinned upstream revision `c1c8ca18b70b30822735633991d9ebf4898e47d4`, conversion/audit commands, checkpoint provenance and weight hashes in each summary |
| PyTorch behavior parity | pass | hash-bound reports in `reports/parity/`; exact templates/tokens/codec/masks/shapes and precision-specific tensor tolerances are verified |
| Objective quality gates | pass | real Whisper and ECAPA results in release-v2: corpus CER `0.01042/0.01786/0.00595`, clone cosine and P10 above `0.25`, leakage below `0.65` |
| Performance and resource measurements | pass | every summary records load time, load/total peak memory, steady-state and long-text RTF, and streaming TTFA |
| JSON, Markdown, WAV, and HTML evidence | pass | `reports/release-v2/{bf16,8bit,4bit}` and their listening pages |
| Tests, compilation, and package build | pass | 127 tests pass; `compileall`, `uv build`, shell syntax, lock check, and `git diff --check` pass on the project machine |
| Final reviewed release bundle and quantization selection | pending | `reports/release-v2/pre_review_audit-v2.json` has 29 issues: 24 event verdicts, 3 listening documents, and 2 review-dependent quantization selections |
| License and provenance boundaries | pass | `LICENSE`, `THIRD_PARTY_NOTICES.md`, `docs/PROVENANCE.md`; weights, derivatives, and outputs remain research/noncommercial |

## Remaining completion action

Export `manual_reviews.json` from each BF16, 8-bit, and 4-bit listening page,
then run `scripts/finalize_release_evidence.zsh` with those three immutable
review documents. The finalizer binds their hashes and model revision, performs
the two quantization selections, renders final reports, and invokes the
fail-closed bundle verifier. Only a passing final verifier result satisfies the
approved completion definition.
