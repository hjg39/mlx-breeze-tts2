# Standalone implementation handoff

The standalone runtime, conversion/audit path, Python engine, CLI, benchmark
runner, and upstream-compatible FastAPI endpoint are implemented in commits
`cf958c0`, `e998d4c`, and `573ad7d`.

Verification status:

- `pass`: Ruff, compileall, package isolation, wheel/sdist build.
- `pass`: 9 Metal-independent tests, including HTTP PCM headers, input errors,
  409 concurrency, and upload cleanup.
- `pass on precursor`: five real M3 Max 4-bit capability samples, Whisper
  content checks, clipping checks, and ECAPA clone similarity. See
  `reports/baseline/`.
- `pending`: two standalone MLX test modules and fresh model inference because
  this execution sandbox cannot initialize Metal.
- `pending`: official BF16 and derived 8/4-bit matrix. Only about 11.98 GiB is
  free; source plus BF16 output and conversion headroom require at least 20 GiB.
- `missing evidence`: paired upstream PyTorch parity and manual event listening.

No pending item has been represented as complete.
