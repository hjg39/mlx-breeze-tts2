# 8-bit MLX Fast Depth Results

## Status

Automated acceptance is `pass`; release/default promotion is `pending` only on
the eight-event listening review for newly generated fast-path audio.

- Runtime commit: `e40bbfbf55675da56945e82178c6a81c5bb62cce`
- Model: `models/breeze-8bit-sensitive-bf16-v2`
- Upstream revision: `c1c8ca18b70b30822735633991d9ebf4898e47d4`
- MLX: `0.32.2`
- Host: Apple M3 Max, 40-core GPU, 128 GB unified memory

## Implementation

- Keeps dependent codebook tokens as MLX arrays and submits device work with
  `mx.async_eval` instead of synchronizing every depth token to the host.
- Uses an incremental KV cache for the 15 dependent codebooks.
- Compiles the complete fixed-width depth frame as one graph and uses an
  exact-size cache rather than padded general-purpose cache scatter updates.
- Batches depth conditional/unconditional CFG branches while preserving the
  official `uncond + scale * (cond - uncond)` formula.
- Lazily creates the compiled function after checkpoint loading.
- On a first-frame compile/allocation failure, restores the exact MLX RNG state
  and falls back to eager decoding. Failures after any generated frame surface
  immediately instead of mixing execution paths.

## Performance evidence

Authoritative final run:
`reports/optimization/8bit-fast-depth/run-e40bbfb-fast/`.

| Measurement | Median | P90 | Target | Result |
|---|---:|---:|---:|---|
| Ordinary steady-state RTF | 1.026 | 1.027 | <= 2.0 | pass |
| Voice-design CFG=4 RTF | 1.321 | 1.323 | <= 4.0 | pass |
| Eager streaming TTFA | 2.314 s | n/a | reference | pass |
| Fast streaming TTFA | 1.106 s | n/a | <= 10% regression | pass (-52.2%) |

Both five-run RTF groups and both five-run TTFA groups are fixed-seed
reproducible. Focused peak memory is `8.683 GB` for ordinary generation and
`8.334 GB` for CFG=4.

The initial sequential compare runs are retained as diagnostic evidence. They
showed severe thermal-order bias after many eager generations, including large
late-run outliers. Acceptance therefore uses the fast-only five-run protocol;
its TTFA check still alternates eager and fast by pair on the same loaded model.
The full matrix peak was `10.820 GB`, versus `10.954 GB` in the prior verified
8-bit release matrix, so peak memory did not regress.

## Functional and quality evidence

Full matrix directory:
`reports/optimization/8bit-fast-depth/full-matrix-2d698df/`.

- 23/23 cases generated on Metal; matrix coverage has no missing or duplicate
  required capability.
- Python, CLI, HTTP, non-streaming, streaming, cancellation, sampling control,
  deterministic seed, and waveform-integrity checks: `pass`.
- Objective metrics: max CER `0.030303`, corpus CER `0.001488`, clone cosine
  minimum `0.641503`, clone P10 minimum `0.623924`, leakage maximum `0.169492`.
- Objective backend: `mlx-community/whisper-large-v3-turbo` revision
  `a4aaeec0636e6fef84abdcbe3544cb2bf7e9f6fb` plus SpeechBrain ECAPA.
- Precision-specific parity:
  `reports/parity/comparison-8bit-fast-2d698df.json`, status `pass`, `atol=0.05`,
  `rtol=0.05`. Template, IDs, reference codec codes, masks, shapes,
  deterministic tokens, and all intermediate tensors pass.
- Unit/integration suite: `141 passed`; focused Ruff fatal checks and formatting:
  `pass`.

Fast and eager real-model waveforms are not bit-exact because cached attention
changes legal floating-point evaluation order and therefore stochastic token
selection. The fast output is separately deterministic and has passed the full
objective, waveform, interface, feature, and PyTorch parity gates. The remaining
manual gate must use the new `events.html`; prior eager listening evidence is not
reused.

## Remaining gate

Open
`reports/optimization/8bit-fast-depth/full-matrix-2d698df/events.html`, review
all eight English/Chinese laugh, cough, throat-clear, and sigh samples, export
the revision-bound review JSON, and apply it to `summary.parity.json`. Only then
may `fast_depth` become the default across Python, CLI, server, and benchmark
surfaces.
