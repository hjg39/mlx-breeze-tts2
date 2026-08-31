# PyTorch-to-MLX parity evidence

The completion gate requires paired snapshots from the same pinned model revision
and fixed test case. A snapshot uses `schema_version: 1` and contains:

- `model_id`, `model_revision`, and `case_id`;
- `template_render`, `token_ids`, and `reference_audio_codes`;
- `masks`, `weight_shapes`, and `deterministic_tokens`;
- `intermediate_tensors`, represented as JSON numeric arrays for the fixed small
  input.

Capture one snapshot from the official PyTorch runtime and one from this MLX
runtime with identical inputs. Do not reuse a snapshot across revisions. Compare
them with explicit numeric tolerances:

```bash
PYTHONPATH=src python scripts/capture_pytorch_parity.py \
  --official-repo /path/to/clean/breeze-tts \
  --model models/breeze-bf16 \
  --ref-audio /path/to/reference.wav --ref-text "Exact transcript." \
  --text "Target sentence." --instruction "Speak calmly." \
  --device cpu --output reports/parity/pytorch.json

PYTHONPATH=src python scripts/capture_mlx_parity.py \
  --model models/breeze-bf16 \
  --ref-audio /path/to/reference.wav --ref-text "Exact transcript." \
  --text "Target sentence." --instruction "Speak calmly." \
  --output reports/parity/mlx.json
```

The PyTorch capture refuses a dirty source checkout and records its exact git
commit. Both captures use the official `ref_edit_tata` path and preserve full
input IDs, reference codec codes, masks, canonical weight shapes, deterministic
argmax, and small prompt/backbone/logit slices.

On the project machine, prepare the isolated official environment and run a
paired capture without mixing PyTorch dependencies into the MLX package:

```bash
./scripts/setup_official_parity_env.zsh
./scripts/run_parity_capture.zsh models/breeze-bf16 bf16
```

The setup report at `reports/parity/official_environment.json` records the clean
source commit, Python version, and complete installed distribution set. The
capture script refuses to overwrite existing evidence. Quantized candidates
must use distinct labels and tolerances justified by their numeric format; an
intentionally changed packed weight shape or deterministic token remains a
failure rather than being hidden by a looser floating-point tolerance.

```bash
mlx-breeze-tts2 compare-parity \
  reports/parity/pytorch.json reports/parity/mlx.json \
  --atol 1e-4 --rtol 1e-3 \
  --output reports/parity/comparison.json
```

Exact sections must match byte-for-byte after JSON decoding. Every intermediate
tensor must have the same name and shape, contain finite values, and pass
`numpy.allclose` at the recorded tolerances. The report stores both input hashes.

Merge only a passing report into a generated benchmark summary. This is
non-destructive and records the evidence file hash:

```bash
mlx-breeze-tts2 apply-parity-evidence \
  reports/full_matrix/bf16/summary.metrics.json \
  reports/parity/comparison.json \
  --output reports/full_matrix/bf16/summary.parity.json
```

`verify-evidence` independently reopens the comparison report, checks its
SHA-256 and all required check names, and rejects missing, changed, or failing
evidence. Real paired snapshots remain `pending` until both runtimes can execute.
