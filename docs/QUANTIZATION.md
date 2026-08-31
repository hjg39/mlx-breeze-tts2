# Quantization policy and ablation

Both 8-bit and 4-bit artifacts must be derived from the same verified BF16
artifact. The converter exposes two deterministic policy candidates:

- `full`: quantize every eligible affine/embedding module whose input width is
  divisible by the group size. This is closest to the existing LunaFox 4-bit
  reference, which has 468 modules with stored scales.
- `sensitive-bf16`: keep the text/audio embeddings, projection bridges,
  `lm_head`, and depth input projections in BF16 while quantizing transformer
  attention and MLP modules.

Norms are not eligible for MLX affine quantization and remain BF16. The external
audio-tokenizer directory remains BF16 under both policies. Its immutable files
are hard-linked on the same APFS volume (with a copy fallback across volumes),
so the four ablation candidates do not duplicate the codec weights. Root
config and audit files are always regenerated and never linked.

Generate candidates explicitly:

```bash
mlx-breeze-tts2 convert --source models/breeze-bf16 --bits 4 \
  --quantization-policy full --output models/breeze-4bit-full
mlx-breeze-tts2 convert --source models/breeze-bf16 --bits 4 \
  --quantization-policy sensitive-bf16 --output models/breeze-4bit-sensitive
```

The artifact config records the policy version plus exact quantized and excluded
module lists. `inspect-checkpoint` independently reconstructs the module list
from safetensors `.scales` keys and fails if metadata differs. Conversion runs
both strict runtime and static audits before returning.

Run the same 23-case benchmark and objective/listening/PyTorch-parity workflow
on both candidates, then compare completed summaries:

```bash
mlx-breeze-tts2 compare-quantization \
  reports/ablation/4bit-full/summary.final.json \
  reports/ablation/4bit-sensitive/summary.final.json \
  --output reports/ablation/4bit-comparison.json

mlx-breeze-tts2 apply-quantization-evidence \
  reports/ablation/4bit-full/summary.final.json \
  reports/ablation/4bit-comparison.json \
  --output reports/full_matrix/4bit/summary.final.json
```

The comparator first applies every approved quality, interface, event,
waveform, parity, and performance gate. It selects the only passing candidate,
or—when both pass—the smaller artifact, then lower steady-state RTF, with the
sensitive policy as an exact tie-break. Inputs and the selected summary are
hashed. `verify-evidence` independently checks the report hash, precision,
revision, and selected policy.

Selection remains `pending` until those measurements exist on the local
machine. Do not infer the winner from static size or the precursor LunaFox
checkpoint alone.
