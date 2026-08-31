# Quantization policy and ablation

Both 8-bit and 4-bit artifacts must be derived from the same verified BF16
artifact. The converter exposes two deterministic policy candidates:

- `full`: quantize every eligible affine/embedding module whose input width is
  divisible by the group size. This is closest to the existing LunaFox 4-bit
  reference, which has 468 modules with stored scales.
- `sensitive-bf16` policy v2: keep the complete text encoder and depth decoder,
  text/audio embeddings, projection bridges, and `lm_head` in BF16 while
  quantizing the main generation backbone. Policy v1 retained only six bridge
  modules and failed the real-device content gate; its artifacts and reports
  remain historical evidence rather than release candidates.

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

Candidate state is tri-valued. A measured threshold or hash-verifiable parity
failure is `fail`; absent listening, performance, interface, or parity evidence
is `pending`; only complete passing evidence is `pass`. The comparator never
selects another candidate merely because one candidate is pending. A terminal
failure may stop further evaluation only when its source report and paired
snapshots are present and their SHA-256 hashes match. This allows the current
full-quantization candidates to stop on their exact parity failures without
requiring unnecessary event listening, while the sensitive-BF16 candidates
remain pending until their human event reviews are exported.

Selection remains `pending` until those measurements exist on the local
machine. Do not infer the winner from static size or the precursor LunaFox
checkpoint alone.

Content quality uses corpus CER over standard-content cases. Per-case CER and
its maximum remain in each report for diagnosis; non-verbal event cases use the
separate listening gate, and exploratory cross-language cases are reported but
do not change the standard-content corpus denominator.
