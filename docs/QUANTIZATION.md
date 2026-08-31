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
audio-tokenizer directory is copied unchanged and remains BF16 under both
policies.

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

Run the same 23-case benchmark and objective/listening workflow on both
candidates. Selection remains `pending` until CER, ECAPA cosine/P10, leakage,
events, waveform integrity, RTF, TTFA, memory, and artifact size have been
compared on the local machine. Do not infer the winning policy from static size
or the precursor LunaFox checkpoint alone.
