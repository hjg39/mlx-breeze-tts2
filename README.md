# MLX Breeze TTS 2

Standalone Apple Silicon port of Breeze TTS 2. The production package vendors
the required MLX transformer and Qwen3 speech-codec primitives and does **not**
depend on `mlx-audio` at runtime.

Status: the isolated source, generation API, CLI, conversion/audit surface, and
upstream-compatible HTTP endpoint are implemented. Existing 4-bit baseline
inference passed on an Apple M3 Max; a fresh standalone Metal run, official BF16
conversion, 8-bit run, PyTorch parity, and manual event listening remain
`pending`.

The current automated handoff is `77 passed, 2 Metal-dependent modules skipped`;
wheel and source distributions build successfully with
`uv build --no-build-isolation`.

## Capabilities

- Voice design from text plus instruction
- Zero-shot voice cloning from one reference WAV and exact transcript
- Voice direction by combining a reference pair and an instruction
- English `(laugh)` / `(sigh)` and Chinese `[笑]` / `[叹气]` event syntax
- Non-streaming and incremental generation
- Temperature, top-p, top-k, repetition penalty, CFG, seed, and token limits
- Python API, command line, and upstream-compatible FastAPI endpoint
- Official BF16 source conversion plus affine 8-bit and 4-bit outputs
- Strict checkpoint key audit
- Fail-fast validation for every generation control and reference-audio input
- Empty/non-finite codec output rejection and streaming-state cleanup

## Install

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[dev,server]'
```

The default model is the locally tested unofficial 4-bit conversion
`LunaFox/Breeze-TTS-2-mlx-4bit`. Pin a revision for reproducibility:

```bash
mlx-breeze-tts2 generate \
  --model LunaFox/Breeze-TTS-2-mlx-4bit \
  --text "Welcome aboard. Your journey begins now." \
  --instruction "A warm, thoughtful young woman with a clear voice." \
  --cfg-scale 4 \
  --seed 7 \
  --output outputs/design.wav
```

Voice cloning and direction:

```bash
mlx-breeze-tts2 generate \
  --ref-audio reference.wav \
  --ref-text "This is the exact transcript of the reference audio." \
  --text "(sigh) It is good to hear your voice again." \
  --instruction "Speak slowly with a restrained, serious tone." \
  --stream --streaming-interval 1 \
  --output outputs/directed.wav
```

Python:

```python
from mlx_breeze_tts import load, write_audio

model = load("LunaFox/Breeze-TTS-2-mlx-4bit")
result = next(model.generate(
    text="[笑] 欢迎来到今晚的故事时间。",
    instruct="一位温柔自信的年轻女性，声音清晰。",
    cfg_scale=4,
    seed=7,
))
write_audio("output.wav", result.audio, result.sample_rate)
```

## Conversion

The converter is fail-closed: source weights must load strictly and the saved
artifact must pass a second key audit.

When disk space cannot safely hold both the 7.68 GB official snapshot and a
second BF16 copy, create a same-volume hard-linked BF16 candidate. This keeps
the immutable safetensors blocks shared while copying mutable metadata. It is
only a candidate until the Metal-dependent strict audit passes:

```bash
mlx-breeze-tts2 link-bf16 \
  --source /absolute/path/to/pinned-official-snapshot \
  --output models/breeze-bf16
mlx-breeze-tts2 inspect-checkpoint models/breeze-bf16
mlx-breeze-tts2 audit models/breeze-bf16
```

`link-bf16` is local-only, rejects quantized or non-BF16 sources, verifies the
safetensors index without importing MLX, records hashes, and fails if source
and output are on different filesystems.

```bash
mlx-breeze-tts2 convert \
  --source BreezeBlue/Breeze-TTS-2 \
  --revision c1c8ca18b70b30822735633991d9ebf4898e47d4 \
  --dtype bfloat16 \
  --output models/breeze-bf16

mlx-breeze-tts2 convert --source models/breeze-bf16 --bits 8 --output models/breeze-8bit
mlx-breeze-tts2 convert --source models/breeze-bf16 --bits 4 --output models/breeze-4bit
mlx-breeze-tts2 audit models/breeze-4bit
```

Quantized conversion supports reproducible `full` and `sensitive-bf16`
policies. Exact exclusions, artifact metadata, static consistency checks, and
the required real-device ablation are documented in
[`docs/QUANTIZATION.md`](docs/QUANTIZATION.md).

The completion verifier is intentionally fail-closed. Point it at a directory
containing `bf16/`, `8bit/`, and `4bit/` evidence bundles:

```bash
mlx-breeze-tts2 verify-evidence reports/full_matrix \
  --output reports/full_matrix/completion_audit.json
```

It exits non-zero until strict artifact audits, the complete capability/event
matrix, CER/speaker/leakage thresholds, interfaces, reproducibility, WAVs,
Markdown, and listening HTML are all present and passing. It also requires
finite load/RTF/TTFA/memory measurements, PyTorch parity, and waveform-integrity
evidence covering non-finite output, severe clipping, repeated tails, and stream
boundary discontinuities. Every variant summary must also contain the immutable
40-character resolved model revision and structured checkpoint provenance.

Run the complete 23-case matrix with separate exact English and Chinese
reference pairs. It covers both languages for design/clone/direction,
cross-language cloning, eight vocal events, non-streaming, streaming cancel and
state reset, long text, steady state, every sampling control, and fixed-seed
reproducibility. The CLI invocation also probes the real FastAPI contract:

```bash
mlx-breeze-tts2 benchmark \
  --model models/breeze-4bit \
  --ref-audio-en references/english.wav \
  --ref-text-en "Exact English transcript." \
  --ref-audio-zh references/chinese.wav \
  --ref-text-zh "精确的中文转写。" \
  --output reports/full_matrix/4bit
```

Missing reference pairs are retained as `missing_input` cases, never silently
removed. Automated generation does not mark vocal events audible; reviewers
must record those eight verdicts after listening to `index.html`. Use its
**Export manual_reviews.json** button.

Generate the objective-metric worksheet, fill `asr_text` plus ECAPA
`speaker_cosine`/`speaker_p10`, and merge it without overwriting the raw report.
CER and reference leakage are computed deterministically during the merge:

```bash
mlx-breeze-tts2 objective-template \
  reports/full_matrix/4bit/summary.json \
  --output reports/full_matrix/4bit/objective_metrics.json
mlx-breeze-tts2 apply-objective-metrics \
  reports/full_matrix/4bit/summary.json \
  reports/full_matrix/4bit/objective_metrics.json \
  --output reports/full_matrix/4bit/summary.metrics.json
```

Finally merge listening verdicts into the metrics-bearing summary:

```bash
mlx-breeze-tts2 apply-listening-review \
  reports/full_matrix/4bit/summary.metrics.json manual_reviews.json \
  --output reports/full_matrix/4bit/summary.final.json
mlx-breeze-tts2 render-report \
  reports/full_matrix/4bit/summary.final.json \
  --output reports/full_matrix/4bit
```

`verify-evidence` prefers `summary.final.json`, then reviewed/metrics summaries,
while retaining the untouched generation summary as provenance. Missing ASR,
speaker, leakage, or listening values remain pending and fail closed.

PyTorch parity uses paired revision-pinned JSON snapshots and a hash-verified,
non-destructive merge. The required fields, tolerances, and commands are in
[`docs/PYTORCH_PARITY.md`](docs/PYTORCH_PARITY.md).

The pinned local English/Chinese reference paths, exact transcripts, observed
audio properties, hashes, and usage boundary are recorded in
`reports/full_matrix/input_manifest.json`.

```bash
mlx-breeze-tts2 validate-inputs reports/full_matrix/input_manifest.json \
  --output reports/full_matrix/input_validation.json
```

## HTTP compatibility

```bash
mlx-breeze-tts2 serve --host 127.0.0.1 --port 8000
curl -X POST http://127.0.0.1:8000/v1/audio/speech \
  -F 'text=Hello from MLX.' \
  -F 'instruction=A calm and clear voice.' \
  -F 'cfg_scale=4' \
  --output output.pcm
```

The response is mono 24 kHz little-endian PCM16 and includes
`X-Sample-Rate: 24000`, `X-Sample-Format: s16le`, and `Cache-Control: no-store`.
PCM chunks are emitted incrementally as inference advances. Only one synthesis
request runs at a time; overlapping requests receive HTTP 409. Reference audio
and transcript must be supplied together.

## Evidence and license

- X/GitHub selection research: `docs/research/2026-08-31-hot-tts-selection.md`
- Approved implementation spec: `docs/design/2026-08-31-approved-spec.md`
- Baseline machine report and WAVs: `reports/baseline/`
- Provenance and evidence status: `docs/PROVENANCE.md`

Project source is MIT licensed. Breeze model weights, derivatives, and generated
outputs are governed by the BreezeBlue Research and Non-Commercial License.
Review the upstream model card before conversion or use. No model weights are
included in this repository.
