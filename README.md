# MLX Breeze TTS 2

Standalone Apple Silicon port of Breeze TTS 2. The production package vendors
the required MLX transformer and Qwen3 speech-codec primitives and does **not**
depend on `mlx-audio` at runtime.

Status: the approved standalone port is complete. Official BF16 plus policy-v2
8-bit/4-bit artifacts passed the real M3 Max generation, objective-quality,
waveform, interface, performance, precision-specific PyTorch parity, and human
event-listening gates. The independent final verifier reports zero issues in
`reports/release-v2/completion_audit.json`.

The current automated handoff is `132 passed` on the project machine;
wheel and source distributions build successfully with
`uv build --no-build-isolation`.

## Capabilities

- Voice design from text plus instruction
- Zero-shot voice cloning from one reference WAV and exact transcript
- Voice direction by combining a reference pair and an instruction
- English `(laugh)` / `(cough)` / `(clears throat)` / `(sigh)` and Chinese
  `[笑]` / `[咳嗽]` / `[清嗓子]` / `[叹气]` event syntax
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

The recommended model is the verified 4-bit release
[`vanch007/Sirocco-MLX-4bit`](https://huggingface.co/vanch007/Sirocco-MLX-4bit).
The model repositories are gated: request access and accept the upstream
research/non-commercial terms before downloading.

| Precision | Hugging Face repository | Verified revision |
|---|---|---|
| BF16 | [`vanch007/Sirocco-MLX-BF16`](https://huggingface.co/vanch007/Sirocco-MLX-BF16) | `4c7eec64281272a32376234075e955dce50c9252` |
| 8-bit sensitive-BF16 | [`vanch007/Sirocco-MLX-8bit`](https://huggingface.co/vanch007/Sirocco-MLX-8bit) | `45c58f3a91ddee3c4ce90cd10ec86ddf97e933a0` |
| 4-bit sensitive-BF16 | [`vanch007/Sirocco-MLX-4bit`](https://huggingface.co/vanch007/Sirocco-MLX-4bit) | `0c4f095035f82e06574c41cdd1212ebada404638` |

Generate with the current 4-bit release from the command line:

```bash
mlx-breeze-tts2 generate \
  --model vanch007/Sirocco-MLX-4bit \
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

The Python API can pin the verified revision for reproducibility:

```python
from mlx_breeze_tts import load, write_audio

model = load(
    "vanch007/Sirocco-MLX-4bit",
    revision="0c4f095035f82e06574c41cdd1212ebada404638",
)
result = next(
    model.generate(
        text="[笑] 欢迎来到今晚的故事时间。",
        instruct="一位温柔自信的年轻女性，声音清晰。",
        cfg_scale=4,
        seed=7,
    )
)
write_audio("output.wav", result.audio, result.sample_rate)
```

## Experimental 8-bit fast depth path

The 8-bit runtime includes an opt-in incremental depth-decoder path. It keeps
dependent codebook tokens on the MLX device and uses a short per-frame KV cache
instead of recomputing the complete depth prefix. Until real-device quality and
performance gates pass, the compatible eager path remains the default.

Use `--fast-depth` for a candidate generation or `--no-fast-depth` to select the
eager reference explicitly. Compare both paths with cold/warm scope and fixed-seed
hashes recorded in JSON:

```bash
PYTHONPATH=src .venv/bin/python scripts/benchmark_8bit_speed.py \
  --model models/breeze-8bit-sensitive-bf16-v2 \
  --output reports/optimization/8bit-fast-depth \
  --runs 5
```

`speed.json` records exactly one model-cold run, one first-use prewarm for every
path/case, five warmed acceptance runs by default, median and p90 RTF, peak
memory, output materialization and WAV-write time, plus a separate
synchronization-based stage profile. The profiled run is excluded from speed
acceptance so its diagnostic barriers cannot make the RTF result look faster or
slower.

The candidate is not considered accepted until the report reaches steady-state
RTF `<= 2.0`, CFG-4 RTF `<= 4.0`, and the existing parity, objective-quality,
waveform, streaming, interface, event, and listening gates remain passing.
After the focused speed comparison passes, rerun the complete 23-case matrix
with `benchmark --fast-depth`; its `summary.json` records the selected runtime
path under `runtime_options.fast_depth`. The default matrix remains eager until
the candidate clears every gate.

## Conversion

The converter is fail-closed: source weights must load strictly and the saved
artifact must pass a second key audit.

Download the approved official revision into the shared Hugging Face cache.
The command verifies the resolved 40-character revision and only writes its
report after the required config and safetensors files are complete:

```bash
mlx-breeze-tts2 download-official \
  --output-report reports/official_download.json
```

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

On this machine, the resumable real-device pipeline performs the pinned
download, Metal preflight, linked BF16 preparation, both full/sensitive 8-bit
and 4-bit conversions, all 23 generation cases, and automatic ASR/ECAPA merge:

```bash
cd /Users/vanch/mlx-breeze-tts2
./scripts/run_real_device_pipeline.zsh
```

It never removes artifacts and refuses incomplete existing model or evidence
directories. Completed stages are validated and skipped on rerun. The current
paired parity evidence is under `reports/parity/`; eight-event manual listening
remains the final explicit completion gate.

It exits non-zero until strict artifact audits, the complete capability/event
matrix, CER/speaker/leakage thresholds, interfaces, reproducibility, WAVs,
Markdown, and listening HTML are all present and passing. It also requires
finite load/RTF/TTFA/memory measurements, PyTorch parity, and waveform-integrity
evidence covering non-finite output, severe clipping, repeated tails, and stream
boundary discontinuities. Every variant summary must also contain the immutable
40-character resolved model revision and structured checkpoint provenance.
Quantized variants additionally require a hash-verified full-versus-sensitive
policy ablation tied to the same precision and revision.

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
must record those eight verdicts. Use the focused `events.html` page and its
**Export manual_reviews.json** button; it contains only the required events and
downloads a precision-labelled file. `index.html` remains the complete 23-case
inspection page.

Generate the objective-metric worksheet, fill it automatically with the cached
MLX Whisper large-v3-turbo and segment-aware SpeechBrain ECAPA backend, then
merge it without overwriting the raw report. The evaluator records audio/model
hashes, resolved Whisper revision, actual speaker device, and returns non-zero
when any required metric is missing. CER and reference leakage are computed
deterministically during the merge:

```bash
mlx-breeze-tts2 objective-template \
  reports/full_matrix/4bit/summary.json \
  --output reports/full_matrix/4bit/objective_metrics.json
mlx-breeze-tts2 evaluate-objective-metrics \
  reports/full_matrix/4bit/objective_metrics.json \
  --output reports/full_matrix/4bit/objective_metrics.evaluated.json
mlx-breeze-tts2 apply-objective-metrics \
  reports/full_matrix/4bit/summary.json \
  reports/full_matrix/4bit/objective_metrics.evaluated.json \
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

`verify-evidence` prefers final, reviewed, parity, then metrics summaries while
retaining the untouched generation summary as provenance. Missing ASR,
speaker, leakage, or listening values remain pending and fail closed. Content
uses corpus CER over standard cases; per-case and maximum CER remain diagnostic.

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

For the checked-in `release-v2` bundle, export one review from each BF16,
8-bit, and 4-bit `events.html` page, then run the fail-closed finalizer. It copies
the original review documents into their variant directories, records their
SHA-256 hashes, performs both quantization selections, renders final reports,
and runs the independent bundle verifier:

```bash
./scripts/finalize_release_evidence.zsh \
  /path/to/bf16-manual_reviews.json \
  /path/to/8bit-manual_reviews.json \
  /path/to/4bit-manual_reviews.json
```

Before writing anything, the finalizer validates all three documents against
their immutable model revision and requires an explicit `audible` or `missing`
verdict for every event. It refuses to overwrite any existing review, reviewed
summary, selection report, or final summary. Quantization comparison treats
missing evidence as `pending`, never as a reason to reject a candidate. The
full-quantization candidates have separate hash-bound terminal parity failures;
the sensitive-BF16 candidates still require the exported listening verdicts.

If a benchmark predates an HTTP fix, capture and merge a revision-bound real
probe without rerunning the 23 audio cases:

```bash
mlx-breeze-tts2 probe-http --model models/breeze-4bit \
  --output reports/full_matrix/4bit/http_evidence.json
mlx-breeze-tts2 apply-http-evidence \
  reports/full_matrix/4bit/summary.parity.json \
  reports/full_matrix/4bit/http_evidence.json \
  --output reports/full_matrix/4bit/summary.http.json
```

The server owns a single dedicated inference executor and initializes MLX
CPU/GPU streams on that worker. Generator creation, iteration, PCM conversion,
and cleanup remain on the same thread while the event loop can return HTTP 409
for concurrent requests.

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
- Requirement-by-requirement completion audit: `docs/COMPLETION_AUDIT.md`
- Baseline machine report and WAVs: `reports/baseline/`
- Provenance and evidence status: `docs/PROVENANCE.md`

Project source is MIT licensed. Breeze model weights, derivatives, and generated
outputs are governed by the BreezeBlue Research and Non-Commercial License.
Review the upstream model card before conversion or use. No model weights are
included in this repository.
