# 8-bit MLX Inference Speed Design

## Status

Approved design for implementation. This specification covers runtime optimization of the
verified `sensitive-bf16` 8-bit Breeze TTS 2 artifact on Apple Silicon. It does not change,
replace, or requantize the published model weights.

## Baseline and acceptance target

The verified M3 Max baseline is:

- steady-state RTF: `2.713`
- long-text RTF: `2.596`
- representative voice-design RTF with CFG 4: approximately `5.8-5.9`

The optimized runtime must achieve, on the same M3 Max and 8-bit checkpoint:

- steady-state RTF at or below `2.0`
- voice-design RTF with CFG 4 at or below `4.0`
- no more than 10% regression in TTFA or peak memory without an explicit evidence note
- no regression in the existing quality, parity, waveform, interface, streaming, event,
  cancellation, reproducibility, and feature-completeness gates

RTF measurements exclude model loading, state whether reference preprocessing and WAV writing
are included, and report both cold and warmed runs. Final acceptance uses warmed generation and
the existing benchmark case definitions.

## Scope

### Included

1. Stage-level profiling for text/reference encoding, backbone prefill, backbone decode, depth
   decode, codec decode, synchronization, and WAV writing.
2. Removal of unnecessary host synchronization from the 15 dependent depth-codebook samples.
3. MLX compilation and prewarming of stable decode shapes when correctness is preserved.
4. If required to meet the target, preallocated KV caches and a batched CFG decode path.
5. Unit, integration, real-device performance, objective-quality, parity, and listening evidence.

### Excluded

- changing the Breeze TTS 2 architecture or tokenizer
- changing the public checkpoint format
- publishing the previously rejected full 8-bit candidate as the default
- weakening parity or quality thresholds to claim a speed improvement
- dropping dynamic text length, voice cloning, voice direction, events, streaming, or cancel/reset
  behavior

## Selected approach

Optimize the verified sensitive-BF16 runtime in two gated phases. Phase 1 keeps the current
cache and public API while changing the internal token representation and compiling only proven
stable functions. Phase 2 is entered only if Phase 1 misses either RTF target.

The rejected full 8-bit artifact remains useful as a performance reference (`1.422` steady-state
RTF), but it is not part of the production solution because its PyTorch parity gate failed.

## Architecture and data flow

### Profiler

Add a lightweight internal timing collector around these stages:

1. text encoder and projection
2. reference codec encode, when cloning or directing
3. conditional and optional unconditional prompt construction
4. backbone prefill
5. per-frame backbone decode
6. per-frame depth decode
7. waveform codec decode
8. output materialization and WAV writing

Profiling is opt-in and must not add synchronization to normal inference. Machine-readable timing
records are emitted by the benchmark path and can be aggregated across cold and warmed runs.

### Phase 1: device-resident depth tokens

The current depth loop converts every sampled token to a Python integer through `.item()`. The
new path keeps all 15 dependent depth tokens as scalar MLX arrays. Each next depth step consumes
the device-resident token array directly, and the completed frame is stacked on device.

Only the first codebook token is materialized on the host to decide whether generation reached
EOS. Repetition-penalty state for the first codebook remains semantically identical. Random-number
calls, masking, top-k/top-p behavior, and token order remain unchanged.

Stable sampling and depth-forward functions may use `mx.compile`. Compiled functions are cached
by the normal/CFG shape and prewarmed before timed steady-state measurements. Compilation must not
capture mutable streaming state or make random state nondeterministic.

### Phase 2: static decode fast path

If Phase 1 misses a target, add an internal fast-path object that owns preallocated KV storage and
the compiled decode functions. The public generator continues to expose the same results and
cleanup behavior.

Conditional and unconditional CFG work may be combined into a batch only when both branches retain
the exact official formula and independent cache contents. Unsupported shapes or modes use the
existing eager path. Streaming cancellation always resets codec and fast-path state in `finally`.

## Compatibility and fallback

The Python API, CLI arguments, HTTP schema, generated WAV format, and checkpoint layout remain
unchanged. During validation, the optimized path is controlled by an internal feature flag. Once
all gates pass it becomes the default, while an explicit eager/debug override remains available.

Compilation failure, unsupported hardware, unsupported shape, or cache allocation failure falls
back to eager execution before audio generation starts. Mid-generation failures are surfaced; the
runtime must not silently restart with a different random sequence or concatenate outputs from two
paths.

## Verification

### Correctness tests

- token masking, sampling controls, EOS, repetition penalty, and deterministic seed coverage
- device-resident depth-token shape and dtype coverage
- eager/fast token and WAV exact equality for deterministic fixtures
- normal and CFG generation
- streaming chunk order, final marker, cancellation, and state reset
- cloning, direction, long text, English/Chinese events, CLI, HTTP, and checkpoint isolation

If compiled real-model output is not bit-exact due only to legal floating-point reordering, it must
still pass the existing CER, speaker similarity, leakage, waveform, event-listening, and PyTorch
parity gates. The difference and its cause must be recorded rather than treated as exact parity.

### Performance protocol

Run the same 8-bit revision on the same M3 Max with no competing inference workload. Record:

- model revision and runtime commit
- macOS, Python, MLX, memory, and power-mode metadata
- one cold run followed by at least five warmed runs
- median and p90 RTF, TTFA, and peak memory
- stage timing percentages
- ordinary cloning/no-CFG, voice design/CFG 4, long text, and streaming cases

Acceptance uses the median warmed RTF. Raw JSON, WAV files, and rendered reports are retained under
an optimization-specific evidence directory.

## Implementation order

1. Add profiler and reproduce the baseline.
2. Implement device-resident depth tokens with eager execution.
3. Verify deterministic fixtures and real-model quality.
4. Add narrowly scoped compilation and prewarming.
5. Measure against both targets.
6. Only if required, implement preallocated KV cache and batched CFG.
7. Run the full test suite and real-device evidence pipeline.
8. Promote the fast path to default only after every gate passes.

## Risks and controls

- **Random sequence drift:** compare token streams before audio metrics; preserve sampling call order.
- **Compilation shape explosion:** compile only bounded decode shapes and expose cache statistics in
  profiling evidence.
- **Memory regression:** measure load and run peaks; reject unexplained regressions above 10%.
- **Streaming state leaks:** retain fail-safe reset and add repeated cancel/restart tests.
- **Benchmark optimism:** keep cold and warmed results separate and do not include compile time in a
  warmed result without also reporting it.
- **Quality-speed tradeoff:** do not replace the verified sensitive-BF16 weights or relax gates.

## Deliverables

- optimized inference implementation and focused tests
- stage profiler and raw timing evidence
- before/after M3 Max comparison for 8-bit
- refreshed 8-bit release report and completion audit
- README documentation of fast/eager behavior and measurement scope

