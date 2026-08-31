#!/bin/zsh
set -euo pipefail

ROOT=${0:A:h:h}
PYTHON_BIN=${PYTHON_BIN:-/Users/vanch/mlx-qwen3-tts/.venv/bin/python}
export PYTHONPATH="$ROOT/src"

cli() {
  "$PYTHON_BIN" -m mlx_breeze_tts.cli "$@"
}

require_metal() {
  "$PYTHON_BIN" - <<'PY'
import mlx.core as mx
x = mx.array([1.0])
mx.eval(x)
print("Metal preflight: pass", x.tolist())
PY
}

json_value() {
  "$PYTHON_BIN" - "$1" "$2" <<'PY'
import json
import sys
value = json.load(open(sys.argv[1]))
for key in sys.argv[2].split("."):
    value = value[key]
print(value)
PY
}

assert_json_value() {
  local file=$1
  local field=$2
  local expected=$3
  local actual
  actual=$(json_value "$file" "$field")
  if [[ "$actual" != "$expected" ]]; then
    print -u2 "Refusing incomplete evidence: $file ($path=$actual, expected $expected)"
    return 1
  fi
}

ensure_model() {
  local destination=$1
  shift
  if [[ -f "$destination/config.json" ]]; then
    cli inspect-checkpoint "$destination"
    cli audit "$destination"
    return
  fi
  if [[ -e "$destination" ]]; then
    print -u2 "Refusing incomplete existing model path: $destination"
    return 1
  fi
  cli "$@" --output "$destination"
}

evaluate_variant() {
  local model=$1
  local report_dir=$2
  mkdir -p "$report_dir"
  if [[ -f "$report_dir/summary.json" ]]; then
    assert_json_value "$report_dir/summary.json" status audio_generated_evaluation_pending
  else
    cli benchmark \
      --model "$model" \
      --ref-audio-en "$REF_AUDIO_EN" \
      --ref-text-en "$REF_TEXT_EN" \
      --ref-audio-zh "$REF_AUDIO_ZH" \
      --ref-text-zh "$REF_TEXT_ZH" \
      --output "$report_dir"
  fi
  if [[ -f "$report_dir/objective_metrics.evaluated.json" ]]; then
    assert_json_value "$report_dir/objective_metrics.evaluated.json" evaluation.status complete
  else
    cli evaluate-objective-metrics \
      "$report_dir/objective_metrics.json" \
      --output "$report_dir/objective_metrics.evaluated.json"
  fi
  if [[ -f "$report_dir/summary.metrics.json" ]]; then
    assert_json_value "$report_dir/summary.metrics.json" validation.objective_metrics complete
  else
    cli apply-objective-metrics \
      "$report_dir/summary.json" \
      "$report_dir/objective_metrics.evaluated.json" \
      --output "$report_dir/summary.metrics.json"
  fi
  cli render-report "$report_dir/summary.metrics.json" --output "$report_dir"
}

cd "$ROOT"
require_metal
cli validate-inputs reports/full_matrix/input_manifest.json \
  --output reports/full_matrix/input_validation.json
cli download-official --output-report reports/official_download.json

OFFICIAL_SNAPSHOT=$(json_value reports/official_download.json snapshot)
REF_AUDIO_EN=$(json_value reports/full_matrix/input_manifest.json references.en.path)
REF_TEXT_EN=$(json_value reports/full_matrix/input_manifest.json references.en.transcript)
REF_AUDIO_ZH=$(json_value reports/full_matrix/input_manifest.json references.zh.path)
REF_TEXT_ZH=$(json_value reports/full_matrix/input_manifest.json references.zh.transcript)

ensure_model models/breeze-bf16 link-bf16 --source "$OFFICIAL_SNAPSHOT"
evaluate_variant models/breeze-bf16 reports/full_matrix/bf16

for bits in 8 4; do
  for policy in full sensitive-bf16; do
    model="models/breeze-${bits}bit-${policy}"
    report="reports/ablation/${bits}bit-${policy}"
    ensure_model "$model" convert \
      --source models/breeze-bf16 \
      --bits "$bits" \
      --quantization-policy "$policy"
    evaluate_variant "$model" "$report"
  done
done

print "Automated Metal generation and objective metrics are ready."
print "Next gates: PyTorch parity, both 8/4-bit policy comparisons, and eight-event manual listening."
