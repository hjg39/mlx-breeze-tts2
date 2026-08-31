#!/bin/zsh
set -euo pipefail

if (( $# < 2 || $# > 4 )); then
  print -u2 "usage: $0 MODEL LABEL [ATOL] [RTOL]"
  exit 2
fi
ROOT=${0:A:h:h}
MODEL=$1
LABEL=$2
ATOL=${3:-0.0001}
RTOL=${4:-0.001}
OFFICIAL_CHECKOUT=${OFFICIAL_CHECKOUT:-$ROOT/.parity/official-breeze-tts}
OFFICIAL_PYTHON=${OFFICIAL_PYTHON:-$ROOT/.parity/official-venv/bin/python}
MLX_PYTHON=${MLX_PYTHON:-/Users/vanch/mlx-qwen3-tts/.venv/bin/python}
export PYTHONPATH="$ROOT/src"

json_value() {
  "$MLX_PYTHON" - "$1" "$2" <<'PY'
import json
import sys
value = json.load(open(sys.argv[1]))
for key in sys.argv[2].split("."):
    value = value[key]
print(value)
PY
}

cd "$ROOT"
if [[ ! -x "$OFFICIAL_PYTHON" || ! -d "$OFFICIAL_CHECKOUT/.git" ]]; then
  print -u2 "Run scripts/setup_official_parity_env.zsh first."
  exit 1
fi
REF_AUDIO=$(json_value reports/full_matrix/input_manifest.json references.en.path)
REF_TEXT=$(json_value reports/full_matrix/input_manifest.json references.en.transcript)
TEXT="The restored observatory clock now keeps perfect time."
INSTRUCTION="Speak calmly and clearly with a measured pace."
PYTORCH_SNAPSHOT=reports/parity/pytorch.json
MLX_SNAPSHOT="reports/parity/mlx-${LABEL}.json"
COMPARISON="reports/parity/comparison-${LABEL}.json"

SOURCE_REVISION=$(git -C "$OFFICIAL_CHECKOUT" rev-parse HEAD)
if [[ -f "$PYTORCH_SNAPSHOT" ]]; then
  SNAPSHOT_REVISION=$(json_value "$PYTORCH_SNAPSHOT" runtime_provenance.official_source_revision)
  if [[ "$SNAPSHOT_REVISION" != "$SOURCE_REVISION" ]]; then
    print -u2 "Existing PyTorch snapshot uses a different official source commit."
    exit 1
  fi
else
  "$OFFICIAL_PYTHON" scripts/capture_pytorch_parity.py \
    --official-repo "$OFFICIAL_CHECKOUT" \
    --model models/breeze-bf16 \
    --ref-audio "$REF_AUDIO" --ref-text "$REF_TEXT" \
    --text "$TEXT" --instruction "$INSTRUCTION" \
    --device cpu --output "$PYTORCH_SNAPSHOT"
fi
if [[ -e "$MLX_SNAPSHOT" || -e "$COMPARISON" ]]; then
  print -u2 "Refusing to overwrite existing parity evidence for label: $LABEL"
  exit 1
fi
"$MLX_PYTHON" scripts/capture_mlx_parity.py \
  --model "$MODEL" \
  --ref-audio "$REF_AUDIO" --ref-text "$REF_TEXT" \
  --text "$TEXT" --instruction "$INSTRUCTION" \
  --output "$MLX_SNAPSHOT"
"$MLX_PYTHON" -m mlx_breeze_tts.cli compare-parity \
  "$PYTORCH_SNAPSHOT" "$MLX_SNAPSHOT" \
  --atol "$ATOL" --rtol "$RTOL" --output "$COMPARISON"
