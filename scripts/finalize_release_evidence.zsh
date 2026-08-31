#!/bin/zsh
set -euo pipefail

if (( $# != 3 )); then
  print -u2 "usage: $0 BF16_REVIEWS 8BIT_REVIEWS 4BIT_REVIEWS"
  exit 2
fi

ROOT=${0:A:h:h}
PYTHON_BIN=${PYTHON_BIN:-/Users/vanch/mlx-qwen3-tts/.venv/bin/python}
export PYTHONPATH="$ROOT/src"

cli() {
  "$PYTHON_BIN" -m mlx_breeze_tts.cli "$@"
}

require_file() {
  if [[ ! -f "$1" ]]; then
    print -u2 "required file is missing: $1"
    exit 1
  fi
}

require_absent() {
  if [[ -e "$1" ]]; then
    print -u2 "refusing to overwrite existing final evidence: $1"
    exit 1
  fi
}

cd "$ROOT"
reviews=("${1:A}" "${2:A}" "${3:A}")
variants=(bf16 8bit 4bit)

for index in 1 2 3; do
  variant=${variants[$index]}
  source=${reviews[$index]}
  destination="reports/release-v2/$variant/manual_reviews.json"
  require_file "$source"
  require_absent "$destination"
  cp "$source" "$destination"
done

require_absent reports/release-v2/bf16/summary.final.json
cli apply-listening-review \
  reports/release-v2/bf16/summary.parity.json \
  reports/release-v2/bf16/manual_reviews.json \
  --output reports/release-v2/bf16/summary.final.json

for bits in 8 4; do
  directory="reports/release-v2/${bits}bit"
  full="reports/ablation/${bits}bit-full/summary.metrics-corpus-v2.json"
  reviewed="$directory/summary.reviewed.json"
  comparison="$directory/quantization-comparison.json"
  final="$directory/summary.final.json"
  require_file "$full"
  require_absent "$reviewed"
  require_absent "$comparison"
  require_absent "$final"
  cli apply-listening-review \
    "$directory/summary.parity.json" \
    "$directory/manual_reviews.json" \
    --output "$reviewed"
  cli compare-quantization "$full" "$reviewed" --output "$comparison"
  cli apply-quantization-evidence "$reviewed" "$comparison" --output "$final"
done

for variant in $variants; do
  cli render-report \
    "reports/release-v2/$variant/summary.final.json" \
    --output "reports/release-v2/$variant"
done

cli verify-evidence reports/release-v2 \
  --output reports/release-v2/completion_audit.json

print "Final release evidence passed independent verification."
