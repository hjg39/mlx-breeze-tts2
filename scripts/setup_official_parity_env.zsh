#!/bin/zsh
set -euo pipefail

ROOT=${0:A:h:h}
CHECKOUT=${OFFICIAL_CHECKOUT:-$ROOT/.parity/official-breeze-tts}
VENV=${OFFICIAL_VENV:-$ROOT/.parity/official-venv}
UV_BIN=${UV_BIN:-$(command -v uv)}

mkdir -p "$ROOT/.parity" "$ROOT/reports/parity"
if [[ ! -d "$CHECKOUT/.git" ]]; then
  git clone https://github.com/breezeblue-ai/breeze-tts.git "$CHECKOUT"
fi
ORIGIN=$(git -C "$CHECKOUT" remote get-url origin)
if [[ "$ORIGIN" != "https://github.com/breezeblue-ai/breeze-tts.git" ]]; then
  print -u2 "Unexpected official source origin: $ORIGIN"
  exit 1
fi
if [[ -n $(git -C "$CHECKOUT" status --porcelain) ]]; then
  print -u2 "Official source checkout must be clean: $CHECKOUT"
  exit 1
fi
if [[ ! -x "$VENV/bin/python" ]]; then
  "$UV_BIN" venv --python 3.11 "$VENV"
fi
"$UV_BIN" pip install --python "$VENV/bin/python" -r "$CHECKOUT/requirements.txt"

"$VENV/bin/python" - "$CHECKOUT" "$ROOT/reports/parity/official_environment.json" <<'PY'
import importlib.metadata
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

checkout = Path(sys.argv[1]).resolve()
output = Path(sys.argv[2]).resolve()
revision = subprocess.run(
    ["git", "-C", str(checkout), "rev-parse", "HEAD"],
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()
packages = {
    distribution.metadata["Name"]: distribution.version
    for distribution in importlib.metadata.distributions()
    if distribution.metadata.get("Name")
}
output.write_text(json.dumps({
    "schema_version": 1,
    "status": "pass",
    "source": "https://github.com/breezeblue-ai/breeze-tts",
    "source_revision": revision,
    "checkout": str(checkout),
    "python": sys.version,
    "packages": dict(sorted(packages.items(), key=lambda item: item[0].lower())),
    "captured_at": datetime.now(timezone.utc).isoformat(),
}, indent=2))
print(output)
PY
