#!/usr/bin/env bash
# Create one isolated venv per model (their torch/transformers pins conflict).
# Usage: bash envs/setup_envs.sh [chatterbox voxcpm2 f5 seedvc eval]
# Needs Python 3.11 and an NVIDIA driver. Install a CUDA build of torch FIRST
# when the default wheel doesn't match your driver, e.g.
#   TORCH_INDEX=https://download.pytorch.org/whl/cu124 bash envs/setup_envs.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3.11}
NAMES=${*:-chatterbox voxcpm2 f5 seedvc eval}
for n in $NAMES; do
  echo "== $n"
  $PY -m venv "venvs/$n"
  "venvs/$n/bin/pip" install -q --upgrade pip
  if [ -n "${TORCH_INDEX:-}" ] && [ "$n" != chatterbox ]; then
    "venvs/$n/bin/pip" install -q torch torchaudio --index-url "$TORCH_INDEX"
  fi
  "venvs/$n/bin/pip" install -q -r "envs/$n.txt"
  "venvs/$n/bin/pip" install -q -e ".[analysis]"
done
