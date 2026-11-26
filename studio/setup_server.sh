#!/usr/bin/env bash
# Pasang server Latif Voice di PC (Linux / WSL2). Guna venv model utama (VoxCPM2).
# Jalankan dari folder vo-studio:  bash studio/setup_server.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3.11}
V=benchmark/venvs/voxcpm2
if [ ! -d "$V" ]; then
  $PY -m venv "$V"
  "$V/bin/pip" install -q --upgrade pip
  [ -n "${TORCH_INDEX:-}" ] && "$V/bin/pip" install -q torch torchaudio --index-url "$TORCH_INDEX"
  "$V/bin/pip" install -q -r benchmark/envs/voxcpm2.txt
fi
"$V/bin/pip" install -q -e "benchmark[analysis]" -e studio/server
# Pilihan: transkripsi automatik rakaman (Whisper)
"$V/bin/pip" install -q faster-whisper || echo "(faster-whisper gagal dipasang: isi transkrip secara manual)"
echo "Siap. Mulakan dengan: bash studio/start_server.sh"
