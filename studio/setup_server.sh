#!/usr/bin/env bash
# Pasang server Latif Voice di PC (Linux / WSL2).
#   bash studio/setup_server.sh                 # Chatterbox (lalai, sesuai GPU 6 GB seperti RTX 4050)
#   bash studio/setup_server.sh voxcpm2         # VoxCPM2 (perlu ~8 GB VRAM; boleh dicuba)
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL=${1:-chatterbox_ml}
case "$MODEL" in
  chatterbox_ml) ENV=chatterbox ;;
  voxcpm2) ENV=voxcpm2 ;;
  *) echo "Model tidak dikenali: $MODEL (pilih chatterbox_ml atau voxcpm2)"; exit 1 ;;
esac
PY=${PYTHON:-python3}
V=benchmark/venvs/$ENV
if [ ! -d "$V" ]; then
  $PY -m venv "$V"
  "$V/bin/pip" install -q --upgrade pip
  if [ -n "${TORCH_INDEX:-}" ] && [ "$ENV" != chatterbox ]; then
    "$V/bin/pip" install -q torch torchaudio --index-url "$TORCH_INDEX"
  fi
  "$V/bin/pip" install -q -r "benchmark/envs/$ENV.txt"
fi
"$V/bin/pip" install -q -e "benchmark[analysis]" -e studio/server
"$V/bin/pip" install -q faster-whisper || echo "(faster-whisper gagal dipasang: isi transkrip secara manual)"
"$V/bin/python" -c "import torch; ok=torch.cuda.is_available(); print('GPU:', torch.cuda.get_device_name(0) if ok else 'TIDAK DIKESAN - semak pemacu NVIDIA')"
echo "Memuat turun berat model (sekali sahaja, beberapa GB)..."
case "$MODEL" in
  chatterbox_ml) PRE="from chatterbox.mtl_tts import ChatterboxMultilingualTTS as M; M.from_pretrained(device='cpu')" ;;
  voxcpm2) PRE="from voxcpm import VoxCPM; VoxCPM.from_pretrained('openbmb/VoxCPM2', load_denoiser=False, device='cpu')" ;;
esac
"$V/bin/python" -c "$PRE" >/dev/null && echo "Model sedia." || echo "AMARAN: muat turun model gagal; ia akan dicuba semula semasa server dimulakan."
echo "Siap. Mulakan dengan: bash studio/start_server.sh $MODEL"
