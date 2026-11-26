#!/usr/bin/env bash
# Mulakan server. Pautan untuk fon akan dipaparkan.
#   bash studio/start_server.sh              # Chatterbox
#   bash studio/start_server.sh voxcpm2      # VoxCPM2
cd "$(dirname "$0")/.."
MODEL=${1:-chatterbox_ml}
case "$MODEL" in chatterbox_ml) ENV=chatterbox ;; voxcpm2) ENV=voxcpm2 ;; *) echo "Model tidak dikenali: $MODEL"; exit 1 ;; esac
# kurangkan pemecahan memori GPU (penting untuk GPU 6 GB)
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export LATIF_MODEL=$MODEL
exec "benchmark/venvs/$ENV/bin/python" -m latif_studio
