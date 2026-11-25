#!/usr/bin/env bash
# Full Phase-1 benchmark on the target machine. Run after setup_envs.sh and
# after filling refs/manifest.yaml. Each step is resumable.
set -euo pipefail
cd "$(dirname "$0")"
M=refs/manifest.yaml
venvs/eval/bin/vobench sysinfo | tee results_sysinfo.json
venvs/chatterbox/bin/vobench run --tts chatterbox_ml --manifest $M
venvs/voxcpm2/bin/vobench run --tts voxcpm2 --manifest $M
if [ -n "${F5_CKPT:-}" ]; then
  venvs/f5/bin/vobench run --tts f5_malaysian --manifest $M \
    --adapter-arg ckpt_file="$F5_CKPT" --adapter-arg vocab_file="$F5_VOCAB" \
    --adapter-arg model_cfg="${F5_CFG:-F5TTS_Base}"
fi
# Mode B: finish tts_then_vc jobs + speech-to-speech route
TTS_DIRS=$(for d in results/chatterbox_ml results/voxcpm2 results/f5_malaysian; do [ -d $d ] && echo "--tts-results $d"; done)
venvs/seedvc/bin/vobench vc --vc seed_vc2 --manifest $M $TTS_DIRS
venvs/chatterbox/bin/vobench vc --vc chatterbox_vc --manifest $M $TTS_DIRS || true
venvs/eval/bin/vobench score --manifest $M
venvs/eval/bin/vobench report
echo "Open results/REPORT.md and results/listening.html"
