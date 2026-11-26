#!/usr/bin/env bash
# Mulakan server. Pautan untuk fon akan dipaparkan.
cd "$(dirname "$0")/.."
exec benchmark/venvs/voxcpm2/bin/python -m latif_studio
