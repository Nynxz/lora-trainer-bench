#!/bin/sh
# The whole benchmark, start to finish: install the three trainers at their pinned commits, fetch the
# Krea 2 files, make the dataset and caches, then run every variant in matrix.toml.
#     ./run.sh                 everything
#     ./run.sh --only compiled  just the variants whose group mentions "compiled"
set -e
cd "$(dirname "$0")"
for t in zentrainer ai-toolkit fizgig; do trainers/install.sh "$t"; done
[ -f models/diffusion_models/krea2_raw_bf16.safetensors ] || scripts/download_models.sh
uv run --no-project --with pyyaml --with pillow python bench.py prepare
uv run --no-project --with pyyaml --with pillow python bench.py all "$@"
