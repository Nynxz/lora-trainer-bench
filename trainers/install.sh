#!/bin/sh
# Install one trainer from its own repository, at the commit pins.json names, the way that project
# says to install it. Nothing of theirs lives in this repository: it's cloned into trainers/src/.
#
#     trainers/install.sh zentrainer|ai-toolkit|fizgig
set -e
cd "$(dirname "$0")/.."
name="$1"
repo=$(python3 -c "import json,sys; print(json.load(open('pins.json'))[sys.argv[1]]['repo'])" "$name")
commit=$(python3 -c "import json,sys; print(json.load(open('pins.json'))[sys.argv[1]]['commit'])" "$name")
dir="trainers/src/$name"
mkdir -p trainers/src
[ -d "$dir/.git" ] || git clone -q "$repo" "$dir"
git -C "$dir" fetch -q origin
git -C "$dir" checkout -q "$commit"
echo "$name at $(git -C "$dir" rev-parse --short HEAD)"
cd "$dir"
case "$name" in
  zentrainer)
    # its own setup: a pinned ComfyUI with its own environment, torch as its comfy.lock pins
    python3 scripts/zt setup --torch cu130 --models-from "$OLDPWD/models"
    ;;
  ai-toolkit)
    # its README: Python 3.12, torch 2.13.0 for CUDA 13.0, then its requirements
    uv venv --python 3.12 .venv
    uv pip install --python .venv/bin/python torch==2.13.0 torchvision==0.28.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu130
    uv pip install --python .venv/bin/python -r requirements.txt
    ;;
  fizgig)
    # its own installer (Linux): makes venv/ with its pinned torch
    python3 install_fizgig.py < /dev/null
    ;;
  *) echo "unknown trainer $name"; exit 1 ;;
esac
