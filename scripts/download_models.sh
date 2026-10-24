#!/bin/sh
# The four Krea 2 files every trainer reads, from Comfy-Org's public repository (not gated), into
# models/ in ComfyUI's folder layout. About 45 GB.
set -e
cd "$(dirname "$0")/.."
for f in diffusion_models/krea2_raw_bf16.safetensors diffusion_models/krea2_raw_int8_convrot.safetensors \
         text_encoders/qwen3vl_4b_bf16.safetensors vae/qwen_image_vae.safetensors; do
  uvx --from huggingface_hub hf download Comfy-Org/Krea-2 "$f" --local-dir models
done
# sha256 of each, so a result can say exactly which files it used
( cd models && sha256sum diffusion_models/*.safetensors text_encoders/*.safetensors vae/*.safetensors ) > results/models.sha256
cat results/models.sha256
