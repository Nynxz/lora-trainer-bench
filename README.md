# LoRA trainer benchmark

How fast do LoRA trainers train the same job on the same GPU? This repository measures
[ZenTrainer](https://github.com/Nynxz/ZenTrainer), [AI Toolkit](https://github.com/ostris/ai-toolkit)
and [Fizgig](https://github.com/shootthesound/Fizgig) side by side, and is set up so anyone can rerun
it and check every number. The first comparison is **Krea 2**; other models can join as matrices of
their own.

None of the trainers' code is in here. `trainers/install.sh` clones each one from its own repository
at the commit in [`pins.json`](pins.json) and installs it the way that project says to. What's here
is only the configurations, the dataset generator, the runner and the results.

## Results

> Filled in from [`results/`](results/) once the runs are done. Each number links back to the raw
> rows and logs that produced it.

The headline numbers compare **like with like**: each pair below does the same maths, so the
difference within a pair is the trainers' implementations, not their choice of precision.

| Same maths | ZenTrainer | The other trainer |
|---|---|---|
| int8 weights + int8 matmuls (W8A8 forward, bf16 backward) | `int8_forward: true` | Fizgig `--precision int8` |
| int8 weights, bf16 matmuls (weight-only) | `int8_forward: false` | AI Toolkit `qtype: int8` |

Comparing across the pairs mostly measures int8 matmuls against bf16 ones, which is worth knowing
(it's most of the gap you'd see from each trainer's defaults), but it isn't a fair head-to-head of
the trainers, so it's not presented as one.

## What's measured

Everything below is the same for every trainer:

| | |
|---|---|
| Model | Krea 2 Raw (12.9B), from [Comfy-Org/Krea-2](https://huggingface.co/Comfy-Org/Krea-2) |
| LoRA | rank 16, alpha 16, on the transformer's blocks and text-fusion blocks (256 layers) |
| Data | 20 synthetic 1024×1024 pictures with captions ([`dataset/make_dataset.py`](dataset/make_dataset.py)), one bucket |
| Training | batch 1, AdamW, learning rate 1e-4 constant, no warm-up, grad clipping 1.0, bf16 |
| Memory | gradient checkpointing on; latents and caption encodings cached first; no block swap or offloading |
| Off | sampling, validation, checkpoints during the run, torch.compile (except in the compiled variants) |

**Seconds per step** comes from timing each variant twice from the outside, at 40 and at 160 steps:
`(time₁₆₀ − time₄₀) / 120`. Loading the model, caching and saving cost the same in both runs, so
they cancel, and no trainer's own progress readout is trusted. 40 steps is past every trainer's
warm-up (Fizgig, for example, warms up over its first two epochs).

**Peak memory** is the GPU's total use during the run (`nvidia-smi`, every half second), less what
was in use before it started.

Every configuration is listed in [`matrix.toml`](matrix.toml); a variant only states what differs.

### Differences that remain, and what they mean

We matched everything we could. These are left, and none changes which maths a pair does:

- **The model file.** ZenTrainer reads ComfyUI's ready-quantised int8 file (`krea2_raw_int8_convrot`,
  rotated, a scale per row). AI Toolkit and Fizgig quantise the bf16 file as they load it. This
  affects accuracy slightly, not speed.
- **Caption length.** Fizgig pads every caption to 512 tokens and masks the padding; the other two
  use the caption's real length (about 27 tokens here). The padding doesn't change Fizgig's results,
  but its blocks' linear layers do about 12% more work per step.
- **LoRA layers.** Fizgig also trains 8 small layers outside the blocks (input, output, time and text
  projections): 264 against 256. A fraction of a percent of the work.
- **PyTorch.** Each trainer uses the version it ships with: 2.13.0 for ZenTrainer and AI Toolkit,
  2.10.0 for Fizgig.
- **Attention.** All three use PyTorch's SDPA (none has flash-attn, SageAttention or xformers
  installed). AI Toolkit asks for the cuDNN kernel; the others let PyTorch choose.

If you think a trainer is set up wrong, the configuration is in [`configs/`](configs/) and the
flags in [`bench.py`](bench.py). Open an issue or a pull request: the point is a fair number.

### Checking a trainer against its own figures

Fizgig's code notes 0.59 s/step on int8 and 0.29 compiled, on a 5090. The `check: 512 px, rank 8`
variants run Fizgig, and ZenTrainer beside it, at the settings we believe that was measured at, to
confirm the setup here reproduces it.

## Running it yourself

Needs Linux, an NVIDIA GPU with at least 24 GB (32 GB recommended), about 64 GB of RAM, about 120 GB
of disk, `git`, `python3` (3.11 or newer) and [uv](https://docs.astral.sh/uv/).

```sh
git clone <this repository> && cd lora-trainer-bench
./run.sh                    # install, download the models, prepare, run everything (a few hours)
```

Or step by step:

```sh
trainers/install.sh zentrainer        # also ai-toolkit, fizgig
scripts/download_models.sh            # the four Krea 2 files, about 45 GB, with their sha256
uv run --no-project --with pyyaml python bench.py prepare      # dataset, caches, results/environment.json
uv run --no-project --with pyyaml python bench.py run zt-w8a8  # one variant, at both step counts
uv run --no-project --with pyyaml python bench.py results      # seconds per step so far
```

Already have the Krea 2 files? Point `paths.models` in [`bench.toml`](bench.toml) at a folder with
ComfyUI's layout (`diffusion_models/`, `text_encoders/`, `vae/`).

Run nothing else on the GPU meanwhile: another program's work would show up as a slower trainer.

## Layout

```
pins.json            each trainer's repository and commit
bench.toml           where the trainers, models and work folder live
matrix.toml          every variant the benchmark runs
configs/             each trainer's configuration (the settings every variant shares)
trainers/install.sh  clones and installs one trainer, as its project documents
scripts/             fetching the models
dataset/             the synthetic dataset generator
bench.py             the runner: prepare, run, results
results/             results.jsonl (every run), environment.json (the machine), models.sha256
chart/               the published chart, made from results/
```

## Licences

This repository's own files are MIT. The trainers are fetched from their own repositories and stay
under their own licences; Krea 2 is under Krea's licence, from Comfy-Org's repository.
