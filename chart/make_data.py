"""Chart data from the bench's results: each variant's latest finished run (bench.py results).

    python3 make_data.py [VARIANT…]   -> data_1024.json, data_512.json
    python3 -m http.server -d chart   then open chart.html?data=data_1024.json (or data_512.json)

A variant listed in PENDING shows as "running…" whatever its latest row says (a result known to be
invalid, waiting on its rerun).
"""

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
RESULTS = HERE.parent / "results" / "results.jsonl"
PENDING = set(sys.argv[1:])
DRAFT = True
FINAL = {"512", "1024"}  # charts whose every run is in and checked: no DRAFT mark

TRAINERS = {
    "zt": {"name": "ZenTrainer", "version": "0.2.0 · abd4261", "logo": "zt_badge.png", "ours": True,
           "built_on": {"logo": "comfy-logo-single.svg", "text": "Built on ComfyUI"}},
    "aitk": {"name": "AI Toolkit", "version": "0.13.39 · 609f3e5", "logo": "aitk_logo.png", "mark": True},
    "fizgig": {"name": "Fizgig", "version": "8.8 · 4b3f07f", "logo": "fizgig_badge.png"},
}
HOW = {
    "zt-w8a8": "ComfyUI's int8 layers (comfy-kitchen kernels)",
    "aitk-w8a8": "convrot8: int8 weights and activations, torch._int_mm",
    "fizgig-w8a8": "its own int8 path, quantised at load",
    "zt-weightonly": "ComfyUI's int8 layers, int8_forward off",
    "aitk-weightonly": "torchao int8 weight-only",
}


def latest() -> dict:
    rows = {}
    for line in RESULTS.read_text().splitlines():
        r = json.loads(line)
        if r.get("exit") == 0 and r.get("s_per_step"):
            rows[r["variant"]] = r
    return rows


def bar(rows, vid, trainer, how):
    r = None if vid in PENDING else rows.get(vid)
    return {"trainer": trainer, "s_per_step": r and round(r["s_per_step"], 2),
            "peak_gb": r and round(r["peak_over_base_mb"] / 1024, 1), "how": how}


NOT_OFFERED = {"fizgig-weightonly": "not offered: its int8 is always W8A8 (int8 matmuls)"}


def pair(rows, suffix, kind):
    bars = []
    for t in ("zt", "aitk", "fizgig"):
        key = f"{t}-{kind}"
        if key in NOT_OFFERED:
            bars.append({"trainer": t, "absent": NOT_OFFERED[key]})
        else:
            bars.append(bar(rows, key + suffix, t, HOW[key] + (", compiled" if suffix else "")))
    return bars


COMMON = {
    "draft": DRAFT, "title": "Krea 2 LoRA training speed", "date": "2026-10-24 · 100 steps each",
    "unit": "seconds per step", "trainers": TRAINERS,
    "footnote": "Seconds per step: the time from step 40 to step 100, read from each trainer's own progress output "
                "as it happens (past every trainer's warm-up and compile). Peak memory: the GPU's use during the "
                "run, less what was in use before. *Fizgig also trains its 8 small input, output and time layers "
                "(264), and pads captions to 512 tokens (its attention trims the padding). All three use PyTorch's "
                "attention: Fizgig's Krea 2 model loads with attn_mode \"torch\", flash-attn installed or not. Every run is reproducible from github.com/Nynxz/lora-trainer-bench.",
}
SETTINGS = [
    ["Model", "Krea 2 Raw (12.9B), int8 weights"],
    ["LoRA", "rank 16, alpha 16, 256 layers*"],
    ["Resolution", "{res}, batch 1"],
    ["Optimizer", "AdamW, 1e-4, constant"],
    ["Memory", "gradient checkpointing, cached latents and captions"],
    ["Not used", "block swap, offloading, samples"],
    ["PyTorch", "2.13.0 (Fizgig: its own 2.10.0)"],
    ["Dataset", "20 synthetic pictures"],
]


PROFILES = HERE.parent / "results" / "profiles"
KINDS = [  # as profile/sitecustomize.py: first match wins, on the CUDA kernel's name
    ("quantise / rotate / dequantise", r"quant|hadamard|convrot|rotat|scale_|amax|absmax"),
    ("int8 matmul", r"signed char|i8|int8.*gemm|imma|_int_mm|int8_linear|s8"),
    ("attention", r"flash|fmha|attention|sdpa|cudnn.*(mha|attn)|softmax"),
    ("bf16/fp32 matmul", r"gemm|sm\d+_xmma|cutlass|nvjet|ampere_|hopper_|blackwell_|cublas"),
    ("optimizer", r"adam|multi_tensor|foreach"),
    ("copies", r"memcpy|memset|copy_|cat_|contiguous"),
    ("norms", r"norm"),
]


def kind(name: str) -> str:
    low = name.lower()
    return next((k for k, pat in KINDS if re.search(pat, low)), "other elementwise / fused")


def profiled(vid: str, trainer: str, note: str = "") -> dict:
    """A profile's step by kind of work. Older profiles kept only their top kernels: those are
    re-sorted and the rest stays where it was."""
    d = json.loads((PROFILES / f"{vid}.json").read_text())
    if "kernels" in d:
        by = {}
        for name, (ms, _) in d["kernels"].items():
            by[kind(name)] = by.get(kind(name), 0) + ms
    else:
        by = dict(d["by_kind_ms"])
        for t in d["top_kernels"]:
            if kind(t["name"]) != t["kind"]:
                by[t["kind"]] -= t["ms"]
                by[kind(t["name"])] = by.get(kind(t["name"]), 0) + t["ms"]
    return {"trainer": trainer, "note": note, "step_ms": d["step_ms"], "idle_ms": d["idle_ms"],
            "kernels": d["kernels_per_step"], "by_kind": {k: round(v, 1) for k, v in by.items()}}


FINDINGS = [
    "<b>The matmuls cost about the same</b> in all three (0.8–1.0 s of int8 and bf16 matmuls a step); "
    "the steps differ in the work around them.",
    "<b>Elementwise ops and copies:</b> 0.38 s for ZenTrainer, 0.88 for AI Toolkit, 1.10 for Fizgig. "
    "ComfyUI's comfy-kitchen kernels fuse the quantise and dequantise around each int8 matmul that "
    "the others run as separate steps.",
    "<b>Attention:</b> 0.27 s on PyTorch's flash kernel (ZenTrainer), 0.38 on cuDNN (AI Toolkit). "
    "Fizgig's padded, masked captions put it on the slower memory-efficient kernel: 1.01 s.",
    "<b>With bf16 matmuls</b> ZenTrainer and AI Toolkit spend the same on matmuls (1.54 and 1.56 s); "
    "the gap is again elementwise ops, copies and attention.",
    "<b>Idle:</b> 16 ms a step for ZenTrainer since it reads the GPU once a step (a8c8a6e; 85 ms before).",
]


def profile_data() -> dict:
    return {
        "draft": False, "title": "Where a training step goes", "date": "2026-10-24 · steps 45–47 profiled",
        "subtitle": "Krea 2 LoRA at 1024 px: GPU time per step by kind of work, from torch.profiler",
        "trainers": {**TRAINERS, "zt": {**TRAINERS["zt"], "version": "0.2.0 · a8c8a6e"}}, "findings": FINDINGS,
        "groups": [
            {"title": "int8 weights + int8 matmuls", "note": "W8A8 forward, bf16 backward", "bars": [
                profiled("zt-w8a8", "zt"), profiled("aitk-w8a8", "aitk"), profiled("fizgig-w8a8", "fizgig")]},
            {"title": "int8 weights, bf16 matmuls", "note": "weight-only", "bars": [
                profiled("zt-weightonly", "zt"), profiled("aitk-weightonly", "aitk")]},
        ],
        "footnote": "Measured with torch.profiler over three steps after warm-up (profile/sitecustomize.py hooks "
                    "PyTorch's optimizer step, so no trainer's code is changed); steps run a few percent slower "
                    "under the profiler than in the timed runs. Kernels are sorted into kinds by name. Idle: the "
                    "step's wall time with no GPU kernel running, mostly Python queuing the next work.",
    }


def main() -> None:
    rows = latest()
    big = {**COMMON, "subtitle": "Three trainers, the same job, compared where they do the same maths",
           "settings": [[k, v.format(res="1024 × 1024")] for k, v in SETTINGS],
           "groups": [
               {"title": "int8 weights + int8 matmuls", "note": "W8A8 forward, bf16 backward", "bars": pair(rows, "", "w8a8")},
               {"title": "int8 weights, bf16 matmuls", "note": "weight-only: dequantised for every matmul", "bars": pair(rows, "", "weightonly")},
               {"title": "Compiled · int8 weights + int8 matmuls", "note": "torch.compile on the blocks", "bars": pair(rows, "-compiled", "w8a8")},
               {"title": "Compiled · int8 weights, bf16 matmuls", "note": "torch.compile on the blocks", "bars": pair(rows, "-compiled", "weightonly")},
           ]}
    small = {**COMMON, "subtitle": "At 512 px, where Fizgig publishes its figures (0.59 s, 0.29 s compiled)",
             "settings": [[k, v.format(res="512 × 512")] for k, v in SETTINGS],
             "groups": [
                 {"title": "int8 weights + int8 matmuls", "note": "512 px, rank 16", "bars": [
                     bar(rows, "zt-w8a8-512-r8", "zt", HOW["zt-w8a8"]),
                     bar(rows, "aitk-w8a8-512", "aitk", HOW["aitk-w8a8"]),
                     bar(rows, "fizgig-w8a8-512-r8", "fizgig", HOW["fizgig-w8a8"] + " · claims 0.59 s")]},
                 {"title": "Compiled · int8 weights + int8 matmuls", "note": "512 px, rank 16", "bars": [
                     bar(rows, "zt-w8a8-512-compiled", "zt", HOW["zt-w8a8"] + ", compiled"),
                     bar(rows, "aitk-w8a8-512-compiled", "aitk", HOW["aitk-w8a8"] + ", compiled"),
                     bar(rows, "fizgig-w8a8-512-r8-compiled", "fizgig", HOW["fizgig-w8a8"] + ", compiled · claims 0.29 s")]},
             ]}
    big["draft"] = "1024" not in FINAL
    small["draft"] = "512" not in FINAL
    (HERE / "data_profile.json").write_text(json.dumps(profile_data(), indent=2) + "\n")
    (HERE / "data_1024.json").write_text(json.dumps(big, indent=2) + "\n")
    (HERE / "data_512.json").write_text(json.dumps(small, indent=2) + "\n")


if __name__ == "__main__":
    main()
