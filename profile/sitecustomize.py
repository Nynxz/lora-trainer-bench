"""Profiles a few training steps of whichever trainer imports it (bench.py profile).

Loaded by Python at start-up through PYTHONPATH, only when BENCH_PROFILE names an output file. It
hooks PyTorch's global optimizer-step callback, which every torch.optim optimizer fires, so no
trainer's code is touched. Steps BENCH_PROFILE_FROM .. +BENCH_PROFILE_STEPS run under
torch.profiler; the GPU kernels' time is written out by kind, with the time the GPU sat idle.
"""

import os

OUT = os.environ.get("BENCH_PROFILE")


def _install() -> None:
        """Runs once the trainer itself has imported torch.optim: importing torch any earlier would
        change start-up (ComfyUI sets CUDA's allocator before torch loads)."""
        import json
        import re
        import time

        import torch
        from torch.optim.optimizer import register_optimizer_step_post_hook

        FROM = int(os.environ.get("BENCH_PROFILE_FROM", "45"))
        STEPS = int(os.environ.get("BENCH_PROFILE_STEPS", "3"))
        KINDS = [  # first match wins; matched against the CUDA kernel's name
            ("quantise / rotate / dequantise", r"quant|hadamard|convrot|rotat|scale_|amax|absmax"),
            ("int8 matmul", r"signed char|i8|int8.*gemm|imma|_int_mm|int8_linear|s8"),
            ("attention", r"flash|fmha|attention|sdpa|cudnn.*(mha|attn)|softmax"),
            ("bf16/fp32 matmul", r"gemm|sm\d+_xmma|cutlass|nvjet|ampere_|hopper_|blackwell_|cublas"),
            ("optimizer", r"adam|multi_tensor|foreach"),
            ("copies", r"memcpy|memset|copy_|cat_|contiguous"),
            ("norms", r"norm"),
        ]
        state = {"n": 0, "prof": None, "t0": 0.0}

        def kind(name: str) -> str:
            low = name.lower()
            for k, pat in KINDS:
                if re.search(pat, low):
                    return k
            return "other elementwise / fused"

        def hook(opt, args, kwargs):
            state["n"] += 1
            n = state["n"]
            if n == FROM - 1:
                torch.cuda.synchronize()
                state["prof"] = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                                   torch.profiler.ProfilerActivity.CUDA],
                                                       with_stack=bool(os.environ.get("BENCH_PROFILE_TRACE")))
                state["prof"].__enter__()
                state["t0"] = time.perf_counter()
            elif n == FROM - 1 + STEPS and state["prof"] is not None:
                torch.cuda.synchronize()
                wall = time.perf_counter() - state["t0"]
                prof, state["prof"] = state["prof"], None
                prof.__exit__(None, None, None)
                if os.environ.get("BENCH_PROFILE_TRACE"):  # the timeline, for where the GPU waits (gaps.py)
                    prof.export_chrome_trace(OUT.removesuffix(".json") + ".trace.json")
                kernels: dict[str, list] = {}
                for e in prof.key_averages():
                    t = getattr(e, "self_device_time_total", None) or getattr(e, "self_cuda_time_total", 0)
                    if t and e.device_type == torch.autograd.DeviceType.CUDA:
                        kernels[e.key] = [t / 1e3 / STEPS, e.count / STEPS]  # ms and calls per step
                by_kind: dict[str, float] = {}
                for name, (ms, _) in kernels.items():
                    by_kind[kind(name)] = by_kind.get(kind(name), 0) + ms
                busy = sum(by_kind.values())
                step_ms = wall * 1e3 / STEPS
                top = sorted(kernels.items(), key=lambda kv: -kv[1][0])[:40]
                with open(OUT, "w") as f:
                    json.dump({"steps": [FROM, FROM + STEPS - 1], "step_ms": round(step_ms, 1),
                               "gpu_busy_ms": round(busy, 1), "idle_ms": round(step_ms - busy, 1),
                               "kernels_per_step": round(sum(c for _, c in kernels.values())),
                               "by_kind_ms": {k: round(v, 1) for k, v in sorted(by_kind.items(), key=lambda kv: -kv[1])},
                               "top_kernels": [{"name": n[:160], "ms": round(ms, 2), "calls": round(c, 1), "kind": kind(n)}
                                               for n, (ms, c) in top],
                               "kernels": {n: [round(ms, 3), round(c, 1)] for n, (ms, c) in kernels.items()}}, f, indent=1)

        register_optimizer_step_post_hook(hook)


if OUT:
    import importlib.abc
    import sys

    class _After(importlib.abc.MetaPathFinder):
        """Calls _install right after torch.optim.optimizer finishes importing."""

        def find_spec(self, name, path=None, target=None):
            if name != "torch.optim.optimizer":
                return None
            sys.meta_path.remove(self)
            import importlib.util

            spec = importlib.util.find_spec(name)
            run = spec.loader.exec_module

            def exec_module(module):
                run(module)
                _install()

            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _After())
