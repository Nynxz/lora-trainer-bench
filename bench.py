"""Run the benchmark: each variant in matrix.toml, at two step counts, timed from the outside.

    uv run --no-project --with pyyaml --with pillow python bench.py prepare        dataset, caches, environment record
    uv run --no-project --with pyyaml --with pillow python bench.py run VARIANT [STEPS]
    uv run --no-project --with pyyaml --with pillow python bench.py all [--only GROUP]
    uv run --no-project --with pyyaml --with pillow python bench.py results        seconds per step for each variant

Each variant runs once, for [runs] steps. The runner reads the trainer's output as it comes and
notes, on its own clock, when each training step is reported; seconds per step is the time from
step `timed_from` to the last step, over the steps between. Loading, caching, warm-up and saving
fall outside that window, and no trainer's own speed figure is used. The GPU's peak memory
(nvidia-smi, every half second) and the run's wall-clock time go to results/results.jsonl too; the
whole output to work/logs/.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CFG = tomllib.loads((ROOT / "bench.toml").read_text())
MATRIX = tomllib.loads((ROOT / "matrix.toml").read_text())
SRC = ROOT / CFG["paths"]["trainers"]
MODELS = (ROOT / CFG["paths"]["models"]).resolve()
WORK = (ROOT / CFG["paths"]["work"]).resolve()
RESULTS = ROOT / "results" / "results.jsonl"
FILL = {"models": str(MODELS), "work": str(WORK), **CFG["models"]}
ZT_PORT = 8299  # the benchmark's own ZenTrainer, never a ZenTrainer you already run
IMAGES = 20  # pictures in the dataset: a Fizgig epoch is this many steps


def fill(text: str, **extra) -> str:
    for k, v in {**FILL, **extra}.items():
        text = text.replace("{" + k + "}", str(v))
    return text


def variant(vid: str) -> dict:
    for v in MATRIX["variant"]:
        if v["id"] == vid:
            return v
    raise SystemExit(f"no variant {vid!r} in matrix.toml")


def set_dotted(d: dict, key: str, value) -> None:
    *parents, leaf = key.split(".")
    for p in parents:
        d = d.setdefault(p, {})
    d[leaf] = value


# ---- each trainer's command for a variant ----------------------------------------------------------

def zentrainer(v: dict, steps: int) -> tuple[list[str], Path, dict]:
    spec = json.loads(fill((ROOT / "configs/zentrainer.json").read_text()))
    spec["name"] = f"bench-{v['id']}-{steps}"
    spec["optim"]["steps"] = steps
    for k, val in v.get("set", {}).items():
        if k == "resolution":
            spec["dataset"]["resolution"] = val
        elif k == "rank":
            spec["network"].update(rank=val, alpha=val)
        else:
            set_dotted(spec, k, val)
    path = WORK / "configs" / f"{v['id']}_{steps}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec, indent=2))
    # --every 1 --interval 0.5: a line per step, within half a second of the server logging it
    return [str(SRC / "zentrainer/scripts/zt"), "train", str(path), "--port", str(ZT_PORT), "--follow",
            "--every", "1", "--interval", "0.5"], ROOT, {}


def ai_toolkit(v: dict, steps: int) -> tuple[list[str], Path, dict]:
    import yaml

    conf = yaml.safe_load(fill((ROOT / "configs/ai-toolkit.yaml").read_text(), name=f"bench-{v['id']}-{steps}"))
    proc = conf["config"]["process"][0]
    proc["train"]["steps"] = steps
    for k, val in v.get("set", {}).items():
        if k == "resolution":
            proc["datasets"][0]["resolution"] = [val]
        elif k == "rank":
            proc["network"].update(linear=val, linear_alpha=val)
        else:
            set_dotted(proc, k, val)
    path = WORK / "configs" / f"{v['id']}_{steps}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(conf, sort_keys=False))
    return [str(SRC / "ai-toolkit/.venv/bin/python"), "run.py", str(path)], SRC / "ai-toolkit", {}


FIZGIG_FLAGS = ["--learning_rate", "1e-4", "--lr_scheduler", "constant", "--max_grad_norm", "1.0", "--seed", "42",
                "--optimizer_type", "adamw", "--blocks_to_swap", "0", "--save_every_n_epochs", "100000"]


def fizgig(v: dict, steps: int) -> tuple[list[str], Path, dict]:
    s = {"precision": "int8", "compile_blocks": "off", "resolution": 1024, "rank": 16, **v.get("set", {})}
    dataset = WORK / "configs" / f"fizgig_{s['resolution']}.toml"
    dataset.parent.mkdir(parents=True, exist_ok=True)
    dataset.write_text(fill((ROOT / "configs/fizgig.toml").read_text(), resolution=s["resolution"]))
    return [str(SRC / "fizgig/venv/bin/python"), "src/fizgig/families/train.py", "--family", "krea2",
            "--dataset_config", str(dataset), "--dit", str(MODELS / "diffusion_models" / CFG["models"]["dit_bf16"]),
            "--vae", str(MODELS / "vae" / CFG["models"]["vae"]),
            "--text_encoder", str(MODELS / "text_encoders" / CFG["models"]["text_encoder"]),
            "--output_dir", str(WORK / "out/fizgig" / f"{v['id']}_{steps}"), "--output_name", "bench",
            "--network_dim", str(s["rank"]), "--network_alpha", str(s["rank"]),
            "--max_train_epochs", str(steps // IMAGES), "--precision", s["precision"],
            "--compile_blocks", s["compile_blocks"], *FIZGIG_FLAGS], SRC / "fizgig", {}


COMMANDS = {"zentrainer": zentrainer, "ai-toolkit": ai_toolkit, "fizgig": fizgig}
# each trainer's line for a training step (its own caching progress bars don't match): step, total
STEP_LINE = {
    "zentrainer": re.compile(r"^step (\d+)/(\d+)"),
    "ai-toolkit": re.compile(r"bench-[^:\r\n]*:\s+\d+%\|[^|]*\|\s*(\d+)/(\d+) \["),
    "fizgig": re.compile(r"steps:\s+\d+%\|[^|]*\|\s*(\d+)/(\d+) \["),
}


# ---- timing -----------------------------------------------------------------------------------------

def gpu_used_mb() -> int:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return int(out.split()[0])


def stop_zentrainer() -> None:
    """The benchmark's ZenTrainer keeps running after `zt train` (it holds GPU memory, and the code
    it loaded): stop it before every run, so each starts fresh on the pinned commit. Only the one on
    ZT_PORT that lives under trainers/src/zentrainer (resolved: it may be a symlink)."""
    home = os.path.realpath(SRC / "zentrainer")
    out = subprocess.run(["ss", "-ltnp"], capture_output=True, text=True).stdout
    for pid in re.findall(rf":{ZT_PORT}\s.*?pid=(\d+)", out):
        cwd = os.path.realpath(f"/proc/{pid}/cwd")
        if cwd == home or cwd.startswith(home + os.sep):
            os.kill(int(pid), signal.SIGTERM)
            for _ in range(60):  # until it has let go of the port (killed after a minute)
                time.sleep(1)
                if not os.path.exists(f"/proc/{pid}"):
                    break
            else:
                os.kill(int(pid), signal.SIGKILL)
                time.sleep(2)
    for _ in range(30):
        if f":{ZT_PORT} " not in subprocess.run(["ss", "-ltn"], capture_output=True, text=True).stdout:
            break
        time.sleep(1)


def run(vid: str, steps: int) -> dict:
    v = variant(vid)
    stop_zentrainer()
    cmd, cwd, env = COMMANDS[v["trainer"]](v, steps)
    log = WORK / "logs" / f"{vid}_{steps}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    base = gpu_used_mb()
    peak = [base]
    done = threading.Event()

    def sample() -> None:
        while not done.is_set():
            peak[0] = max(peak[0], gpu_used_mb())
            time.sleep(0.5)

    threading.Thread(target=sample, daemon=True).start()
    seen: dict[int, float] = {}  # step -> when it was first reported (our clock)
    pattern = STEP_LINE[v["trainer"]]
    t0 = time.time()
    with open(log, "w") as f:
        proc = subprocess.Popen(cmd, cwd=cwd, env={**os.environ, **env, "PYTHONUNBUFFERED": "1"},
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        buf = b""
        while chunk := proc.stdout.read1(65536):
            now = time.time()
            f.write(chunk.decode(errors="replace"))
            f.flush()  # the log fills as the run goes
            buf += chunk
            *lines, buf = re.split(rb"[\r\n]", buf)  # progress bars redraw with \r
            for line in lines:
                m = pattern.search(line.decode(errors="replace"))
                if m:
                    seen.setdefault(int(m.group(1)), now)
        code = proc.wait()
    wall = time.time() - t0
    done.set()
    start = MATRIX["runs"]["timed_from"]
    last = max(seen, default=0)
    sps = (seen[last] - seen[start]) / (last - start) if start in seen and last > start else None
    row = {"variant": vid, "trainer": v["trainer"], "group": v["group"], "steps": steps,
           "s_per_step": round(sps, 4) if sps else None, "timed": [start, last], "steps_seen": len(seen),
           # every step's time since the first, so anyone can check they arrived evenly (not in buffered bunches)
           "step_times": {k: round(t - min(seen.values()), 2) for k, t in sorted(seen.items())},
           "wall_s": round(wall, 1), "exit": code, "gpu_before_mb": base, "gpu_peak_mb": peak[0],
           "peak_over_base_mb": peak[0] - base, "at": time.strftime("%Y-%m-%d %H:%M:%S")}
    RESULTS.parent.mkdir(exist_ok=True)
    with open(RESULTS, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)
    return row


# ---- preparing, and reading the results -------------------------------------------------------------

def prepare() -> None:
    """The dataset, the copies and caches each trainer reads it from, and a record of the machine."""
    subprocess.run([sys.executable, str(ROOT / "dataset/make_dataset.py")], check=True)
    data = WORK / "data/bench20"
    zt_input = SRC / "zentrainer/comfyui/input/bench20"  # ZenTrainer reads datasets from its ComfyUI's input/
    zt_input.mkdir(parents=True, exist_ok=True)
    for f in data.iterdir():
        (zt_input / f.name).write_bytes(f.read_bytes())
    fz = SRC / "fizgig"
    for res in sorted({1024, *(v["set"].get("resolution", 1024) for v in MATRIX["variant"] if v["trainer"] == "fizgig")}):
        dataset = WORK / "configs" / f"fizgig_{res}.toml"
        dataset.parent.mkdir(parents=True, exist_ok=True)
        dataset.write_text(fill((ROOT / "configs/fizgig.toml").read_text(), resolution=res))
        for stage, model in (("latents", MODELS / "vae" / CFG["models"]["vae"]),
                             ("text", MODELS / "text_encoders" / CFG["models"]["text_encoder"])):
            subprocess.run([str(fz / "venv/bin/python"), "src/fizgig/families/cache.py", "--family", "krea2", "--stage",
                            stage, "--dataset_config", str(dataset), "--model", str(model), "--skip_existing"],
                           cwd=fz, check=True)
    environment()


def environment() -> None:
    """GPU, driver, CPU, memory, and each trainer's commit and torch: results/environment.json."""
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,power.limit", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    cpu = next((l.split(":", 1)[1].strip() for l in Path("/proc/cpuinfo").read_text().splitlines() if l.startswith("model name")), "")
    mem = next((l.split()[1] for l in Path("/proc/meminfo").read_text().splitlines() if l.startswith("MemTotal")), "0")
    pythons = {"zentrainer": "zentrainer/comfyui/.venv/bin/python", "ai-toolkit": "ai-toolkit/.venv/bin/python",
               "fizgig": "fizgig/venv/bin/python"}
    trainers = {}
    for name, py in pythons.items():
        torch = subprocess.run([str(SRC / py), "-c", "import torch; print(torch.__version__)"], capture_output=True, text=True)
        trainers[name] = {"commit": subprocess.run(["git", "-C", str(SRC / name), "rev-parse", "--short", "HEAD"],
                                                   capture_output=True, text=True).stdout.strip(),
                          "torch": torch.stdout.strip() or "not installed"}
    env = {"gpu": smi, "cpu": cpu, "ram_gb": round(int(mem) / 2**20), "trainers": trainers,
           "recorded": time.strftime("%Y-%m-%d %H:%M:%S")}
    (ROOT / "results/environment.json").write_text(json.dumps(env, indent=2) + "\n")
    print(json.dumps(env, indent=2))


def results() -> list[dict]:
    """Each variant's latest finished run: seconds per step and peak memory."""
    rows = [json.loads(l) for l in RESULTS.read_text().splitlines() if l.strip()] if RESULTS.exists() else []
    out = []
    for v in MATRIX["variant"]:
        got = [r for r in rows if r.get("variant") == v["id"] and r["exit"] == 0 and r.get("s_per_step")]
        if got:
            r = got[-1]
            out.append({"variant": v["id"], "trainer": v["trainer"], "group": v["group"], "s_per_step": round(r["s_per_step"], 3),
                        "peak_gb": round(r["peak_over_base_mb"] / 1024, 1)})
    for r in out:
        print(f"{r['group']:42} {r['variant']:30} {r['s_per_step']:7.3f} s/step   {r['peak_gb']:5.1f} GB")
    return out


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "results"
    if cmd == "prepare":
        prepare()
    elif cmd == "run":
        run(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else MATRIX["runs"]["steps"])
    elif cmd == "all":
        only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
        # one trainer at a time, so the benchmark's ZenTrainer is started once and stopped once
        todo = [v for v in MATRIX["variant"] if not only or only in v["group"]]
        for v in sorted(todo, key=lambda v: v["trainer"] != "zentrainer"):
            run(v["id"], MATRIX["runs"]["steps"])
        stop_zentrainer()
        results()
    elif cmd == "results":
        results()
    elif cmd == "env":
        environment()
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
