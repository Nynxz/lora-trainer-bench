"""Run the benchmark: each variant in matrix.toml, at two step counts, timed from the outside.

    uv run --no-project --with pyyaml python bench.py prepare        dataset, caches, environment record
    uv run --no-project --with pyyaml python bench.py run VARIANT [STEPS]
    uv run --no-project --with pyyaml python bench.py all [--only GROUP]
    uv run --no-project --with pyyaml python bench.py results        seconds per step for each variant

A run's wall-clock time and the GPU's peak memory (nvidia-smi, every half second) go to
results/results.jsonl; its whole output to work/logs/. Seconds per step for a variant is
(time at `long` steps - time at `short`) / (long - short): loading, caching and saving are the same
in both runs and cancel, whatever each trainer prints about its own speed.
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
    return [str(SRC / "zentrainer/scripts/zt"), "train", str(path), "--port", str(ZT_PORT), "--follow", "--every", "20"], ROOT, {}


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


# ---- timing -----------------------------------------------------------------------------------------

def gpu_used_mb() -> int:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return int(out.split()[0])


def stop_zentrainer() -> None:
    """The benchmark's ZenTrainer keeps running after `zt train` (it holds GPU memory): stop it
    before another trainer runs. Only the one on ZT_PORT that lives under trainers/src."""
    out = subprocess.run(["ss", "-ltnp"], capture_output=True, text=True).stdout
    for pid in re.findall(rf":{ZT_PORT}\s.*?pid=(\d+)", out):
        if os.readlink(f"/proc/{pid}/cwd").startswith(str(SRC)):
            os.kill(int(pid), signal.SIGTERM)
            time.sleep(10)


def run(vid: str, steps: int) -> dict:
    v = variant(vid)
    if v["trainer"] != "zentrainer":
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
    t0 = time.time()
    with open(log, "w") as f:
        code = subprocess.run(cmd, cwd=cwd, env={**os.environ, **env}, stdout=f, stderr=subprocess.STDOUT).returncode
    wall = time.time() - t0
    done.set()
    row = {"variant": vid, "trainer": v["trainer"], "group": v["group"], "steps": steps, "wall_s": round(wall, 1),
           "exit": code, "gpu_before_mb": base, "gpu_peak_mb": peak[0], "peak_over_base_mb": peak[0] - base,
           "at": time.strftime("%Y-%m-%d %H:%M:%S")}
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
    """Seconds per step for each variant with both runs done (the latest of each), and peak memory."""
    rows = [json.loads(l) for l in RESULTS.read_text().splitlines() if l.strip()] if RESULTS.exists() else []
    short, long = MATRIX["runs"]["short"], MATRIX["runs"]["long"]
    out = []
    for v in MATRIX["variant"]:
        got = {r["steps"]: r for r in rows if r.get("variant") == v["id"] and r["exit"] == 0}
        if short in got and long in got:
            sps = (got[long]["wall_s"] - got[short]["wall_s"]) / (long - short)
            out.append({"variant": v["id"], "trainer": v["trainer"], "group": v["group"], "s_per_step": round(sps, 3),
                        "peak_gb": round(max(got[short]["peak_over_base_mb"], got[long]["peak_over_base_mb"]) / 1024, 1)})
    for r in out:
        print(f"{r['group']:42} {r['variant']:30} {r['s_per_step']:7.3f} s/step   {r['peak_gb']:5.1f} GB")
    return out


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "results"
    if cmd == "prepare":
        prepare()
    elif cmd == "run":
        steps = [int(sys.argv[3])] if len(sys.argv) > 3 else [MATRIX["runs"]["short"], MATRIX["runs"]["long"]]
        for s in steps:
            run(sys.argv[2], s)
    elif cmd == "all":
        only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
        # one trainer at a time, so the benchmark's ZenTrainer is started once and stopped once
        todo = [v for v in MATRIX["variant"] if not only or only in v["group"]]
        for v in sorted(todo, key=lambda v: v["trainer"] != "zentrainer"):
            for s in (MATRIX["runs"]["short"], MATRIX["runs"]["long"]):
                run(v["id"], s)
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
