"""Which Python code launches which GPU work, from a profile's timeline (BENCH_PROFILE_TRACE=1):

    python3 profile/who.py results/profiles/<variant>.trace.json [kernel-name regex]

Kernel time (all, or matching the regex) totalled by the innermost frames of the launching thread
that aren't torch's own, biggest first; and gaps between kernels, by the code launching the next.
"""

import bisect
import collections
import json
import re
import sys

SKIP = ("torch/", "site-packages/torch", "<built-in", "contextlib", "threading", "_ops.py", "nn.Module:",
        "functools", "_compile.py", "autograd")


def main(path: str, pattern: str = "") -> None:
    ev = json.load(open(path))["traceEvents"]
    ks = sorted((e for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and "dur" in e), key=lambda e: e["ts"])
    launch = {e["args"]["correlation"]: e for e in ev if e.get("cat") == "cuda_runtime" and "correlation" in e.get("args", {})}
    by_tid = collections.defaultdict(list)
    for e in ev:
        if e.get("cat") == "python_function" and "dur" in e:
            by_tid[e["tid"]].append(e)
    starts = {}
    for t, es in by_tid.items():
        es.sort(key=lambda e: e["ts"])
        starts[t] = [e["ts"] for e in es]

    def where(k) -> str:
        l = launch.get(k["args"].get("correlation"))
        if not l:
            return "?"
        es, st = by_tid.get(l["tid"], []), starts.get(l["tid"], [])
        i = bisect.bisect_right(st, l["ts"])
        fs = [e["name"] for e in es[max(0, i - 3000):i] if e["ts"] <= l["ts"] <= e["ts"] + e["dur"]]
        mine = [f for f in fs if not any(s in f for s in SKIP)][-3:]
        return " > ".join(f.split("/")[-1] for f in mine) or "?"

    rx = re.compile(pattern, re.I) if pattern else None
    t, n = collections.Counter(), collections.Counter()
    gaps, gn = collections.Counter(), collections.Counter()
    end = ks[0]["ts"]
    for k in ks:
        gap = (k["ts"] - end) / 1e3
        w = None
        if gap > 0.05:
            w = where(k)
            gaps[w] += gap
            gn[w] += 1
        if rx is None or rx.search(k["name"]):
            w = w or where(k)
            t[w] += k["dur"] / 1e3
            n[w] += 1
        end = max(end, k["ts"] + k["dur"])
    print(f"kernels{' matching ' + pattern if pattern else ''}: {sum(t.values()):.1f} ms")
    for w, ms in t.most_common(20):
        print(f"{ms:8.1f} ms {n[w]:6}x  {w[:200]}")
    print(f"\ngaps over 0.05 ms: {sum(gaps.values()):.1f} ms")
    for w, ms in gaps.most_common(12):
        print(f"{ms:8.1f} ms {gn[w]:6}x  {w[:200]}")


if __name__ == "__main__":
    main(*sys.argv[1:])
