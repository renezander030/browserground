"""Profile the browserground daemon: end-to-end latency across N images.

Measures:
  - cold first-call latency (model load + inference)
  - warm steady-state latency (p50, p95, mean) over N parses
  - per-stage timing if py-spy is available (flamegraph at out/flame.svg)

Usage:
    python profile-daemon.py <screenshots-dir> [--n 50] [--out report.json]

Assumes `browserground` (or `browserground-py`) is on PATH and the daemon
is running on the unix socket (`browserground serve &`).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import time
from pathlib import Path


def list_images(d: Path, limit: int = 50):
    out = []
    for ext in (".png", ".jpg", ".jpeg", ".webp"):
        out.extend(sorted(d.rglob(f"*{ext}")))
    return out[:limit]


def one_call(image: str, target: str) -> float:
    t0 = time.time()
    r = subprocess.run(
        ["browserground", "parse", image, "--target", target],
        capture_output=True, text=True, timeout=120,
    )
    dt = time.time() - t0
    if r.returncode != 0:
        return -1.0
    return dt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("screenshots")
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--target", default="the primary action button")
    p.add_argument("--out", default=None)
    p.add_argument("--flamegraph", action="store_true",
                   help="record py-spy flamegraph of the daemon during run (needs py-spy + sudo)")
    args = p.parse_args()

    imgs = list_images(Path(args.screenshots), args.n)
    if not imgs:
        raise SystemExit(f"no images under {args.screenshots}")

    # Cold call (start daemon if not up)
    is_up = subprocess.run(["browserground", "status"], capture_output=True, text=True)
    daemon_was_running = '"daemon_running": true' in is_up.stdout

    if not daemon_was_running:
        print("[profile] starting daemon (cold)…")
        d = subprocess.Popen(["browserground", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        # wait for socket
        for _ in range(120):
            time.sleep(1)
            r = subprocess.run(["browserground", "status"], capture_output=True, text=True)
            if '"daemon_running": true' in r.stdout:
                break

    pyspy_proc = None
    if args.flamegraph and shutil.which("py-spy"):
        # py-spy attaches to the daemon PID — find it from status
        st = subprocess.run(["browserground", "status"], capture_output=True, text=True)
        # very simple PID discovery via the worker pid file (browserground writes it)
        import json as _j
        try:
            info = _j.loads(st.stdout)
            cache = Path(info["cache"]); pid = int((cache / "worker.pid").read_text())
            print(f"[profile] py-spy record → out/flame.svg (daemon pid {pid})")
            Path("out").mkdir(exist_ok=True)
            pyspy_proc = subprocess.Popen(
                ["sudo", "py-spy", "record", "-o", "out/flame.svg", "-p", str(pid), "-d", str(max(args.n, 20))],
                stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
            )
        except Exception as e:
            print(f"[profile] could not attach py-spy: {e}")

    # First call (cold): the actual model_load happens server-side on `serve` start
    first = one_call(str(imgs[0]), args.target)
    print(f"[profile] first warm call: {first:.2f}s")

    # Subsequent calls
    durs = []
    for i, img in enumerate(imgs[1:], 1):
        d = one_call(str(img), args.target)
        if d > 0:
            durs.append(d)
        if i % 10 == 0:
            print(f"  [{i}/{len(imgs)-1}] last={d:.2f}s")

    if pyspy_proc:
        pyspy_proc.wait(timeout=60)

    report = {
        "n_images": len(imgs),
        "n_successful": len(durs),
        "first_call_s": round(first, 3),
        "mean_s": round(statistics.fmean(durs), 3) if durs else None,
        "p50_s": round(statistics.median(durs), 3) if durs else None,
        "p95_s": round(sorted(durs)[int(len(durs) * 0.95)], 3) if len(durs) > 20 else None,
        "min_s": round(min(durs), 3) if durs else None,
        "max_s": round(max(durs), 3) if durs else None,
    }
    print(json.dumps(report, indent=2))

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
