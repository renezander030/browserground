"""browserground inference worker.

Loads Qwen/Qwen3-VL-2B-Instruct + the browserground LoRA from HF, runs single-shot
or daemon-mode inference. Output is strict JSON: {"bbox_2d": [x1,y1,x2,y2]}.

Usage (typically wrapped by the Node CLI):
    python worker.py parse <image> --target "<description>"
    python worker.py parse <image> --targets targets.txt   # batch mode
    python worker.py serve                                  # unix-socket daemon
    python worker.py serve --http :8401                     # HTTP daemon (REST)
    python worker.py eval <dir-or-images.json> <targets.json>
    python worker.py status
    python worker.py stop
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import re
import signal
import socket
import sys
import time
import traceback
from pathlib import Path

CACHE = Path(os.environ.get("BROWSERGROUND_CACHE", Path.home() / ".cache" / "browserground"))
SOCKET_PATH = CACHE / "worker.sock"
PID_PATH = CACHE / "worker.pid"
LOG_PATH = CACHE / "worker.log"

ADAPTER_REPO = os.environ.get("BROWSERGROUND_ADAPTER", "renezander030/browserground")
BASE_MODEL = os.environ.get("BROWSERGROUND_BASE", "Qwen/Qwen3-VL-2B-Instruct")
MAX_WIDTH = int(os.environ.get("BROWSERGROUND_MAX_WIDTH", "1024"))

SYSTEM_PROMPT = (
    "You are a UI-grounding model. Given a screenshot and a target description, output the "
    "bounding box of the SINGLE UI element to click. Output ONLY a JSON object: "
    '{"bbox_2d": [x1, y1, x2, y2]} with pixel coordinates, origin at top-left.'
)

_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def parse_bbox(text: str):
    """Extract bbox_2d from model output, tolerating fences and other wrappings."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        d = json.loads(text)
        if isinstance(d, dict):
            for k in ("bbox_2d", "bbox", "box"):
                v = d.get(k)
                if isinstance(v, list) and len(v) >= 4:
                    return [int(float(x)) for x in v[:4]]
    except Exception:
        pass
    nums = _NUM.findall(text)
    if len(nums) >= 4:
        return [int(float(n)) for n in nums[:4]]
    return None


def bbox_center(b):
    return [(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0]


def point_in_box(p, b):
    return b[0] <= p[0] <= b[2] and b[1] <= p[1] <= b[3]


class _Worker:
    def __init__(self):
        self.processor = None
        self.model = None
        self.device = None

    def load(self):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        from peft import PeftModel

        print(f"[worker] loading base {BASE_MODEL}", flush=True)
        t0 = time.time()
        dtype = torch.bfloat16
        self.processor = AutoProcessor.from_pretrained(BASE_MODEL)
        m = Qwen3VLForConditionalGeneration.from_pretrained(BASE_MODEL, dtype=dtype, device_map="auto")
        print(f"[worker] loading adapter {ADAPTER_REPO}", flush=True)
        m = PeftModel.from_pretrained(m, ADAPTER_REPO)
        m = m.merge_and_unload()
        m.eval()
        self.model = m
        self.device = next(m.parameters()).device
        print(f"[worker] ready in {time.time()-t0:.1f}s on {self.device}", flush=True)

    def _prep_image(self, image_path):
        from PIL import Image
        img = Image.open(image_path).convert("RGB")
        w0, h0 = img.size
        scale = min(1.0, MAX_WIDTH / w0)
        if scale < 1.0:
            img = img.resize((int(w0 * scale), int(h0 * scale)))
        return img, (w0, h0), scale

    def _build_inputs(self, img, target):
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": f"Locate the element described: {target}"},
            ]},
        ]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return self.processor(text=[prompt], images=[[img]], return_tensors="pt").to(self.device)

    def parse(self, image_path: str, target: str, max_new_tokens: int = 64,
              with_confidence: bool = False, num_alternatives: int = 0):
        """Run inference. Returns dict with bbox_2d, optionally confidence + alternatives.

        Confidence is the mean per-token probability of the chosen output sequence
        (geometric-mean log-prob), in [0, 1].

        Alternatives: top-K alternate bbox guesses via diverse beam search.
        """
        import torch

        img, (w0, h0), scale = self._prep_image(image_path)
        new_w, new_h = img.size
        inputs = self._build_inputs(img, target)

        t0 = time.time()
        gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=False)
        if with_confidence:
            gen_kwargs.update(return_dict_in_generate=True, output_scores=True)

        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)
        elapsed = time.time() - t0

        if with_confidence:
            seqs = out.sequences
            scores = out.scores  # tuple of (batch, vocab) per gen step
            gen_ids = seqs[0, inputs.input_ids.shape[1]:]
            text = self.processor.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            # mean log-prob of the chosen sequence
            logps = []
            for step_logits, tok in zip(scores, gen_ids):
                lp = torch.log_softmax(step_logits[0].float(), dim=-1)
                logps.append(float(lp[tok].item()))
            if logps:
                mean_lp = sum(logps) / len(logps)
                confidence = float(math.exp(mean_lp))
            else:
                confidence = 0.0
        else:
            gen_ids = out[0, inputs.input_ids.shape[1]:]
            text = self.processor.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            confidence = None

        bbox = parse_bbox(text)
        if bbox and scale < 1.0:
            bbox = [int(c / scale) for c in bbox]

        result = {
            "bbox_2d": bbox,
            "image_size": [w0, h0],
            "resized_size": [new_w, new_h],
            "model_elapsed_s": round(elapsed, 2),
            "raw_text": text,
        }
        if confidence is not None:
            result["confidence"] = round(confidence, 4)

        # Alternatives via diverse beam search (separate generate call).
        if num_alternatives > 0:
            try:
                with torch.no_grad():
                    alt_out = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        num_beams=max(num_alternatives + 1, 3),
                        num_beam_groups=max(num_alternatives + 1, 3),
                        diversity_penalty=0.8,
                        num_return_sequences=num_alternatives + 1,
                        do_sample=False,
                        return_dict_in_generate=True,
                        output_scores=False,
                    )
                alts = []
                for seq in alt_out.sequences[1:num_alternatives + 1]:
                    sub = seq[inputs.input_ids.shape[1]:]
                    txt = self.processor.tokenizer.decode(sub, skip_special_tokens=True).strip()
                    bb = parse_bbox(txt)
                    if bb and scale < 1.0:
                        bb = [int(c / scale) for c in bb]
                    if bb is not None and bb != bbox:
                        alts.append({"bbox_2d": bb, "raw_text": txt})
                result["alternatives"] = alts
            except Exception as e:
                result["alternatives_error"] = str(e)

        return result


# ---------------------------------------------------------------- daemon ---

def _read_line(conn):
    buf = b""
    while not buf.endswith(b"\n"):
        chunk = conn.recv(65536)
        if not chunk: break
        buf += chunk
    return buf


def _send(conn, payload):
    conn.sendall((json.dumps(payload) + "\n").encode("utf-8"))


def _handle_parse_req(w: _Worker, req: dict) -> dict:
    img = req.get("image_path"); tgt = req.get("target", "")
    if not img or not Path(img).exists():
        return {"ok": False, "error": f"image not found: {img}"}
    try:
        res = w.parse(
            img, tgt,
            max_new_tokens=int(req.get("max_new_tokens", 64)),
            with_confidence=bool(req.get("with_confidence", False)),
            num_alternatives=int(req.get("num_alternatives", 0)),
        )
        return {"ok": True, **res}
    except Exception as e:
        return {"ok": False, "error": str(e), "trace": traceback.format_exc()}


def _serve_http(w: _Worker, bind: str):
    """Minimal HTTP daemon. Routes:
       GET  /api/health        → {"ok": true, "adapter": "..."}
       POST /api/ground        → body {image_path, target, with_confidence?, num_alternatives?, max_new_tokens?}
    """
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    host, _, port = bind.partition(":")
    host = host or "127.0.0.1"
    port = int(port or "8401")

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a, **kw): pass

        def _json(self, code, payload):
            data = (json.dumps(payload) + "\n").encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/api/health":
                return self._json(200, {"ok": True, "ready": True, "adapter": ADAPTER_REPO, "base": BASE_MODEL})
            return self._json(404, {"ok": False, "error": "not found"})

        def do_POST(self):
            if self.path != "/api/ground":
                return self._json(404, {"ok": False, "error": "not found"})
            n = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(n) if n else b"{}"
            try:
                req = json.loads(body.decode())
            except Exception as e:
                return self._json(400, {"ok": False, "error": f"bad json: {e}"})
            resp = _handle_parse_req(w, req)
            return self._json(200 if resp.get("ok") else 400, resp)

    srv = ThreadingHTTPServer((host, port), H)
    print(f"[worker] HTTP listening http://{host}:{port}", flush=True)

    def _sig(*_):
        try: srv.shutdown()
        except: pass
    signal.signal(signal.SIGTERM, _sig); signal.signal(signal.SIGINT, _sig)
    try:
        srv.serve_forever()
    finally:
        srv.server_close()


def cmd_serve(args):
    CACHE.mkdir(parents=True, exist_ok=True)
    w = _Worker(); w.load()

    if getattr(args, "http", None):
        return _serve_http(w, args.http)

    if SOCKET_PATH.exists():
        SOCKET_PATH.unlink()
    PID_PATH.write_text(str(os.getpid()))

    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(str(SOCKET_PATH)); SOCKET_PATH.chmod(0o600); srv.listen(8)
    print(f"[worker] listening {SOCKET_PATH}", flush=True)

    stop = False
    def _sig(*_):
        nonlocal stop; stop = True
        try: srv.close()
        except: pass
    signal.signal(signal.SIGTERM, _sig); signal.signal(signal.SIGINT, _sig)

    while not stop:
        try: conn, _ = srv.accept()
        except OSError: break
        try:
            line = _read_line(conn)
            if not line: conn.close(); continue
            req = json.loads(line.decode())
            cmd = req.get("cmd", "parse")
            if cmd == "ping":
                _send(conn, {"ok": True, "ready": True, "adapter": ADAPTER_REPO})
            elif cmd == "shutdown":
                _send(conn, {"ok": True, "shutting_down": True}); stop = True
            elif cmd == "parse":
                _send(conn, _handle_parse_req(w, req))
            else:
                _send(conn, {"ok": False, "error": f"unknown cmd: {cmd}"})
        except Exception as e:
            try: _send(conn, {"ok": False, "error": str(e)})
            except: pass
        finally:
            try: conn.close()
            except: pass

    for p in (SOCKET_PATH, PID_PATH):
        try: p.unlink()
        except FileNotFoundError: pass


def _alive():
    if not PID_PATH.exists() or not SOCKET_PATH.exists(): return False
    try:
        pid = int(PID_PATH.read_text().strip()); os.kill(pid, 0); return True
    except (ValueError, OSError): return False


def cmd_status(args):
    CACHE.mkdir(parents=True, exist_ok=True)
    info = {
        "daemon_running": _alive(),
        "socket": str(SOCKET_PATH),
        "cache": str(CACHE),
        "adapter": ADAPTER_REPO,
        "base": BASE_MODEL,
    }
    print(json.dumps(info, indent=2))


def _call_daemon(req: dict) -> dict:
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); s.settimeout(600.0)
    s.connect(str(SOCKET_PATH))
    s.sendall((json.dumps(req) + "\n").encode())
    buf = b""
    while not buf.endswith(b"\n"):
        chunk = s.recv(65536)
        if not chunk: break
        buf += chunk
    s.close()
    return json.loads(buf.decode())


def _load_targets(path: str):
    """Load batch targets. Supports two formats:

       1. plain text — one target per line, paired with single --image argument
          (returns list of (image, target) by zipping the single image to each line)
       2. JSON list — [{"image": "...", "target": "..."}, ...]
    """
    p = Path(path).expanduser().resolve()
    text = p.read_text()
    text_stripped = text.strip()
    if text_stripped.startswith("[") or text_stripped.startswith("{"):
        data = json.loads(text)
        if isinstance(data, dict):
            data = [data]
        out = []
        for row in data:
            img = row.get("image") or row.get("image_path")
            tgt = row.get("target") or row.get("prompt")
            if img and tgt:
                out.append((str(Path(img).expanduser().resolve()), tgt, row))
        return out, "json"
    # plain text
    targets = [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.startswith("#")]
    return targets, "text"


def cmd_parse(args):
    """Single-shot OR batch parse. If daemon is up, route through it; else load on demand."""
    use_daemon = _alive()
    w = None
    if not use_daemon:
        w = _Worker(); w.load()

    def _one(image_path: str, target: str) -> dict:
        if use_daemon:
            return _call_daemon({
                "cmd": "parse",
                "image_path": image_path,
                "target": target,
                "max_new_tokens": args.max_new_tokens,
                "with_confidence": bool(args.confidence),
                "num_alternatives": int(args.alternatives or 0),
            })
        try:
            r = w.parse(
                image_path, target,
                max_new_tokens=args.max_new_tokens,
                with_confidence=bool(args.confidence),
                num_alternatives=int(args.alternatives or 0),
            )
            return {"ok": True, **r}
        except Exception as e:
            return {"ok": False, "error": str(e), "trace": traceback.format_exc()}

    # ----- Batch mode -----
    if args.targets:
        items, mode = _load_targets(args.targets)
        results = []
        if mode == "text":
            if not args.image:
                print(json.dumps({"ok": False, "error": "--targets <text-file> requires <image> arg"}), file=sys.stderr)
                return 2
            img = str(Path(args.image).expanduser().resolve())
            for tgt in items:
                r = _one(img, tgt)
                row = {"image": img, "target": tgt, **r}
                results.append(row)
                if args.jsonl:
                    print(json.dumps(row), flush=True)
        else:  # json
            for img, tgt, src in items:
                r = _one(img, tgt)
                row = {"image": img, "target": tgt, **r}
                results.append(row)
                if args.jsonl:
                    print(json.dumps(row), flush=True)
        if not args.jsonl:
            print(json.dumps(results, indent=2))
        return 0

    # ----- Single-shot -----
    if not args.image or not args.target:
        print(json.dumps({"ok": False, "error": "need <image> and --target (or --targets)"}), file=sys.stderr)
        return 2
    img = str(Path(args.image).expanduser().resolve())
    if not Path(img).exists():
        print(json.dumps({"ok": False, "error": f"image not found: {img}"}), file=sys.stderr)
        return 2
    resp = _one(img, args.target)

    if args.text:
        print(resp.get("raw_text", ""))
    elif args.confidence or args.alternatives:
        # Extended JSON when user opts in
        keep = {k: resp[k] for k in ("bbox_2d", "confidence", "alternatives") if k in resp}
        if not resp.get("ok"):
            keep = resp
        print(json.dumps(keep))
    elif resp.get("ok") and resp.get("bbox_2d"):
        print(json.dumps({"bbox_2d": resp["bbox_2d"]}))
    else:
        print(json.dumps(resp), file=sys.stderr)
        return 1
    return 0


def cmd_eval(args):
    """Run an eval set and report hit-rate, format-OK, mean latency.

    targets.json schema: list of {"image": "...", "target": "...", "bbox": [x1,y1,x2,y2]}
    A "hit" is the predicted bbox's CENTER falling inside the ground-truth bbox.
    """
    items_path = Path(args.targets).expanduser().resolve()
    items = json.loads(items_path.read_text())
    if isinstance(items, dict):
        items = [items]

    # If <data> is a dir, prepend it to image paths that aren't already absolute.
    base_dir = None
    if args.data_dir:
        base_dir = Path(args.data_dir).expanduser().resolve()

    use_daemon = _alive()
    w = None
    if not use_daemon:
        w = _Worker(); w.load()

    n = len(items); hits = 0; format_ok = 0; latencies = []; per_item = []

    for i, row in enumerate(items):
        img = row.get("image") or row.get("image_path")
        if base_dir and not Path(img).is_absolute():
            img = str(base_dir / img)
        img = str(Path(img).expanduser().resolve())
        tgt = row.get("target") or row.get("prompt")
        gt = row.get("bbox") or row.get("ground_truth") or row.get("gt")

        if not Path(img).exists():
            per_item.append({"i": i, "image": img, "ok": False, "error": "image not found"})
            continue

        if use_daemon:
            r = _call_daemon({"cmd": "parse", "image_path": img, "target": tgt, "max_new_tokens": args.max_new_tokens})
        else:
            try:
                rr = w.parse(img, tgt, max_new_tokens=args.max_new_tokens)
                r = {"ok": True, **rr}
            except Exception as e:
                r = {"ok": False, "error": str(e)}

        if r.get("ok"):
            if r.get("model_elapsed_s") is not None:
                latencies.append(r["model_elapsed_s"])
            pred = r.get("bbox_2d")
            if pred:
                format_ok += 1
                if gt and point_in_box(bbox_center(pred), gt):
                    hits += 1
            per_item.append({
                "i": i, "image": img, "target": tgt,
                "pred": pred, "gt": gt,
                "hit": bool(pred and gt and point_in_box(bbox_center(pred), gt)),
                "latency_s": r.get("model_elapsed_s"),
            })
        else:
            per_item.append({"i": i, "image": img, "target": tgt, "ok": False, "error": r.get("error")})

        if i % 10 == 0 or i == n - 1:
            sys.stderr.write(f"[eval] {i+1}/{n}  hits={hits}  format_ok={format_ok}\n")
            sys.stderr.flush()

    summary = {
        "n": n,
        "hits": hits,
        "accuracy": round(hits / n, 4) if n else 0.0,
        "format_ok": format_ok,
        "format_ok_pct": round(format_ok / n, 4) if n else 0.0,
        "mean_latency_s": round(sum(latencies) / len(latencies), 3) if latencies else None,
        "p50_latency_s": round(sorted(latencies)[len(latencies)//2], 3) if latencies else None,
        "p95_latency_s": round(sorted(latencies)[int(len(latencies)*0.95)], 3) if latencies else None,
    }
    out = {"summary": summary, "items": per_item}
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(json.dumps(summary, indent=2))
    else:
        print(json.dumps(out, indent=2))
    return 0


def cmd_stop(args):
    if not _alive():
        print(json.dumps({"ok": True, "was_running": False}))
        return 0
    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); s.settimeout(5.0)
        s.connect(str(SOCKET_PATH))
        s.sendall(b'{"cmd":"shutdown"}\n')
        s.recv(65536); s.close()
    except Exception: pass
    print(json.dumps({"ok": True, "stopped": True}))


def main():
    p = argparse.ArgumentParser(prog="browserground")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("parse")
    sp.add_argument("image", nargs="?")
    sp.add_argument("--target", default=None)
    sp.add_argument("--targets", default=None, help="batch: text file (one target/line) or JSON list of {image,target}")
    sp.add_argument("--max-new-tokens", type=int, default=64)
    sp.add_argument("--text", action="store_true", help="print raw text instead of bbox JSON")
    sp.add_argument("--confidence", action="store_true", help="include sequence-level confidence in output")
    sp.add_argument("--alternatives", type=int, default=0, help="emit N diverse alternate bboxes")
    sp.add_argument("--jsonl", action="store_true", help="batch mode: emit one JSON object per line as work streams")
    sp.set_defaults(func=cmd_parse)

    sp = sub.add_parser("serve")
    sp.add_argument("--http", default=None, metavar="HOST:PORT",
                    help="serve HTTP REST on HOST:PORT instead of unix socket (e.g. :8401)")
    sp.set_defaults(func=cmd_serve)

    sp = sub.add_parser("status"); sp.set_defaults(func=cmd_status)
    sp = sub.add_parser("stop"); sp.set_defaults(func=cmd_stop)

    sp = sub.add_parser("eval", help="evaluate on a targets.json with ground-truth bboxes")
    sp.add_argument("data_dir", nargs="?", default=None,
                    help="optional dir to resolve relative image paths against")
    sp.add_argument("targets", help="JSON list of {image, target, bbox}")
    sp.add_argument("--max-new-tokens", type=int, default=64)
    sp.add_argument("--out", default=None, help="write full per-item report to this path")
    sp.set_defaults(func=cmd_eval)

    args = p.parse_args()
    rc = args.func(args)
    sys.exit(rc if isinstance(rc, int) else 0)


if __name__ == "__main__":
    main()
