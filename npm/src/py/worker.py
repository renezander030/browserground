"""browserground inference worker.

Loads Qwen/Qwen3-VL-2B-Instruct + the browserground LoRA from HF, runs single-shot
or daemon-mode inference. Output is strict JSON: {"bbox_2d": [x1,y1,x2,y2]}.

Usage (typically wrapped by the Node CLI):
    python worker.py parse <image> --target "<description>"
    python worker.py serve            # unix-socket daemon
    python worker.py status
    python worker.py stop
"""
from __future__ import annotations

import argparse
import io
import json
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

    def parse(self, image_path: str, target: str, max_new_tokens: int = 64):
        from PIL import Image
        import torch

        img = Image.open(image_path).convert("RGB")
        w0, h0 = img.size
        scale = min(1.0, MAX_WIDTH / w0)
        if scale < 1.0:
            img = img.resize((int(w0 * scale), int(h0 * scale)))
        new_w, new_h = img.size

        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": f"Locate the element described: {target}"},
            ]},
        ]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[prompt], images=[[img]], return_tensors="pt").to(self.device)

        t0 = time.time()
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        elapsed = time.time() - t0
        gen = out[0, inputs.input_ids.shape[1]:]
        text = self.processor.tokenizer.decode(gen, skip_special_tokens=True).strip()

        bbox = parse_bbox(text)
        # Scale bbox BACK to original image coordinates if we resized
        if bbox and scale < 1.0:
            bbox = [int(c / scale) for c in bbox]

        return {
            "bbox_2d": bbox,
            "image_size": [w0, h0],
            "resized_size": [new_w, new_h],
            "model_elapsed_s": round(elapsed, 2),
            "raw_text": text,
        }


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


def cmd_serve(args):
    CACHE.mkdir(parents=True, exist_ok=True)
    if SOCKET_PATH.exists():
        SOCKET_PATH.unlink()
    PID_PATH.write_text(str(os.getpid()))

    w = _Worker(); w.load()

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
                img = req.get("image_path"); tgt = req.get("target", "")
                if not img or not Path(img).exists():
                    _send(conn, {"ok": False, "error": f"image not found: {img}"}); continue
                try:
                    res = w.parse(img, tgt, max_new_tokens=int(req.get("max_new_tokens", 64)))
                    _send(conn, {"ok": True, **res})
                except Exception as e:
                    _send(conn, {"ok": False, "error": str(e), "trace": traceback.format_exc()})
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


def cmd_parse(args):
    """Single-shot parse. If daemon is up, route through it. Otherwise load on demand."""
    img = str(Path(args.image).expanduser().resolve())
    if not Path(img).exists():
        print(json.dumps({"ok": False, "error": f"image not found: {img}"}), file=sys.stderr)
        return 2

    if _alive():
        # talk to the daemon
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); s.settimeout(600.0)
        s.connect(str(SOCKET_PATH))
        s.sendall((json.dumps({
            "cmd": "parse", "image_path": img, "target": args.target,
            "max_new_tokens": args.max_new_tokens,
        }) + "\n").encode())
        buf = b""
        while not buf.endswith(b"\n"):
            chunk = s.recv(65536)
            if not chunk: break
            buf += chunk
        s.close()
        resp = json.loads(buf.decode())
    else:
        # one-shot load
        w = _Worker(); w.load()
        try:
            r = w.parse(img, args.target, max_new_tokens=args.max_new_tokens)
            resp = {"ok": True, **r}
        except Exception as e:
            resp = {"ok": False, "error": str(e), "trace": traceback.format_exc()}

    if args.text:
        print(resp.get("raw_text", ""))
    elif resp.get("ok") and resp.get("bbox_2d"):
        # Strict-JSON output: just the bbox object
        print(json.dumps({"bbox_2d": resp["bbox_2d"]}))
    else:
        # Print full diagnostic on error
        print(json.dumps(resp), file=sys.stderr)
        return 1
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
    sp.add_argument("image")
    sp.add_argument("--target", required=True)
    sp.add_argument("--max-new-tokens", type=int, default=64)
    sp.add_argument("--text", action="store_true", help="print raw text instead of bbox JSON")
    sp.set_defaults(func=cmd_parse)
    sp = sub.add_parser("serve"); sp.set_defaults(func=cmd_serve)
    sp = sub.add_parser("status"); sp.set_defaults(func=cmd_status)
    sp = sub.add_parser("stop"); sp.set_defaults(func=cmd_stop)
    args = p.parse_args()
    rc = args.func(args)
    sys.exit(rc if isinstance(rc, int) else 0)


if __name__ == "__main__":
    main()
