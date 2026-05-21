"""Minimal Python-only CLI. The full-featured CLI is the npm package
(`npm install -g browserground`), which adds daemon mode, HTTP server,
batch & eval subcommands. This `browserground-py` entry point is meant
for users in pure-Python environments who don't want Node."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import ground, __version__


def main():
    p = argparse.ArgumentParser(prog="browserground-py",
                                description="Local UI grounding (Python entry point).")
    p.add_argument("--version", action="version", version=f"browserground {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("parse", help="parse one screenshot")
    sp.add_argument("image")
    sp.add_argument("--target", required=True)
    sp.add_argument("--max-new-tokens", type=int, default=64)
    sp.add_argument("--confidence", action="store_true")

    args = p.parse_args()
    if args.cmd == "parse":
        r = ground(args.image, args.target,
                   max_new_tokens=args.max_new_tokens,
                   with_confidence=args.confidence)
        bbox = r.get("bbox_2d")
        if bbox is None:
            print(json.dumps(r), file=sys.stderr)
            sys.exit(1)
        out = {"bbox_2d": bbox}
        if "confidence" in r:
            out["confidence"] = r["confidence"]
        print(json.dumps(out))


if __name__ == "__main__":
    main()
