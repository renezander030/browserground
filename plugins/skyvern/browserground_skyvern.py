"""Skyvern ↔ browserground adapter.

Skyvern's grounding stage maps a natural-language target (`"the blue Pay
button"`) to a DOM element via a frontier vision model. This adapter
replaces that call with the local `browserground` specialist.

Skyvern is plugin-friendly: you can either replace the grounding step
end-to-end, or augment it (use browserground as a cheap first pass and
fall back to the cloud only when confidence is low).

Two integration patterns:

  1. **Replace** (cheapest): every grounding call goes local.
  2. **Augment** (most robust): local first, escalate on low confidence.

Both use the public `ground(...)` function below.

Usage with Skyvern (high level):

    from browserground_skyvern import ground

    # In your custom Skyvern block / web-runner override:
    result = ground(screenshot_path, target,
                    with_confidence=True, num_alternatives=2)
    bbox = result["bbox_2d"]
    conf = result.get("confidence", 0.0)
    if conf < 0.55:
        # fall back to cloud grounding here
        ...
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional


class BrowsergroundError(RuntimeError):
    pass


def _via_http(image_path: str, target: str, base_url: str,
              with_confidence: bool, num_alternatives: int) -> dict:
    import urllib.request
    body = json.dumps({
        "image_path": image_path,
        "target": target,
        "with_confidence": with_confidence,
        "num_alternatives": num_alternatives,
    }).encode()
    req = urllib.request.Request(
        base_url.rstrip("/") + "/api/ground",
        data=body, headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode())


def _via_cli(image_path: str, target: str, with_confidence: bool,
             num_alternatives: int) -> dict:
    if not shutil.which("browserground"):
        raise BrowsergroundError(
            "browserground CLI not on PATH. `npm install -g browserground`."
        )
    cmd = ["browserground", "parse", image_path, "--target", target]
    if with_confidence:
        cmd.append("--confidence")
    if num_alternatives > 0:
        cmd += ["--alternatives", str(num_alternatives)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        raise BrowsergroundError(f"browserground failed: {r.stderr.strip()}")
    return json.loads(r.stdout.strip())


def ground(image_path: str, target: str, with_confidence: bool = False,
           num_alternatives: int = 0) -> dict:
    """Locate a UI element on a screenshot.

    Returns: {"bbox_2d": [x1, y1, x2, y2],
              "confidence": float?,
              "alternatives": [...]?}

    Set BROWSERGROUND_HTTP for daemon mode (zero subprocess overhead).
    """
    image_path = str(Path(image_path).expanduser().resolve())
    http_base = os.environ.get("BROWSERGROUND_HTTP")
    if http_base:
        res = _via_http(image_path, target, http_base, with_confidence, num_alternatives)
        if not res.get("ok", True):
            raise BrowsergroundError(res.get("error", "HTTP daemon error"))
        return res
    return _via_cli(image_path, target, with_confidence, num_alternatives)


def ground_with_fallback(image_path: str, target: str,
                         confidence_threshold: float = 0.55,
                         cloud_fallback=None) -> Optional[list[int]]:
    """Local-first grounding with optional cloud fallback.

    `cloud_fallback` is a callable(image_path, target) -> bbox or None.
    Provide your existing Skyvern cloud-grounding entry point.

    Returns the bbox or None.
    """
    try:
        r = ground(image_path, target, with_confidence=True, num_alternatives=0)
    except BrowsergroundError:
        return cloud_fallback(image_path, target) if cloud_fallback else None
    bbox = r.get("bbox_2d")
    conf = float(r.get("confidence", 0.0))
    if bbox and conf >= confidence_threshold:
        return bbox
    if cloud_fallback:
        return cloud_fallback(image_path, target)
    return bbox  # best-effort local answer if no fallback configured


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Skyvern ↔ browserground adapter smoke test")
    p.add_argument("image"); p.add_argument("target")
    p.add_argument("--confidence", action="store_true")
    p.add_argument("--alternatives", type=int, default=0)
    a = p.parse_args()
    print(json.dumps(ground(a.image, a.target,
                            with_confidence=a.confidence,
                            num_alternatives=a.alternatives), indent=2))
