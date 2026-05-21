"""browser-use ↔ browserground adapter.

Drop-in click-grounding for the `browser-use` framework. Replaces the
frontier-vision-model call that emits click coordinates with a local
2B specialist (`browserground`) that returns a strict JSON bbox.

Two integration modes:
  1. Subprocess shim (no extra deps) — calls the `browserground` CLI.
     Works with any browser-use version. ~1.8s/call cold, ~0.3s/call warm.

  2. HTTP daemon (zero per-call subprocess overhead) — start
     `browserground serve --http :8401` once; this adapter POSTs to it.

Usage with browser-use:

    from browserground_adapter import ground_bbox

    # Inside your custom browser-use action / tool:
    bbox = ground_bbox(screenshot_path="/tmp/page.png",
                       target="the green Subscribe button")
    if bbox:
        x = (bbox[0] + bbox[2]) // 2
        y = (bbox[1] + bbox[3]) // 2
        await page.mouse.click(x, y)

Set BROWSERGROUND_HTTP=http://127.0.0.1:8401 to use the daemon mode.
Otherwise the adapter shells out to the `browserground` CLI on PATH.
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


def _ground_via_http(image_path: str, target: str, base_url: str,
                     with_confidence: bool = False,
                     num_alternatives: int = 0) -> dict:
    import urllib.request
    body = json.dumps({
        "image_path": image_path,
        "target": target,
        "with_confidence": with_confidence,
        "num_alternatives": num_alternatives,
    }).encode()
    req = urllib.request.Request(
        base_url.rstrip("/") + "/api/ground",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode())


def _ground_via_cli(image_path: str, target: str, with_confidence: bool,
                    num_alternatives: int) -> dict:
    if not shutil.which("browserground"):
        raise BrowsergroundError(
            "browserground CLI not on PATH. Install: `npm install -g browserground`, "
            "or set BROWSERGROUND_HTTP to a running daemon URL."
        )
    cmd = ["browserground", "parse", image_path, "--target", target]
    if with_confidence:
        cmd.append("--confidence")
    if num_alternatives > 0:
        cmd += ["--alternatives", str(num_alternatives)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        raise BrowsergroundError(f"browserground failed: {r.stderr.strip()}")
    out = r.stdout.strip()
    try:
        return json.loads(out)
    except json.JSONDecodeError as e:
        raise BrowsergroundError(f"non-JSON output from browserground: {out!r}") from e


def ground(image_path: str, target: str, with_confidence: bool = False,
           num_alternatives: int = 0) -> dict:
    """Return the full grounding result dict.

    Result shape:
        {"bbox_2d": [x1, y1, x2, y2],
         "confidence": 0.92,            # if with_confidence=True
         "alternatives": [...]}         # if num_alternatives > 0
    """
    image_path = str(Path(image_path).expanduser().resolve())
    http_base = os.environ.get("BROWSERGROUND_HTTP")
    if http_base:
        res = _ground_via_http(image_path, target, http_base, with_confidence, num_alternatives)
        if not res.get("ok", True):
            raise BrowsergroundError(res.get("error", "unknown HTTP error"))
        return res
    return _ground_via_cli(image_path, target, with_confidence, num_alternatives)


def ground_bbox(screenshot_path: str, target: str) -> Optional[list[int]]:
    """Convenience wrapper: returns just the [x1,y1,x2,y2] list, or None on failure."""
    try:
        r = ground(screenshot_path, target)
    except BrowsergroundError:
        return None
    return r.get("bbox_2d")


def click_xy(screenshot_path: str, target: str) -> Optional[tuple[int, int]]:
    """Even more convenience: returns (center_x, center_y) ready for page.mouse.click."""
    b = ground_bbox(screenshot_path, target)
    if not b:
        return None
    return (b[0] + b[2]) // 2, (b[1] + b[3]) // 2


# -------------------------------------------------------------- browser-use ---
#
# Optional helper: a ready-made `browser_use` controller action.  Import only
# if the user has browser-use installed; otherwise this block is a no-op.

try:
    from browser_use import Controller  # type: ignore
    from playwright.async_api import Page  # type: ignore

    def register(controller: Controller):
        """Register `click_target` action on a browser-use Controller.

        Example:
            from browser_use import Agent, Controller
            controller = Controller()
            from browserground_adapter import register
            register(controller)
        """
        @controller.action("Locally ground and click a UI element on the page",
                           param_model=None)
        async def click_target(target: str, page: Page) -> str:
            shot = await page.screenshot(full_page=False)
            tmp = Path("/tmp/_bg_shot.png"); tmp.write_bytes(shot)
            xy = click_xy(str(tmp), target)
            if not xy:
                return f"browserground: could not locate '{target}'"
            await page.mouse.click(xy[0], xy[1])
            return f"clicked '{target}' at {xy}"

        return controller
except ImportError:
    # browser-use not installed; subprocess + HTTP entry points still work.
    pass


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="browser-use ↔ browserground adapter smoke test")
    p.add_argument("image"); p.add_argument("target")
    a = p.parse_args()
    print(json.dumps(ground(a.image, a.target), indent=2))
