"""browserground — local UI-grounding specialist for hybrid AI agents.

Drop in a screenshot + a text description of the element you want to click,
get back a strict JSON bbox. Local. Fast. Apache 2.0.

Quick start:

    from browserground import ground

    res = ground("screenshot.png", "the green Subscribe button")
    print(res["bbox_2d"])

Backends:
    - MLX (Apple Silicon, ~1-2s/call):     pip install "browserground[mlx]"
    - transformers (everywhere, ~10-14s):  pip install "browserground[transformers]"

The library auto-detects which backend is available. Set
BROWSERGROUND_BACKEND=mlx|transformers to force.

For the CLI experience (daemon, http daemon, batch, eval), use the npm
package instead: `npm install -g browserground`.
"""
from .core import ground, ground_bbox, click_xy, BrowsergroundError

__version__ = "0.3.0"
__all__ = ["ground", "ground_bbox", "click_xy", "BrowsergroundError", "__version__"]
