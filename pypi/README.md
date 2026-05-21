# browserground (Python)

> **The local UI-grounding specialist for hybrid AI agents.**
> Drop in a screenshot + text target, get a strict JSON bbox.
> 2B params. MLX-native on Apple Silicon. Apache 2.0.

This is the Python entry point. For the full-featured CLI (daemon, HTTP
server, batch mode, eval), install the npm package:
`npm install -g browserground`.

## Install

```bash
# Apple Silicon (recommended) — uses the MLX 4-bit build, ~1-2s/call
pip install "browserground[mlx]"

# Or, CUDA / CPU (slower, ~10-14s/call on M-series via MPS)
pip install "browserground[transformers]"
```

## Use

```python
from browserground import ground, ground_bbox, click_xy

# Full result with timing + raw text
res = ground("screenshot.png", "the green Subscribe button")
print(res)
# {'bbox_2d': [344, 612, 478, 658], 'model_elapsed_s': 1.4, 'backend': 'mlx', ...}

# Just the bbox
bbox = ground_bbox("screenshot.png", "Submit button")

# Center coords for browser-use / Playwright / etc.
x, y = click_xy("screenshot.png", "the back arrow")
```

## How it works

`browserground` is a Qwen3-VL-2B base + a LoRA fine-tune for UI grounding
(rank 32, 26k training examples across macOS / Android / UIBert / web).
Output is **strict JSON** (`{"bbox_2d": [x1, y1, x2, y2]}`), **100% parseable**
on the held-out eval. **60.0% on ScreenSpot-v2** (300 items, vs SeeClick's
55.1% at 9.6B params — that's 4.8× smaller).

## browser-use / Skyvern integration

```python
from browserground import click_xy

# Inside your browser-use action:
xy = click_xy("/tmp/page.png", "the green Subscribe button")
if xy:
    await page.mouse.click(*xy)
```

Plug-in templates: <https://github.com/renezander030/browserground/tree/main/plugins>.

## Why this exists

Most agents send every screenshot to a frontier vision model just to find
click coordinates. That's a $0.01–0.05 multimodal call, 20–50× per run.
A 2B local specialist costs $0/call, runs on a laptop, doesn't send your
screenshots anywhere. The hybrid pattern: cheap fast local specialist
for the parser-style task, frontier model only for reasoning.

## Links

- **Recipe + numbers**: <https://huggingface.co/renezander030/browserground>
- **MLX build**: <https://huggingface.co/renezander030/browserground-mlx>
- **GGUF build**: <https://huggingface.co/renezander030/browserground-gguf>
- **GitHub**: <https://github.com/renezander030/browserground>
- **npm CLI** (daemon, HTTP, batch, eval): `npm install -g browserground`

License: Apache 2.0.
