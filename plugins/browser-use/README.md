# browserground adapter for `browser-use`

Drop the cloud-vision call from your `browser-use` loop. Use the local
**browserground** specialist (2B params, ~1.8s/call, $0/call) for click
grounding. The frontier model stays in the loop only for the things it's
actually good at — deciding *what* to click, not *where* it is.

## Install

```bash
npm install -g browserground         # the CLI + worker
pip install browser-use playwright   # your existing browser-use install
```

Drop `browserground_adapter.py` next to your agent script.

## Use

### 1. Minimal: subprocess shim

```python
from browserground_adapter import click_xy

xy = click_xy("screenshot.png", "the green Subscribe button")
if xy:
    await page.mouse.click(*xy)
```

### 2. Register as a `browser-use` Controller action

```python
from browser_use import Agent, Controller
from browserground_adapter import register

controller = Controller()
register(controller)            # adds `click_target` action

agent = Agent(
    task="Subscribe to the newsletter",
    controller=controller,
    ...
)
```

The model now uses `click_target("the green Subscribe button")` instead of
asking the frontier vision model for pixel coordinates every step.

### 3. Daemon mode (zero subprocess overhead)

```bash
browserground serve --http :8401 &
```

```python
import os
os.environ["BROWSERGROUND_HTTP"] = "http://127.0.0.1:8401"
# adapter calls now POST to the daemon — first-call warm, ~0.3s/call subsequent
```

## What this saves

| | pure `browser-use` (GPT-4V) | with browserground |
|---|---|---|
| Vision tokens / step | 1500+ multimodal | 0 (local) |
| Cost / step | $0.01–0.05 | $0 |
| Latency / step | 800ms – 2s | ~1.8s cold, ~0.3s warm |
| Screenshots leave machine | yes | no |

Numbers + recipe: [renezander030/browserground](https://huggingface.co/renezander030/browserground).
