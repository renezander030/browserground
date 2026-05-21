# browserground adapter for Skyvern

Replace (or augment) Skyvern's frontier-vision grounding step with the
local **browserground** specialist. Two patterns:

## 1. Replace — every grounding call goes local

```python
from browserground_skyvern import ground

result = ground(screenshot_path, target)
bbox = result["bbox_2d"]
# pass bbox into Skyvern's DOM-mapping stage
```

## 2. Augment — local first, cloud fallback on low confidence

```python
from browserground_skyvern import ground_with_fallback

def cloud_ground(img, tgt):
    # your existing Skyvern cloud-grounding entry point
    ...

bbox = ground_with_fallback(
    screenshot_path, target,
    confidence_threshold=0.55,
    cloud_fallback=cloud_ground,
)
```

The model returns a sequence-level confidence score (geometric mean of
per-token probabilities). Tune `confidence_threshold` against your
own eval set — start at 0.55 and adjust.

## Daemon mode (recommended for production)

```bash
browserground serve --http :8401 &
export BROWSERGROUND_HTTP=http://127.0.0.1:8401
```

The adapter now POSTs to the daemon (no subprocess spawn per call).

## What this saves

| | pure Skyvern (cloud vision) | with browserground |
|---|---|---|
| Vision tokens / step | 1500+ multimodal | 0 (local) |
| Cost / step | $0.01–0.05 | $0 |
| Screenshots leave machine | yes | no |

Recipe + numbers: [renezander030/browserground](https://huggingface.co/renezander030/browserground).
