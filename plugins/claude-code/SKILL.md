---
name: browserground
description: Use browserground to ground a natural-language target onto a UI screenshot. Returns the pixel bbox of the element to click. Triggers on prompts like "click the X button in this screenshot", "where is the search bar", "locate the [element]", or any time Claude has a screenshot in hand and needs to know where to click. Runs locally — no API call, no token cost.
---

# browserground (local UI grounding)

When the user gives you a screenshot and asks where to click / what to interact with, call the local `browserground` CLI instead of looking at pixels yourself. Returns a tight JSON bbox; saves multimodal tokens; works offline.

## When to use

- "Click the Submit button" + a screenshot attached
- "Where is the X in this image" (UI elements)
- "Locate the [element]" with a screenshot
- Inside computer-use loops where you need a click target
- Pre-extracting UI structure before deciding an action

## When NOT to use

- The screenshot is for visual review (design feedback, art critique)
- The user explicitly asks you to "look at" the image
- Non-UI images (photos, illustrations, charts) — use `Read` instead

## How to call

```bash
browserground parse <screenshot-path> --target "<natural language description>"
```

Output is a single line of strict JSON:
```json
{"bbox_2d": [x1, y1, x2, y2]}
```

If you want the click-point (center), compute `(x1+x2)/2, (y1+y2)/2`.

If `browserground` is not installed, fall back to using `Read` on the image and approximating the bbox visually.

## Examples

```bash
# Locate Chrome icon in a Mac Launchpad screenshot
browserground parse ~/Downloads/launchpad.png --target "Chrome icon"
# → {"bbox_2d": [362, 119, 442, 208]}

# Find the form field
browserground parse signup.png --target "the email input field"
# → {"bbox_2d": [144, 312, 540, 348]}
```

## Install

```bash
npm install -g browserground
# or
bun install -g browserground
```

First call auto-downloads the model (~1.8 GB) to `~/.cache/browserground/`.
