# browserground — Codex CLI extension

Local UI grounding model for screen-element location. Use when the user has a screenshot and needs to know the bbox of a specific element.

## Install

```bash
npm install -g browserground   # or: bun install -g browserground
```

## Tool registration

Add to your Codex config (`~/.codex/AGENTS.md` user-level, or `./AGENTS.md` project-level):

```yaml
tools:
  - name: browserground
    command: browserground parse "$IMAGE_PATH" --target "$TARGET"
    description: Locate a UI element on a screenshot. Returns {"bbox_2d": [x1,y1,x2,y2]} in pixel coordinates.
    parameters:
      IMAGE_PATH: absolute path to the screenshot
      TARGET: natural-language description of the element to locate
```

## When Codex calls it

Codex routes screen-grounding tasks here automatically when the system prompt mentions UI / clicks / element location and a screenshot is in scope. No model API call — runs locally on Apple Silicon via MLX or any CUDA GPU.

## Manual invocation

```
codex run browserground --target "the dark mode toggle" --image ~/Pictures/settings.png
```
