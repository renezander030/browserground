<p align="center">
  <img src="https://raw.githubusercontent.com/renezander030/browserground/main/assets/logo.svg" alt="browserground logo" width="120" height="120"/>
</p>

<h1 align="center">browserground</h1>

<p align="center">
  <strong>The local UI-grounding specialist for hybrid AI agents.</strong><br/>
  Drop in a screenshot + text target, get a strict JSON bbox.
</p>

<p align="center">
  <a href="https://huggingface.co/renezander030/browserground"><img src="https://img.shields.io/badge/🤗-Model%20Card-yellow"/></a>
  <a href="https://github.com/renezander030/browserground"><img src="https://img.shields.io/badge/github-renezander030%2Fbrowserground-181717?logo=github"/></a>
</p>

---

## Why this exists — the hybrid AI argument

Today, most AI agents route **every** screenshot to a cloud frontier model (GPT-4V, Claude Vision, Gemini) just to find click coordinates. That's a $0.01–0.05 multimodal call adding 800ms–2s of latency, repeated 20–50× per agent run. Cost and latency compound. Screenshots full of private UI leave your machine.

A general 200B-parameter LLM is **overkill** for "where is the Submit button?" — that's a narrow vision task. The right shape is a **hybrid one**: cheap fast specialist local models for the dedicated tasks they handle better, and a cloud LLM only for the planning and reasoning it's uniquely good at.

That's exactly what **browserground** is — the click-grounding specialist you drop in next to your Claude / GPT-4o / Codex agent.

<p align="center">
  <img src="https://raw.githubusercontent.com/renezander030/browserground/main/assets/hybrid-architecture.svg" alt="Hybrid AI agent architecture: a local specialist grounds the click target; cloud LLM does planning only" width="900"/>
</p>

| | Pure-cloud agent | Hybrid (+ browserground) |
|---|---|---|
| Per-screenshot cost | $0.01–0.05 | **$0** |
| Latency | 800ms–2s round-trip | **~1.8s local** |
| Tokens billed by cloud | 1500+ multimodal | **~40 text** |
| Screenshots leave machine | yes | **no** |
| Rate limits | yes | **no** |

## What you get

```bash
browserground parse screenshot.png --target "Submit button"
# {"bbox_2d": [344, 612, 478, 658]}
```

Single-line strict-JSON bbox of the element to click. **100% format compliance** on the eval set — no markdown fences, no `<ref>` tokens, parseable every time.

## Install

```bash
npm install -g browserground
# or: bun install -g browserground
```

### Prerequisite (one-time)

The CLI shells out to Python for inference. Install once:

```bash
# Recommended: uv
brew install uv     # or: pipx install uv
uv pip install --system 'torch>=2.4' torchvision 'transformers>=4.55' 'peft>=0.13' huggingface_hub pillow

# Or pip:
pip install 'torch>=2.4' torchvision 'transformers>=4.55' 'peft>=0.13' huggingface_hub pillow
```

On first `browserground parse`, the model auto-downloads to `~/.cache/huggingface/`:

- base `Qwen/Qwen3-VL-2B-Instruct` (~4.3 GB)
- LoRA adapter `renezander030/browserground` (~67 MB)

## Use

```bash
# one-shot (loads model each call — slow first time)
browserground parse screenshot.png --target "Submit button"

# daemon mode (loads once, fast subsequent calls)
browserground serve &
browserground parse a.png --target "Chrome icon"
browserground parse b.png --target "the back arrow"
browserground stop

# raw model output (for debugging)
browserground parse screenshot.png --target "X" --text

# status
browserground status
```

## Hook into your agent stack

### Claude Code
```bash
mkdir -p .claude/skills/browserground
curl -sL https://raw.githubusercontent.com/renezander030/browserground/main/plugins/claude-code/SKILL.md \
  > .claude/skills/browserground/SKILL.md
```

### Codex CLI
Add to your `AGENTS.md` — spec at [plugins/codex/AGENTS.md](https://github.com/renezander030/browserground/blob/main/plugins/codex/AGENTS.md).

### browser-use / Skyvern (Python)
```python
import subprocess, json
def ground(screenshot_path, target):
    out = subprocess.check_output(["browserground", "parse", screenshot_path, "--target", target])
    return json.loads(out)["bbox_2d"]
```

## Benchmark

ScreenSpot-v2 point-grounding accuracy (300 items, 100/split):

| Model | Params | Overall | Mobile |
|---|---:|---:|---:|
| GPT-4o (cloud) | — | 18.3% | — |
| **browserground v0.1** | **2 B** | **45.3%** | **64.0%** |
| SeeClick | 9.6 B | 55.1% | — |
| ShowUI-2B | 2 B | 75.5% | — |
| UI-TARS-2B-SFT | 2 B | 89.5% | — |

v0.1 = one-epoch / 12k-example LoRA. v0.2 (Tier 2, target ≥ 60%) in development.

## Limitations

- v0.1 desktop & web accuracy lag mobile (training mix is mobile-heavy)
- English-only training data
- Single-target per call (batch mode in v0.2)
- No mouse-action prediction (only location — pair with an action predictor for full computer-use loops)

## Links

- 🤗 Model: <https://huggingface.co/renezander030/browserground>
- 💻 Source: <https://github.com/renezander030/browserground>
- 📦 npm: <https://www.npmjs.com/package/browserground>

## Work with me

Built by **Rene Zander**, freelance AI engineer (DE/EN, remote). Custom local specialists, hybrid AI agent architectures, on-prem LLM deployments — <https://renezander.com>.

## License

Apache 2.0.
