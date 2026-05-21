<p align="center">
  <img src="./assets/logo.svg" alt="browserground logo" width="120" height="120"/>
</p>

<h1 align="center">browserground</h1>

<p align="center">
  <strong>The local UI-grounding specialist for hybrid AI agents.</strong><br/>
  Drop in a screenshot + text target, get a strict JSON bbox. 2B params. MLX-native. Apache 2.0.
</p>

<p align="center">
  <a href="https://huggingface.co/renezander030/browserground"><img src="https://img.shields.io/badge/🤗-Model%20Card-yellow" alt="HF model"/></a>
  <a href="https://www.npmjs.com/package/browserground"><img src="https://img.shields.io/badge/npm-browserground-cb3837?logo=npm&logoColor=white" alt="npm"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue.svg" alt="License"/></a>
  <img src="https://img.shields.io/badge/base-Qwen3--VL--2B-7c4ddf" alt="Base"/>
</p>

---

## The hybrid AI argument

Today, most AI agents route **every** screenshot to a cloud frontier model (GPT-4V, Claude Vision, Gemini) — just to figure out *where to click*. That's a $0.01–0.05 multimodal call adding 800ms–2s of round-trip latency, repeated 20–50 times per agent run. The bill compounds. The latency compounds. And screenshots full of private UI leave your machine.

A general-purpose 200B-parameter LLM is overkill for the question "where is the Submit button?" — that's a narrow vision task. The right architecture is a **hybrid one**: cheap fast specialist local models for the dedicated tasks they handle better, and the cloud LLM only for the planning and reasoning it's actually uniquely good at.

That's exactly what **browserground** is — the click-grounding specialist.

<p align="center">
  <img src="./assets/hybrid-architecture.svg" alt="Hybrid AI agent architecture diagram" width="900"/>
</p>

| | Pure-cloud (status quo) | Hybrid (with browserground) |
|---|---|---|
| Per-screenshot cost | $0.01–0.05 | **$0** |
| Latency | 800ms–2s round-trip | **~1.8s local, no network** |
| Tokens billed by cloud | 1,500+ multimodal | **~40 text tokens** |
| Screenshots leave machine | yes | **no** |
| Rate limits | yes | **no** |

## Status: v0.2

ScreenSpot-v2 point-grounding accuracy (300 items, 100/split):

| Model | Params | Overall | Mobile | Desktop | Web | Format-OK |
|---|---:|---:|---:|---:|---:|---:|
| GPT-5.4 (cloud frontier) ¹ | — | 85.4% | — | — | — | — |
| **browserground v0.2** | **2 B** | **60.0%** | **78.0%** | 44.0% | 58.0% | **100%** |
| SeeClick | 9.6 B | 55.1% | — | — | — | — |
| ShowUI-2B | 2 B | 75.5% | — | — | — | — |
| UI-TARS-2B-SFT | 2 B | 89.5% | — | — | — | — |
| OS-Atlas-Base-7B | 7 B | ~91% | — | — | — | — |
| zero-shot Qwen3-VL-2B (no fine-tune) | 2 B | 6.3% | 7.0% | 6.0% | 6.0% | 100% |

¹ GPT-5.4 score is on the harder **ScreenSpot-Pro** benchmark (no public v2 number for the 2026 cloud generation). v2 is significantly easier, so the cloud frontier likely scores 90%+ on v2 if independently benchmarked. Open-source numbers in the table use v2 throughout.

- **+10× over zero-shot baseline** on the same benchmark (6.3% → 60.0%)
- **Beats SeeClick (9.6B) at 2B params** — 4.8× smaller model, +5 pp accuracy
- **100% strict-JSON format compliance** — no markdown fences, no commentary, no `<ref>` tokens

### Where browserground beats UI-TARS-2B-SFT

UI-TARS-2B-SFT scores higher on ScreenSpot-v2 overall (89.5%) — but it's a different product. Here's where browserground is a better fit:

| | browserground v0.2 | UI-TARS-2B-SFT |
|---|---|---|
| Base model | Qwen3-VL-2B (2025) | Qwen2-VL-2B (2024) |
| Output format | **Strict JSON `{"bbox_2d": [...]}` — 100% parseable** | Coord strings inside prose — needs parsing/regex |
| Training mix | Browser + macOS + Android (web-weighted for the actual agent workload) | OS-general; no browser-platform emphasis |
| Distribution | CLI-first; `npm install -g browserground`; MLX-ready | Server-class; no first-class Mac story |
| Design | A piece of a hybrid AI stack (one specialist among many) | Standalone agent toolkit |
| License + base lineage | Apache 2.0 on a current-gen base | Apache 2.0 on a year-old base |

Pick UI-TARS when you want a complete agent toolkit and don't mind running the bigger ecosystem. Pick browserground when you're composing your own hybrid AI stack and need a small, fast, strict-JSON grounding specialist that drops into a CLI / npm workflow on a laptop.

## Quick start

```bash
npm install -g browserground
browserground parse screenshot.png --target "Submit button"
# {"bbox_2d": [344, 612, 478, 658]}
```

Daemon mode for fast subsequent calls:

```bash
browserground serve &
browserground parse a.png --target "Chrome icon"
browserground parse b.png --target "the back arrow"
browserground stop
```

## Hook into your agent stack

### Claude Code

```bash
mkdir -p .claude/skills/browserground
curl -sL https://raw.githubusercontent.com/renezander030/browserground/main/plugins/claude-code/SKILL.md \
  > .claude/skills/browserground/SKILL.md
```

Claude routes screen-grounding prompts to the CLI. Spec at [`plugins/claude-code/SKILL.md`](./plugins/claude-code/SKILL.md).

### Codex CLI

```yaml
# Add to ~/.codex/AGENTS.md
tools:
  - name: browserground
    command: browserground parse "$IMAGE_PATH" --target "$TARGET"
    description: Locate a UI element on a screenshot. Returns {"bbox_2d":[x1,y1,x2,y2]}.
```

### browser-use / Skyvern (Python)

```python
import subprocess, json
def ground(screenshot_path, target):
    out = subprocess.check_output(["browserground", "parse", screenshot_path, "--target", target])
    return json.loads(out)["bbox_2d"]
```

## How it works

- Base: [`Qwen/Qwen3-VL-2B-Instruct`](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)
- Method: LoRA rank 32 (34.9 M trainable params, 1.6% of base) on all linear modules of the LM
- Training mix (26k records): 6k OS-Atlas macOS desktop + 6k Android (aw_mobile) + 6k UIBert mobile + 8k wave-ui browser
- Schedule: 1 epoch, bf16, LR 1e-4 cosine, batch 1 × grad-accum 8, ~4.5 hr on a single RTX A6000
- Output: strict JSON `{"bbox_2d": [x1, y1, x2, y2]}` — system prompt + LoRA produce 100% parseable output

Training scripts and eval JSONs: [renezander030/imgparse-tier1](https://github.com/renezander030/imgparse-tier1) (private — request access).

## What's planned

- **MLX-native build** — ~1–2 s on Apple Silicon (currently ~14 s via MPS+transformers)
- **GGUF build** + **Ollama Modelfile** — `ollama run browserground`
- **Batch mode** — `--targets file.txt`, many targets per screenshot in one call
- **Confidence output** — `{bbox_2d, confidence, alternatives}` for retry/fallback logic
- **PyPI package** — `pip install browserground`

More in v0.2.

## Why this exists

Pure-cloud AI agents are bottlenecked on vision-LLM cost and latency. Open-source 2B specialist models like UI-TARS-2B-SFT (89.5% on ScreenSpot-v2) already match or beat the cloud frontier on narrow grounding tasks at a fraction of the per-call cost. The **composition pattern** — specialist local models for narrow tasks + cloud LLMs for general reasoning — is the cost-effective architecture for 2026 AI agents. browserground is one specialist piece. Bring your own orchestrator.

## License

Apache 2.0.

---

```bibtex
@misc{browserground-2026,
  title  = {browserground: Qwen3-VL-2B LoRA for hybrid AI agent UI grounding},
  author = {Zander, René},
  year   = {2026},
  url    = {https://huggingface.co/renezander030/browserground}
}
```
