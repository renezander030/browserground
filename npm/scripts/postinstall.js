#!/usr/bin/env node
/**
 * Postinstall: verify Python + minimal deps are reachable, give clear errors otherwise.
 * We do NOT auto-install Python deps in a global npm postinstall — that's brittle and
 * surprising. Instead we tell the user exactly what to do.
 */
import { spawnSync } from "node:child_process";

const PY_REQS = ["torch", "transformers", "peft", "huggingface_hub", "PIL"];

function pickPython() {
  for (const c of ["uv", "python3", "python"]) {
    const r = spawnSync(c, ["--version"], { stdio: "ignore" });
    if (r.status === 0) return c;
  }
  return null;
}

function checkDeps(py) {
  const code = "import torch, transformers, peft, huggingface_hub, PIL";
  const cmd = py === "uv" ? ["run", "--quiet", "python", "-c", code] : ["-c", code];
  const r = spawnSync(py, cmd, { encoding: "utf8" });
  return r.status === 0;
}

const py = pickPython();
if (!py) {
  console.error("\n[browserground] WARN: no Python found in PATH.");
  console.error("[browserground] Inference requires Python ≥ 3.10 + torch + transformers + peft.");
  console.error("[browserground] Easiest: `brew install uv` (macOS) or `pip install uv` then re-run.\n");
  process.exit(0); // don't fail npm install — just warn
}

const ok = checkDeps(py);
if (!ok) {
  console.warn("\n[browserground] Python found (" + py + ") but missing inference deps.");
  console.warn("[browserground] Install them with one of:");
  console.warn("  uv pip install 'torch>=2.4' 'transformers>=4.55' 'peft>=0.13' huggingface_hub pillow");
  console.warn("  pip install 'torch>=2.4' 'transformers>=4.55' 'peft>=0.13' huggingface_hub pillow");
  console.warn("[browserground] On first `browserground parse`, the model (~4 GB base + 67 MB adapter) auto-downloads.\n");
  process.exit(0);
}

console.log("[browserground] ✓ Python + deps OK. Run `browserground parse <image> --target \"...\"` to use.");
