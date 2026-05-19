#!/usr/bin/env node
import { spawn, spawnSync } from "node:child_process";
import { existsSync, mkdirSync } from "node:fs";
import { homedir } from "node:os";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dirname, "..");
const CACHE = process.env.BROWSERGROUND_CACHE || resolve(homedir(), ".cache", "browserground");
const PY_WORKER = resolve(ROOT, "src", "py", "worker.py");

function help() {
  console.log(`browserground — local UI grounding for AI agents

Usage:
  browserground parse <image> --target "<element description>"   parse one image
  browserground serve                                            run as daemon
  browserground stop                                             stop daemon
  browserground status                                           daemon + cache state

Examples:
  browserground parse screen.png --target "Submit button"
  browserground serve &
  browserground parse a.png --target "Chrome icon"
  browserground stop

Model auto-downloads on first call (~4.3 GB base + 67 MB adapter to ~/.cache/huggingface/).
Docs:    https://github.com/renezander030/browserground
Model:   https://huggingface.co/renezander030/browserground
`);
}

function pickPython() {
  for (const c of ["uv", "python3", "python"]) {
    const r = spawnSync(c, ["--version"], { stdio: "ignore" });
    if (r.status === 0) return c;
  }
  return null;
}

const args = process.argv.slice(2);
if (args.length === 0 || args[0] === "-h" || args[0] === "--help") {
  help();
  process.exit(0);
}

if (!existsSync(CACHE)) mkdirSync(CACHE, { recursive: true });
if (!existsSync(PY_WORKER)) {
  console.error(`[browserground] worker.py missing at ${PY_WORKER}`);
  console.error(`[browserground] reinstall: npm install -g browserground`);
  process.exit(2);
}

const py = pickPython();
if (!py) {
  console.error("[browserground] no python found. Install python>=3.10 or uv (brew install uv) then retry.");
  process.exit(2);
}

// uv runs python via `uv run python ...`; system python invokes directly
const cmd = py === "uv" ? [py, "run", "python", PY_WORKER, ...args] : [py, PY_WORKER, ...args];
const child = spawn(cmd[0], cmd.slice(1), {
  stdio: "inherit",
  env: { ...process.env, BROWSERGROUND_CACHE: CACHE },
});
child.on("exit", (code) => process.exit(code ?? 0));
