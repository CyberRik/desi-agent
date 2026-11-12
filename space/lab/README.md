---
title: Desi Agent Lab
emoji: 🏦
colorFrom: green
colorTo: gray
sdk: static
app_file: index.html
pinned: true
license: apache-2.0
short_description: Base vs fine-tuned Tiny Aya banking agent, real traces
models:
  - CohereLabs/tiny-aya-global
  - ToolW/desi-agent-tinyaya-sft-lora
datasets:
  - ToolW/desi-agent-eval-traces
  - ToolW/desi-bank-toolcalls
---

# Desi Agent Lab

A static page (no model, no server) that replays our **real evaluation traces** from
[`ToolW/desi-agent-eval-traces`](https://huggingface.co/datasets/ToolW/desi-agent-eval-traces):

- **Tour**: a guided, autoplaying comparison of base Tiny Aya Global and our AgentTune SFT model on the same
  customer messages, with every tool call shown as a bank receipt. The episode list and captions come from
  `results/tour.json` (checked against the traces), and include a case the fine-tuned model still gets wrong.
- **Test explorer**: all 486 test conversations for every evaluated model, filterable by language, task type
  and outcome (fixed / still failing / regressions / teacher fails).
- **Results**: overall, per-language and per-task tables, with both views of unwanted bank actions.

Nothing is regenerated: what you see is exactly what was scored (fp16 eval, greedy decoding, mock bank).
Add `?record` to the URL for a chrome-free view (used to record the demo video), and `?speed=1.5` to change pacing.

Code: https://github.com/CyberRik/desi-agent (`space/lab/`). To type your own prompts, run the local demo
(`space/app.py`, see the repo README).
