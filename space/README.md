---
title: Desi Agent
emoji: 🏦
colorFrom: indigo
colorTo: green
sdk: gradio
sdk_version: 6.31.0
python_version: "3.12"
app_file: app.py
pinned: false
license: apache-2.0
short_description: Multilingual banking tool-calling agent on Tiny Aya Global
models:
  - CohereLabs/tiny-aya-global
  - CohereLabs/tiny-aya-global-GGUF
  - ToolW/desi-agent-tinyaya-sft-lora
datasets:
  - ToolW/desi-bank-toolcalls
---

# Desi Agent demo

Base **Tiny Aya Global** vs our **AgentTune SFT** adapter, side by side, as a banking assistant that calls tools
in English, Hindi, Bengali, Hinglish and Banglish. Tools run against a deterministic **mock** bank.

- Agent loop: `desi_agent.fmt.run_episode`, the same loop used for evaluation
- Prompt rendering: the same chat-template helper used in training
- Inference: llama.cpp on CPU, official `tiny-aya-global-q4_k_m.gguf` + our LoRA converted to GGUF

> **Quantized demo.** The Space runs a 4-bit (Q4_K_M) base with an f16 LoRA. All metrics we report come from fp16
> evaluation (vLLM / HF transformers), so behaviour here can differ slightly from the result tables.

Code, data, training and eval: https://github.com/CyberRik/desi-agent
