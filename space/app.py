"""[F] F4: Desi Agent demo (Gradio, CPU). Base Tiny Aya Global vs our SFT adapter, side by side.

Both columns run the *same* agent loop as eval (`fmt.run_episode`) against the deterministic mock bank
(`tools.py`), with prompts rendered by the same chat-template helper as training (`train/chat.py`).
Inference is llama.cpp (official Q4_K_M GGUF of Tiny Aya Global; our LoRA converted with
`lora_to_gguf.py`), so it runs on a free 2-vCPU Space. Tool calls and results stream in as they happen.

Local:  python space/app.py   (env: ADAPTER_REPO, ADAPTER_FILE, N_THREADS)
"""
from __future__ import annotations

import json
import os
import queue
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src" if (HERE / "src").exists() else HERE.parent / "src"))

import gradio as gr  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402
from llama_cpp import Llama  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from desi_agent import fmt  # noqa: E402
from desi_agent.train.chat import apply_template  # noqa: E402

BASE_ID = "CohereLabs/tiny-aya-global"
BASE_GGUF = ("CohereLabs/tiny-aya-global-GGUF", "tiny-aya-global-q4_k_m.gguf")
ADAPTER = (os.environ.get("ADAPTER_REPO", "ToolW/desi-agent-tinyaya-sft-lora"),
           os.environ.get("ADAPTER_FILE", "gguf/desi-agent-sft-lora-f16.gguf"))
N_THREADS = int(os.environ.get("N_THREADS", os.cpu_count() or 2))
MAX_NEW = 200

EXAMPLES = [
    ["en", "What's my savings account balance?"],
    ["en", "I see a duplicate charge in my last 2 transactions. Please dispute the duplicate one."],
    ["hi", "मेरा डेबिट कार्ड खो गया है, आखिरी चार अंक 4821 हैं। तुरंत ब्लॉक करो।"],
    ["hi", "ramesh@okaxis को ₹1500 भेज दो, नोट में 'किराया' लिखना"],
    ["bn", "আমার শেষ ৩টা ডেবিট লেনদেন দেখাও"],
    ["bn", "আমার ভাইকে টাকা পাঠাতে চাই"],
    ["hi-en", "bhai 5 lakh ka loan 9.5% pe 36 mahine ke liye, EMI kitna banega?"],
    ["hi-en", "mera card block kar do jaldi"],
    ["bn-en", "amar kache ekta ATM khuje dao, pincode 700019"],
    ["bn-en", "amar UPI payment ta ki hoyeche? ref 402918375610"],
]

tok = AutoTokenizer.from_pretrained(BASE_ID)
models: dict[str, Llama | None] = {"base": None, "sft": None}
status: dict[str, str] = {"base": "loading…", "sft": "loading…"}
lock = threading.Lock()  # llama.cpp contexts are not thread-safe; one generation at a time


def _load() -> None:
    base_path = hf_hub_download(*BASE_GGUF)
    models["base"] = Llama(model_path=base_path, n_ctx=4096, n_threads=N_THREADS, n_batch=512, verbose=False)
    status["base"] = "ready"
    try:
        lora = os.environ.get("ADAPTER_PATH") or hf_hub_download(*ADAPTER)
        models["sft"] = Llama(model_path=base_path, lora_path=lora, n_ctx=4096, n_threads=N_THREADS,
                              n_batch=512, verbose=False)
        status["sft"] = "ready"
    except Exception as e:  # noqa: BLE001 - adapter not published yet: base-only demo
        status["sft"] = f"not available ({type(e).__name__})"
    # Warm the prefix cache: the ~1.3k-token system prompt is identical on every call.
    warm = fmt.to_chat([{"role": "user", "content": "hi"}])
    for k in ("base", "sft"):
        if models[k] is not None:
            with lock:
                _generate(models[k], warm, max_tokens=1)


def _generate(llm: Llama, chat: list[dict], max_tokens: int = MAX_NEW) -> str:
    text = apply_template(tok, chat, add_generation_prompt=True)
    ids = llm.tokenize(text.encode(), add_bos=False, special=True)  # template already has BOS as text
    out = llm.create_completion(prompt=ids, max_tokens=max_tokens, temperature=0.0, stop=["</tool_call>"])
    c = out["choices"][0]
    txt = c["text"]
    if c["finish_reason"] == "stop" and "<tool_call>" in txt and "</tool_call>" not in txt:
        txt += "</tool_call>"  # llama.cpp drops the stop string; HF policy keeps it
    return txt.strip()


def _to_chatbot(messages: list[dict]) -> list[dict]:
    """OpenAI-style episode messages -> Gradio chatbot messages (tool calls as collapsible blocks)."""
    out = []
    for m in messages:
        if m["role"] == "user":
            out.append({"role": "user", "content": m["content"]})
        elif m["role"] == "assistant" and m.get("tool_calls"):
            for c in m["tool_calls"]:
                fn = c["function"]
                try:
                    args = json.dumps(json.loads(fn["arguments"]), ensure_ascii=False)
                except (TypeError, ValueError):
                    args = str(fn["arguments"])
                out.append({"role": "assistant", "content": f"```json\n{args}\n```",
                            "metadata": {"title": f"🔧 {fn['name']}"}})
        elif m["role"] == "tool":
            try:
                body = json.dumps(json.loads(m["content"]), ensure_ascii=False, indent=1)
            except ValueError:
                body = m["content"]
            out.append({"role": "assistant", "content": f"```json\n{body}\n```",
                        "metadata": {"title": f"↩️ result: {m['name']}"}})
        elif m["role"] == "assistant":
            out.append({"role": "assistant", "content": m["content"] or ""})
    return out


def _run(key: str, history: list[dict], user_msg: str, q: queue.Queue) -> None:
    llm = models[key]
    msgs = [*history, {"role": "user", "content": user_msg}]
    if llm is None:
        q.put((key, "done", msgs + [{"role": "assistant", "content": f"_model {status[key]}_"}], 0.0))
        return
    partial = list(msgs)

    def policy(chat):
        with lock:
            out = _generate(llm, chat)
        turn = fmt.parse_assistant(out)
        if turn.kind == "tool_call" and turn.valid_json:  # live preview of the call before its result
            partial.append({"role": "assistant", "content": None, "tool_calls": [
                {"id": "x", "type": "function", "function": {"name": turn.name, "arguments": json.dumps(turn.arguments)}}]})
            q.put((key, "step", list(partial), None))
        return out

    t0 = time.time()
    try:
        run = fmt.run_episode(policy, msgs)
        out = run.messages
        if run.truncated:  # keep roles alternating for the next turn, and say what happened
            out = [*out, {"role": "assistant", "content": "_(stopped: no final answer after 6 steps)_"}]
    except Exception as e:  # noqa: BLE001 - never leave the UI waiting on a dead worker
        out = [*partial, {"role": "assistant", "content": f"_(error: {type(e).__name__}: {e})_"}]
    q.put((key, "done", out, round(time.time() - t0, 1)))


def respond(user_msg, hist_base, hist_sft):
    if not user_msg or not user_msg.strip():
        yield gr.update(), gr.update(), hist_base, hist_sft, ""
        return
    q: queue.Queue = queue.Queue()
    msgs = {"base": hist_base, "sft": hist_sft}
    shown = {k: _to_chatbot([*v, {"role": "user", "content": user_msg}]) for k, v in msgs.items()}
    timing = {}
    yield shown["base"], shown["sft"], hist_base, hist_sft, ""
    for key in ("sft", "base"):  # fine-tuned first: it is the point of the demo
        threading.Thread(target=_run, args=(key, msgs[key], user_msg, q), daemon=True).start()
        while True:
            k, kind, m, secs = q.get()
            shown[k] = _to_chatbot(m)
            if kind == "done":
                msgs[k] = m
                timing[k] = secs
                break
            yield shown["base"], shown["sft"], msgs["base"], msgs["sft"], ""
        yield shown["base"], shown["sft"], msgs["base"], msgs["sft"], ""
    note = " · ".join(f"{'fine-tuned' if k == 'sft' else 'base'}: {v}s" for k, v in timing.items() if v)
    yield shown["base"], shown["sft"], msgs["base"], msgs["sft"], note


def model_status() -> str:
    return f"**Base:** {status['base']} · **Fine-tuned:** {status['sft']} · CPU threads: {N_THREADS}"


with gr.Blocks(title="Desi Agent: Tiny Aya banking tool-calling") as demo:
    gr.Markdown(
        "# 🏦 Desi Agent: a multilingual banking agent on Tiny Aya Global\n"
        "English · हिन्दी · বাংলা · Hinglish · Banglish. The same message goes to **base Tiny Aya Global** and to our "
        "**AgentTune SFT** version. Watch which tools each one calls, with which arguments, and what it answers.\n\n"
        "_Tools run against a deterministic **mock** bank (no real accounts, no real money). Runs on a free 2-vCPU "
        "Space (llama.cpp, 4-bit), so the first reply takes ~20 s and later ones ~5-20 s per model._")
    stat = gr.Markdown(model_status())
    with gr.Row():
        with gr.Column():
            gr.Markdown("### ✅ Fine-tuned (AgentTune SFT)")
            bot_sft = gr.Chatbot(height=460, label="fine-tuned")
        with gr.Column():
            gr.Markdown("### Base Tiny Aya Global")
            bot_base = gr.Chatbot(height=460, label="base")
    st_base, st_sft = gr.State([]), gr.State([])
    with gr.Row():
        box = gr.Textbox(placeholder="Ask the bank… e.g. 'mera balance batao'", scale=5, show_label=False)
        send = gr.Button("Send", variant="primary", scale=1)
        clear = gr.Button("Clear", scale=1)
    timing = gr.Markdown()
    lang = gr.Textbox(visible=False)
    gr.Examples(EXAMPLES, inputs=[lang, box], label="Examples (language · message)")
    outs = [bot_base, bot_sft, st_base, st_sft, timing]
    send.click(respond, [box, st_base, st_sft], outs).then(lambda: "", None, box)
    box.submit(respond, [box, st_base, st_sft], outs).then(lambda: "", None, box)
    clear.click(lambda: ([], [], [], [], ""), None, outs)
    demo.load(model_status, None, stat)
    gr.Timer(5).tick(model_status, None, stat)

threading.Thread(target=_load, daemon=True).start()

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch()
