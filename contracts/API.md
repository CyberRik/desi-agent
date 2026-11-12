# API — the Python interfaces between lanes

Code-level contract between R (data + eval) and F (train + ship). Signatures are frozen in
`src/desi_agent/api.py`. Each lane implements its side in its own package; the other lane
only ever imports through `desi_agent.api`. That way nobody needs to know how the other
side works inside, and either side can be stubbed while the other is unfinished.

```
                  ┌──────────── shared (frozen) ────────────┐
                  │ tools.execute / tools.validate_call      │
                  │ fmt.to_chat / fmt.parse_assistant        │
                  │ fmt.run_episode(policy, user_msgs)       │
                  └──────────────────────────────────────────┘
        R provides                                    F provides
  load_split(name) -> list[Episode] ───────────▶  train_sft(cfg) -> TrainResult
  score_episode(messages, gold) -> Scores ─────▶  (used as GRPO reward)
  evaluate(policy, split, tag) -> Results  ◀────  load_policy(model_id, adapter) -> Policy
```

## Shared types
- `Message = dict` (OpenAI chat message, see DATA_FORMAT.md)
- `Episode = dict` (one DATA_FORMAT row)
- `Policy = Callable[[list[{"role","content"}]], str]`: takes the **rendered** chat (output of
  `fmt.to_chat`), returns raw assistant text. Greedy decoding unless the caller says otherwise.
  Must be stateless across calls. A batched variant `policy.batch(list_of_chats) -> list[str]` is optional.

## R provides (`desi_agent.data`, `desi_agent.eval`)
| function | contract |
|---|---|
| `load_split(name: "train"\|"val"\|"test"\|"seed") -> list[Episode]` | reads `data/<name>.jsonl` (or the HF dataset if `DESI_DATA=hf`) |
| `score_episode(pred_messages, gold) -> dict[str, float]` | METRICS.md per-episode keys. Pure, fast, no model. **Also F's GRPO reward.** |
| `reward(pred_messages, gold) -> float` | scalar for RL: `1.0*task_success + 0.3*schema_valid - 0.5*false_action - 0.1*redundant_calls`. Tunable only via `CONTRACT:` PR |
| `evaluate(policy, split="test", model_tag=..., limit=None) -> dict` | runs `fmt.run_episode` on each episode, writes `results/<tag>/<split>.json` + traces, returns the results dict |

## F provides (`desi_agent.train`)
| function | contract |
|---|---|
| `load_policy(model_id="CohereLabs/tiny-aya-global", adapter: str\|None=None, **gen_kw) -> Policy` | HF transformers (or vLLM) backed policy. `adapter` = local path or HF repo of a LoRA |
| `train_sft(cfg: dict) -> dict` | AgentTune distillation on `load_split("train")`; returns `{"adapter_path", "hf_repo", "metrics"}` |
| `train_grpo(cfg: dict) -> dict` | AgentTune GRPO using `reward`; same return shape |
| `episodes_to_eventlogs(episodes) -> list[EventLog]` | DATA_FORMAT → AgentTune full-tier `EventLog` |

## Stubs
Until a side is implemented, its functions raise `NotImplementedError("[R] R5 not merged yet")`.
`desi_agent.api.DummyPolicy` (always answers with the gold first call, or a fixed text) lets R
test `evaluate()` and lets F test the training loop without the other side being ready.
