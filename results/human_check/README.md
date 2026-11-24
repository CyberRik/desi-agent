# [R] Language-detector check (Stage 2)

- `grpo_switch_sample.csv`: 60 GRPO (`s2-grpo-mt`) language-switch conversations, 12 per target language, seed 1234,
  sampled with up to 4 detector failures per language. Shows the full turn (tool calls, bank results, final reply).
- `grpo_switch_sample.key.json`: the detector verdicts (`lang_match`) and auto task success, kept apart for blind annotation.
- `grpo_switch_sample.llm_annotated.csv`: blind annotation by an LLM (Claude), **not a native speaker**.
  Agreement with the detector is 56/60, Cohen's kappa 0.85 (3 detector rejections the annotator accepted, 1 acceptance it rejected).
- Also found: replies that claim an action without calling a tool (7/600 GRPO multi-turn, 12/600 RFT2, 0 on the single-turn test sets).
- A native speaker can fill the `YOUR_*` columns of `grpo_switch_sample.csv` to turn this into a human check.
