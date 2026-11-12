"""[R] Score our traces with AuditKIT's agent_eval (independent second opinion on our metrics)."""
from __future__ import annotations

from pathlib import Path

from auditkit.agent_eval import AgentCase, AgentEvalRunner, AgentEvalSpec, episode_from_openai_messages

from .. import tools

SCORERS = ["tool_call_f1", "tool_call_validity", "trajectory_match", "redundant_tool_calls", "agent_loop_detection"]


def auditkit_eval(traces: list[dict], out_path: str | Path) -> dict:
    specs = tools.tool_specs()
    cases, episodes = [], []
    for t in traces:
        task = t["messages"][0]["content"]
        cases.append(AgentCase(
            id=t["id"], task=task, category=t["category"], allowed_tools=specs,
            reference_turns=[[g] for g in t["gold"]["tool_sequence"]],
            metadata={"lang": t["lang"]}))
        # AuditKIT doesn't know our __invalid__ pseudo-tool name: keep it, it scores as an invalid call.
        episodes.append(episode_from_openai_messages(t["messages"], case_id=t["id"]))
    result = AgentEvalRunner().run(AgentEvalSpec(cases=cases, mode="recorded", episodes=episodes,
                                                 scorers=SCORERS, trials=1, name="desi-agent"))
    result.to_json(str(out_path))
    # Mean of every AuditKIT diagnostic over the cases where it was computed.
    sums, counts = {}, {}
    for row in result.to_dict()["rows"]:
        for name, diag in (row.get("diagnostics") or {}).items():
            v = diag.get("value") if isinstance(diag, dict) else None
            if isinstance(v, (int, float)):
                sums[name] = sums.get(name, 0.0) + v
                counts[name] = counts.get(name, 0) + 1
    means = {k: round(sums[k] / counts[k], 4) for k in sorted(sums)}
    return {"path": str(out_path), "scorers": SCORERS, "means": means, "n_scored": counts}
