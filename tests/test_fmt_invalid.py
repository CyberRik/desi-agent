"""An unparseable tool call must not crash the agent loop on the next turn."""
from desi_agent import fmt


def test_invalid_call_is_replayed_and_loop_continues():
    outs = iter(['<tool_call>{"name": "get_account_balance", "arguments": {broken</tool_call>',
                 '<tool_call>{"name": "get_account_balance", "arguments": {"account_type": "savings"}}</tool_call>',
                 "Your balance is shown above."])
    seen = []

    def policy(chat):
        seen.append(chat)
        return next(outs)

    run = fmt.run_episode(policy, [{"role": "user", "content": "balance?"}])
    assert not run.truncated and run.messages[-1]["content"] == "Your balance is shown above."
    # 2nd call: the model sees its own broken text, then the parse error
    assert seen[1][-2] == {"role": "assistant", "content": '<tool_call>{"name": "get_account_balance", "arguments": {broken</tool_call>'}
    assert "could not parse" in seen[1][-1]["content"]
    calls = fmt.predicted_calls(run.messages)
    assert [c["name"] for c in calls] == [fmt.INVALID, "get_account_balance"] and calls[0]["arguments"] is None
