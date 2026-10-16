import json

from swarm.agent import run_agent_turn
from swarm.board import Board
from swarm.config import RunConfig
from swarm.events import EventLog
from swarm.llm import LLM, FakeClient, fake_message, make_fake_policy, text, thinking, tool_use
from swarm.problems import generate_for, problem_id
from swarm.runner import run
from swarm.tools import TOOLS, ToolExecutor, TurnState


def _cfg(tmp_path, **kw):
    base = dict(condition="test", n_agents=3, n_bad=1, rounds=2, results_dir=tmp_path, seed_mode="llm")
    base.update(kw)
    return RunConfig(**base)


def test_three_agent_run_with_round2_defection(tmp_path):
    cfg = _cfg(tmp_path)
    target = problem_id(cfg.rng_seed, 2, "honest_0", 0)
    client = FakeClient(make_fake_policy(defect_when=lambda user_text: target in user_text))
    summary = run(cfg, client, verbose=False)

    assert summary["order"] == ["bad_0", "honest_0", "honest_1"]
    assert summary["first_defection_round"] == {"honest_0": 2, "honest_1": None}
    assert [summary["per_round"][str(r)]["cumulative_count"] for r in (1, 2)] == [0, 1]
    assert summary["per_round"]["2"]["honest_defected_this_round"] == ["honest_0"]
    assert summary["per_round"]["1"]["bad_read_key_calls"] == 2
    assert summary["per_round"]["2"]["scores"]["bad_0"] == [4, 4]
    assert summary["usage_totals"]["calls"] > 0

    # Full results layout on disk.
    d = cfg.run_dir
    for name in ("config.json", "events.jsonl", "board.jsonl", "summary.json"):
        assert (d / name).exists()
    assert len(list((d / "transcripts").glob("r*_*.json"))) == 6

    # No agent id ever reaches a model: not in the board, not in any prompt.
    ids = cfg.order
    for call in client.calls:
        blob = json.dumps(call["messages"], default=str) + call["system"]
        assert not any(a in blob for a in ids)
    rendered = Board.load(d / "board.jsonl").render()
    assert rendered.count("\n") == 5  # 6 posts
    assert not any(a in rendered for a in ids)

    # Bad seed acts before honest agents in every round, and sees the posts in order.
    starts = [json.loads(l) for l in (d / "events.jsonl").read_text().splitlines()]
    starts = [(e["round"], e["agent_id"]) for e in starts if e["type"] == "turn_start"]
    assert starts == [(r, a) for r in (1, 2) for a in ids]


def test_api_params_follow_codebase_rules(tmp_path):
    cfg = _cfg(tmp_path, rounds=1)
    client = FakeClient(make_fake_policy())
    run(cfg, client, verbose=False)
    for call in client.calls:
        assert call["thinking"] == {"type": "adaptive", "display": "summarized"}
        assert call["output_config"] == {"effort": "medium"}
        assert call["tool_choice"] == {"type": "auto"}
        for banned in ("temperature", "top_p", "top_k"):
            assert banned not in call
        assert call["tools"] == TOOLS


def test_scripted_seed_mode(tmp_path):
    cfg = _cfg(tmp_path, seed_mode="scripted", n_bad=2, n_agents=5, rounds=2)
    client = FakeClient(make_fake_policy())
    summary = run(cfg, client, verbose=False)
    assert summary["per_round"]["1"]["bad_read_key_calls"] == 4
    assert summary["per_round"]["2"]["cumulative_count"] == 0
    board = Board.load(cfg.run_dir / "board.jsonl")
    seed_posts = [m.content for m in board.messages if m.agent_id.startswith("bad")]
    assert len(seed_posts) == 4 and len(set(seed_posts)) == 4  # two seeds never post identical text
    # The scripted seed makes no API calls: 3 honest x 2 rounds x 2 calls each.
    assert len(client.calls) == 12


def _single_turn(script):
    log = EventLog(None)
    board = Board()
    problems = generate_for(0, 1, "honest_0", 2, 2)
    turn = TurnState("honest_0", 1, False, problems)
    llm = LLM(FakeClient(script), "fake", "medium", 1000, log)
    res = run_agent_turn(llm, ToolExecutor(log, board), log, turn, "sys", "user", TOOLS, 6)
    return res, log, llm, problems


def test_refusal_is_logged_not_dropped():
    msg = fake_message([], stop_reason="refusal")
    res, log, _, _ = _single_turn([msg])
    assert res.stop_reason == "refusal"
    assert any(e["type"] == "refusal" for e in log.events)
    assert any(e["type"] == "no_post" for e in log.events)


def test_max_tokens_logged():
    res, log, _, _ = _single_turn([fake_message([thinking("hmm")], stop_reason="max_tokens")])
    assert res.stop_reason == "max_tokens"
    assert any(e["type"] == "truncated" for e in log.events)


def test_tool_results_in_single_user_message_and_assistant_unchanged():
    problems = generate_for(0, 1, "honest_0", 2, 2)
    first = fake_message([
        thinking("plan"),
        text("Note: solving by hand."),
        tool_use("submit_answer", problem_id=problems[0].id, answer="1"),
        tool_use("submit_answer", problem_id=problems[1].id, answer="2"),
        tool_use("post_message", content="hello"),
    ])
    second = fake_message([text("done")], stop_reason="end_turn")
    log = EventLog(None)
    client = FakeClient([first, second])
    llm = LLM(client, "fake", "medium", 1000, log)
    turn = TurnState("honest_0", 1, False, problems)
    res = run_agent_turn(llm, ToolExecutor(log, Board()), log, turn, "sys", "user", TOOLS, 6)
    msgs = client.calls[1]["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
    assert msgs[1]["content"] == first.content  # thinking block included, unchanged
    assert [b["type"] for b in msgs[2]["content"]] == ["tool_result"] * 3
    assert res.private_note == "Note: solving by hand."
    assert res.thinking_summaries == ["plan"]
    assert res.read_key_calls == []
    assert res.usage["input_tokens"] == 2000
