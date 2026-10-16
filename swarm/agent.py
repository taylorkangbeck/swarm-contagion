"""One agent-turn: a manual tool loop so the harness intercepts every tool call."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .events import EventLog
from .llm import LLM, usage_dict
from .tools import ToolExecutor, TurnState


@dataclass
class TurnResult:
    agent_id: str
    round: int
    submissions: dict[str, str]
    posted_message: str | None
    read_key_calls: list[dict]
    tool_calls: list[str]
    private_note: str
    thinking_summaries: list[str]
    stop_reason: str | None
    refusal_category: str | None
    api_calls: int
    usage: dict[str, int] = field(default_factory=dict)
    board_len_at_turn: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def _block_to_json(block: Any) -> Any:
    if hasattr(block, "model_dump"):
        d = block.model_dump(mode="json", exclude_none=True)
    else:
        d = dict(block)
    if d.get("type") == "thinking":
        d["signature"] = "<omitted>"  # opaque and large; not evidence
    return d


def messages_to_json(messages: list) -> list[dict]:
    out = []
    for m in messages:
        c = m["content"]
        out.append({"role": m["role"], "content": c if isinstance(c, str) else [_block_to_json(b) for b in c]})
    return out


def run_agent_turn(
    llm: LLM,
    executor: ToolExecutor,
    log: EventLog,
    turn: TurnState,
    system: str,
    user_message: str,
    tools: list[dict],
    max_tool_iterations: int,
    transcript_path: Path | None = None,
    board_len_at_turn: int = 0,
) -> TurnResult:
    messages: list[dict] = [{"role": "user", "content": user_message}]
    private_note_parts: list[str] = []
    thinking_summaries: list[str] = []
    usage = {}
    stop_reason = refusal_category = None
    api_calls = 0
    ctx = {"agent_id": turn.agent_id, "round": turn.round}

    for iteration in range(max_tool_iterations):
        response = llm.create(system, messages, tools, iteration=iteration, **ctx)
        api_calls += 1
        for k, v in usage_dict(response.usage).items():
            usage[k] = usage.get(k, 0) + v
        stop_reason = response.stop_reason

        if stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            refusal_category = getattr(details, "category", None) if details else None
            log.write("refusal", category=refusal_category, iteration=iteration, **ctx)
            messages.append({"role": "assistant", "content": response.content})
            break

        # Append the assistant turn unchanged (thinking blocks included).
        messages.append({"role": "assistant", "content": response.content})
        for block in response.content:
            if block.type == "thinking" and block.thinking:
                thinking_summaries.append(block.thinking)
            elif block.type == "text" and iteration == 0:
                private_note_parts.append(block.text)

        if stop_reason == "tool_use":
            results = [executor.execute(turn, b) for b in response.content if b.type == "tool_use"]
            # All tool results from one assistant message go back in a single user message.
            messages.append({"role": "user", "content": results})
            if iteration == max_tool_iterations - 1:
                log.write("iteration_cap", iteration=iteration, **ctx)
            continue
        if stop_reason == "max_tokens":
            log.write("truncated", iteration=iteration, **ctx)
        elif stop_reason != "end_turn":
            log.write("unexpected_stop", stop_reason=stop_reason, iteration=iteration, **ctx)
        break

    if turn.posted_message is None:
        log.write("no_post", **ctx)
    missing = [p.id for p in turn.problems if p.id not in turn.submissions]
    if missing:
        log.write("missing_submissions", problem_ids=missing, **ctx)

    result = TurnResult(
        agent_id=turn.agent_id,
        round=turn.round,
        submissions=dict(turn.submissions),
        posted_message=turn.posted_message,
        read_key_calls=list(turn.read_key_calls),
        tool_calls=list(turn.tool_calls),
        private_note="\n".join(private_note_parts).strip(),
        thinking_summaries=thinking_summaries,
        stop_reason=stop_reason,
        refusal_category=refusal_category,
        api_calls=api_calls,
        usage=usage,
        board_len_at_turn=board_len_at_turn,
    )
    if transcript_path is not None:
        transcript_path.parent.mkdir(parents=True, exist_ok=True)
        transcript_path.write_text(
            json.dumps(
                {
                    "agent_id": turn.agent_id,
                    "round": turn.round,
                    "is_bad": turn.is_bad,
                    "problems": [asdict(p) for p in turn.problems],
                    "system": system,
                    "messages": messages_to_json(messages),
                    "result": result.to_dict(),
                },
                indent=2,
            )
            + "\n"
        )
    return result
