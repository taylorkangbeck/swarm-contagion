"""Append-only shared board with anonymous rendering.

Sender ids are persisted to board.jsonl for analysis only and never rendered.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

MAX_CHARS = 500
EMPTY = "(The board is empty.)"


@dataclass(frozen=True)
class Message:
    index: int
    agent_id: str
    round: int
    content: str
    truncated: bool
    original_length: int


class Board:
    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path else None
        self.messages: list[Message] = []

    def post(self, agent_id: str, round_: int, content: str) -> Message:
        # Collapse whitespace so one message is always one numbered line and an
        # agent cannot fake extra "[n]" entries with newlines.
        clean = re.sub(r"\s+", " ", content).strip()
        truncated = len(clean) > MAX_CHARS
        msg = Message(
            index=len(self.messages) + 1,
            agent_id=agent_id,
            round=round_,
            content=clean[:MAX_CHARS],
            truncated=truncated,
            original_length=len(clean),
        )
        self.messages.append(msg)
        if self.path:
            with self.path.open("a") as f:
                f.write(json.dumps(asdict(msg)) + "\n")
        return msg

    def render(self, window: int | None = None, upto: int | None = None) -> str:
        """Numbered, content only, oldest first. `upto` limits to the first n messages."""
        msgs = self.messages[:upto] if upto is not None else self.messages
        if window is not None:
            msgs = msgs[-window:] if window > 0 else []
        if not msgs:
            return EMPTY
        return "\n".join(f"[{m.index}] {m.content}" for m in msgs)

    def __len__(self) -> int:
        return len(self.messages)

    @classmethod
    def load(cls, path: Path) -> "Board":
        b = cls()
        if Path(path).exists():
            for line in Path(path).read_text().splitlines():
                if line.strip():
                    b.messages.append(Message(**json.loads(line)))
        return b
