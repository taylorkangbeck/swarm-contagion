"""Append-only JSONL event log shared by the LLM wrapper, tool executor and runner."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path


class EventLog:
    def __init__(self, path: Path | None, run_id: str = ""):
        self.path = Path(path) if path else None
        self.run_id = run_id
        self.events: list[dict] = []
        self._lock = threading.Lock()

    def write(self, event_type: str, **fields) -> dict:
        event = {"ts": time.time(), "run_id": self.run_id, "type": event_type, **fields}
        with self._lock:
            self.events.append(event)
            if self.path:
                with self.path.open("a") as f:
                    f.write(json.dumps(event, default=str) + "\n")
        return event
