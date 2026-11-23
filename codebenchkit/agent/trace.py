"""One JSONL trace per run: every LLM call, tool call, verify round, compaction and guardrail.

The trace is the ground truth for "what did the agent actually do". Tool outputs are kept whole
here even when the loop trims them out of the model's context, so trimming never loses
evidence for a human. Writes are line-buffered and flushed, so a crashed run still leaves a
readable trace.
"""
from __future__ import annotations

import json
import time
from pathlib import Path


class Tracer:
    def __init__(self, path: str | Path | None = None, keep: bool = True):
        self.path = Path(path) if path else None
        self.events: list[dict] = [] if keep else None   # in-memory copy, used by tests and receipts
        self._fh = None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = self.path.open("w", encoding="utf-8")

    def event(self, _event: str, /, **data) -> dict:
        rec = {"ts": round(time.time(), 3), "event": _event, **data}
        if self._fh:
            self._fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            self._fh.flush()
        if self.events is not None:
            self.events.append(rec)
        return rec

    def of(self, kind: str) -> list[dict]:
        return [e for e in self.events or [] if e["event"] == kind]

    def close(self) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def preview(text: str, n: int = 400) -> str:
    text = str(text)
    return text if len(text) <= n else text[:n] + f"... [{len(text):,} chars]"
