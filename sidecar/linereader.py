"""Non-blocking JSON-lines reader over stdin for the sidecars.

Lets a sidecar see every command that is already waiting: the ASR sidecar folds queued audio
chunks into one pass and skips speculative work once a stream_finish is queued; the correction
sidecar aborts a background stream job between forward passes when a run arrives."""

from __future__ import annotations

import json
import os
import select


class LineReader:
    def __init__(self, fd: int = 0):
        self.fd = fd
        self.buf = b""
        self.eof = False

    def poll(self, timeout: float | None) -> list:
        """Parsed messages available within `timeout` seconds (None waits for one). A line that
        is not JSON comes back as its text, so the caller can report it."""
        out: list = []
        while not self.eof:
            ready, _, _ = select.select([self.fd], [], [], timeout)
            if not ready:
                break
            chunk = os.read(self.fd, 1 << 16)
            if not chunk:
                self.eof = True
                break
            self.buf += chunk
            if b"\n" in self.buf:
                timeout = 0.0
        while b"\n" in self.buf:
            line, self.buf = self.buf.split(b"\n", 1)
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                out.append(line.decode("utf-8", "replace"))
        return out
