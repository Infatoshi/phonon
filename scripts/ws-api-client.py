#!/usr/bin/env python3
"""A client for Phonon's WebSocket API, standard library only.

Dictate a WAV file through a Mac running Phonon with Settings > Advanced > WebSocket API
on, or refine a piece of text. It prints every message the Mac sends. Also a short
reference for writing a client: the protocol is in crates/phonon-core/src/remote.rs.

    export PHONON_WS_TOKEN=...        # Settings > Advanced > WebSocket API > Token
    scripts/ws-api-client.py ws://192.168.1.20:8377 recording.wav --app com.Slack --live
    scripts/ws-api-client.py ws://192.168.1.20:8377 --refine "i will be there at five"

The WAV must be 16 kHz mono 16-bit PCM, e.g.
    say -o recording.wav --data-format=LEI16@16000 "Hello from my phone"
"""

import argparse
import base64
import json
import os
import socket
import struct
import sys
import time
import wave
from urllib.parse import urlparse


class WebSocket:
    """One client connection: the opening handshake, masked frames out, frames in."""

    def __init__(self, url, timeout):
        parsed = urlparse(url)
        if parsed.scheme != "ws" or not parsed.hostname:
            raise SystemExit(f"expected ws://host:port, got {url!r}")
        self.sock = socket.create_connection((parsed.hostname, parsed.port or 8377), timeout)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        key = base64.b64encode(os.urandom(16)).decode()
        self.sock.sendall(
            (
                f"GET {parsed.path or '/'} HTTP/1.1\r\nHost: {parsed.netloc}\r\n"
                "Upgrade: websocket\r\nConnection: Upgrade\r\n"
                f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n"
            ).encode()
        )
        head = b""
        while not head.endswith(b"\r\n\r\n"):
            byte = self.sock.recv(1)
            if not byte:
                raise SystemExit("the server closed the connection during the handshake")
            head += byte
        if not head.startswith(b"HTTP/1.1 101"):
            raise SystemExit(f"not a WebSocket server: {head.splitlines()[0].decode()}")

    def _send(self, opcode, payload):
        mask = os.urandom(4)
        length = len(payload)
        if length < 126:
            header = struct.pack("!BB", 0x80 | opcode, 0x80 | length)
        elif length < 65536:
            header = struct.pack("!BBH", 0x80 | opcode, 0x80 | 126, length)
        else:
            header = struct.pack("!BBQ", 0x80 | opcode, 0x80 | 127, length)
        masked = bytes(byte ^ mask[i % 4] for i, byte in enumerate(payload))
        self.sock.sendall(header + mask + masked)

    def send_json(self, message):
        self._send(0x1, json.dumps(message).encode())

    def send_audio(self, pcm):
        self._send(0x2, pcm)

    def _read(self, count):
        data = b""
        while len(data) < count:
            piece = self.sock.recv(count - len(data))
            if not piece:
                raise ConnectionError("the server closed the connection")
            data += piece
        return data

    def receive(self):
        """The next JSON message; None when the server closes."""
        while True:
            first, second = self._read(2)
            length = second & 0x7F
            if length == 126:
                (length,) = struct.unpack("!H", self._read(2))
            elif length == 127:
                (length,) = struct.unpack("!Q", self._read(8))
            payload = self._read(length)
            opcode = first & 0x0F
            if opcode == 0x1:
                return json.loads(payload)
            if opcode == 0x8:
                return None
            if opcode == 0x9:
                self._send(0xA, payload)


def show(message):
    print(json.dumps(message, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("url", help="ws://host:port of the Mac")
    parser.add_argument("wav", nargs="?", help="16 kHz mono 16-bit WAV to dictate")
    parser.add_argument("--refine", metavar="TEXT", help="refine this text instead of dictating")
    parser.add_argument("--token", default=os.environ.get("PHONON_WS_TOKEN", ""),
                        help="the API token (default: $PHONON_WS_TOKEN)")
    parser.add_argument("--app", metavar="PACKAGE", help="the app the text is for, e.g. com.Slack")
    parser.add_argument("--before", default="", help="text before the cursor")
    parser.add_argument("--screen", default="", help="text on screen, to spell its terms")
    parser.add_argument("--live", action="store_true", help="ask for drafts while speaking")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="send the audio this many times faster than it was spoken")
    parser.add_argument("--timeout", type=float, default=90.0)
    args = parser.parse_args()
    if not args.token:
        raise SystemExit("no token: pass --token or set PHONON_WS_TOKEN")
    if bool(args.wav) == bool(args.refine):
        raise SystemExit("give a WAV file to dictate, or --refine TEXT")

    ws = WebSocket(args.url, args.timeout)
    ws.send_json({"type": "hello", "token": args.token, "client": "ws-api-client.py"})
    ready = ws.receive()
    show(ready)
    if not ready or ready.get("type") != "ready":
        return 1

    context = {}
    if args.app:
        context["app"] = {"package": args.app}
    if args.before:
        context["textbox"] = {"before_text": args.before, "selected_text": "", "after_text": ""}
    if args.screen:
        context["screen_text"] = args.screen

    if args.refine:
        ws.send_json({"type": "refine", "id": "r1", "text": args.refine, **context})
    else:
        with wave.open(args.wav) as audio:
            if (audio.getnchannels(), audio.getframerate(), audio.getsampwidth()) != (1, 16000, 2):
                raise SystemExit("the WAV must be 16 kHz mono 16-bit PCM")
            pcm = audio.readframes(audio.getnframes())
        ws.send_json({"type": "start", "id": "d1", "live": args.live, **context})
        started = ws.receive()
        show(started)
        if not started or started.get("type") != "started":
            return 1
        # 100 ms pieces at the pace of speech, as a microphone would deliver them. What
        # the Mac says meanwhile (partial, draft) is printed after the audio is out.
        step = 3200
        began = time.monotonic()
        for offset in range(0, len(pcm), step):
            ws.send_audio(pcm[offset:offset + step])
            due = began + (offset + step) / 32000 / max(args.speed, 0.01)
            time.sleep(max(0.0, due - time.monotonic()))
        ws.send_json({"type": "finish", "id": "d1"})

    while True:
        message = ws.receive()
        if message is None:
            print("closed by the server", file=sys.stderr)
            return 1
        show(message)
        if message.get("type") == "final":
            return 0
        if message.get("type") in ("error", "cancelled"):
            return 1


if __name__ == "__main__":
    sys.exit(main())
