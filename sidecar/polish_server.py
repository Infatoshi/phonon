#!/usr/bin/env python3
"""JSONL correction sidecar: load the pinned MLX correction model, polish transcripts.

Speaks the same line protocol the engine already used for the correction stage:
  stdin:  {"command":"warmup"} | {"command":"run","inputText":"…"}
          {"command":"status"} | {"command":"shutdown"}
  stdout: {"ok":true,"status":{"state":"ready","message":"…"}}
          {"ok":true,"outputText":"…","diagnostics":{…}}
          {"ok":false,"error":"…"}
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
import traceback

# Gemma 4 opens a thought channel by default even when the template omits
# `<|think|>`. Prefilling a closed thought channel puts the model straight into
# the answer channel, which is what a formatter needs: short, literal output.
NO_THINK_PREFILL = "<|channel>thought\nNo thinking needed.\n<channel|>"

# Hard ceiling on generated tokens. A correction pass never legitimately needs
# more; without it a bad expansion decodes until the model's own limit and the
# request feels frozen.
OUTPUT_TOKEN_CEILING = 256
OUTPUT_TOKEN_FLOOR = 48
TRANSCRIPT_PATTERN = re.compile(r"<transcript>(.*?)</transcript>", re.DOTALL)
LEAKED_MARKER_PATTERN = re.compile(r"<\|?/?(?:channel|turn|think)\|?>?")


def emit(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def transcript_payload(text: str) -> str:
    """The part of the prompt that bounds a reasonable output length."""
    match = TRANSCRIPT_PATTERN.search(text)
    return match.group(1).strip() if match else text.strip()


def clean_output(text: str) -> str:
    return LEAKED_MARKER_PATTERN.sub("", text).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--system-prompt-file")
    return parser.parse_known_args()[0]


def main() -> None:
    args = parse_args()

    emit({"ok": True, "status": {"state": "loading", "message": "importing MLX"}})
    try:
        from huggingface_hub import snapshot_download
        from mlx_lm import load
        from mlx_lm.generate import stream_generate
        from mlx_lm.sample_utils import make_sampler
    except Exception as error:  # noqa: BLE001 - report to the engine, never crash silently
        emit({"ok": False, "error": f"correction runtime import failed: {error}"})
        return

    emit(
        {
            "ok": True,
            "status": {
                "state": "loading",
                "message": f"resolving {args.model}@{args.revision[:7]}",
            },
        }
    )
    try:
        local_dir = snapshot_download(args.model, revision=args.revision)
    except Exception as error:  # noqa: BLE001
        emit({"ok": False, "error": f"correction model download failed: {error}"})
        return

    emit(
        {
            "ok": True,
            "status": {"state": "loading", "message": "loading weights (MLX)"},
        }
    )
    started = time.perf_counter()
    try:
        model, tokenizer = load(local_dir)
    except Exception as error:  # noqa: BLE001
        emit({"ok": False, "error": f"correction model load failed: {error}"})
        traceback.print_exc(file=sys.stderr)
        return
    load_seconds = time.perf_counter() - started

    system_prompt = ""
    if args.system_prompt_file:
        try:
            with open(args.system_prompt_file, encoding="utf-8") as handle:
                system_prompt = handle.read().strip()
        except OSError as error:
            emit({"ok": False, "error": f"system prompt unreadable: {error}"})
            return

    sampler = make_sampler(temp=0.0)

    # The prefill above closes Gemma 4's thought channel. Any model whose chat
    # template has no channel mechanism would receive it as literal prompt text,
    # so only prefill when the template actually speaks that protocol.
    template = getattr(tokenizer, "chat_template", None) or ""
    think_prefill = NO_THINK_PREFILL if "channel" in template else ""

    # mlx-lm's TokenizerWrapper.detokenizer is a property that builds a brand new
    # streaming detokenizer on every access, and building one walks this model's
    # 262144-entry vocabulary in Python. stream_generate reads it once per call,
    # so a resident sidecar was paying ~134 ms of table building on every
    # utterance, which was most of the correction stage's time to first token.
    # A detokenizer's reset() clears the whole of its mutable state (offset,
    # pending bytes, text, tokens) and the id-to-piece table it rebuilds is
    # immutable, so handing back one reset instance is indistinguishable from
    # handing back a fresh one.
    shared_detokenizer = tokenizer.detokenizer

    def detokenizer_property(_wrapper):
        shared_detokenizer.reset()
        return shared_detokenizer

    type(tokenizer).detokenizer = property(detokenizer_property)

    def build_prompt(text: str) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": text})
        rendered = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        return rendered + think_prefill

    def output_budget(text: str) -> int:
        spoken = transcript_payload(text)
        spoken_tokens = len(tokenizer.encode(spoken)) if spoken else 0
        # A clean-up pass returns roughly the input again; allow headroom for
        # expanded orthography, then stop.
        budget = math.ceil(spoken_tokens * 1.8) + 24
        return max(OUTPUT_TOKEN_FLOOR, min(OUTPUT_TOKEN_CEILING, budget))

    def run(text: str) -> dict:
        prompt = build_prompt(text)
        max_tokens = output_budget(text)
        started_at = time.perf_counter()
        ttft_ms = None
        pieces: list[str] = []
        generated = 0
        for chunk in stream_generate(
            model, tokenizer, prompt, max_tokens=max_tokens, sampler=sampler
        ):
            if ttft_ms is None:
                ttft_ms = (time.perf_counter() - started_at) * 1000
            pieces.append(chunk.text)
            generated = chunk.generation_tokens
        latency_ms = (time.perf_counter() - started_at) * 1000
        output = clean_output("".join(pieces))
        return {
            "ok": True,
            "outputText": output,
            "diagnostics": {
                "latencyMilliseconds": latency_ms,
                "timeToFirstTokenMilliseconds": ttft_ms or 0.0,
                "tokensPerSecond": (
                    generated / (latency_ms / 1000) if latency_ms > 0 else 0.0
                ),
                "generatedTokenCount": generated,
                "inputCharacterCount": len(text),
                "outputCharacterCount": len(output),
                "maxOutputTokenBudget": max_tokens,
            },
        }

    ready_message = f"{args.model} loaded in {load_seconds:.1f}s"

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError as error:
            emit({"ok": False, "error": f"bad json: {error}"})
            continue
        command = request.get("command")
        if command == "shutdown":
            break
        if command == "status":
            emit({"ok": True, "status": {"state": "ready", "message": ready_message}})
            continue
        if command == "warmup":
            try:
                run("warmup pass")
            except Exception as error:  # noqa: BLE001
                emit({"ok": False, "error": f"warmup failed: {error}"})
                traceback.print_exc(file=sys.stderr)
                continue
            emit({"ok": True, "status": {"state": "ready", "message": ready_message}})
            continue
        if command == "run":
            text = request.get("inputText") or ""
            if not text.strip():
                emit({"ok": False, "error": "empty inputText"})
                continue
            try:
                emit(run(text))
            except Exception as error:  # noqa: BLE001
                emit({"ok": False, "error": f"correction failed: {error}"})
                traceback.print_exc(file=sys.stderr)
            continue
        emit({"ok": False, "error": f"unknown command: {command}"})


if __name__ == "__main__":
    main()
