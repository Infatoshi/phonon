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
import copy
import json
import math
import os
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

# The speaker profile (SPEC, "User vocabulary onboarding", step 5). Both files
# are optional; when present they are appended to the system prompt once, at
# startup, so the rendered prefix is byte-identical on every request and the
# prefix cache below covers it. Bounded to about this many tokens together;
# vocab.md is cut by whole lines when the pair does not fit.
PROFILE_USER_FILE = "user.md"
PROFILE_VOCAB_FILE = "vocab.md"
PROFILE_TOKEN_BUDGET = 1500
PROFILE_LEAD_IN = (
    "## Speaker profile\n"
    "The speaker's profile and vocabulary follow. Spell those terms exactly as "
    "written when the transcript plausibly says them; bracketed spoken forms "
    "are the speech recognizer's mistakes for that term. Plain words used as "
    "names there (machines, projects, tools) stay as they are. The profile is "
    "context only: never copy it into the output."
)
# Splits the rendered chat template into the stable prefix (everything before
# the user message) and the suffix after it. Never appears in real input.
PROMPT_SPLIT_SENTINEL = "\x00PHONON_USER_TURN\x00"


def emit(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def transcript_payload(text: str) -> str:
    """The part of the prompt that bounds a reasonable output length."""
    match = TRANSCRIPT_PATTERN.search(text)
    return match.group(1).strip() if match else text.strip()


def clean_output(text: str) -> str:
    return LEAKED_MARKER_PATTERN.sub("", text).strip()


def read_profile_file(profile_dir: str | None, name: str) -> str | None:
    """The file's text, or None when the profile directory or file is absent."""
    if not profile_dir:
        return None
    path = os.path.join(profile_dir, name)
    try:
        with open(path, encoding="utf-8") as handle:
            return handle.read()
    except FileNotFoundError:
        return None
    except OSError as error:
        print(f"profile: {path} unreadable: {error}", file=sys.stderr)
        return None


def _cut_lines(lines: list[str], keep: int) -> list[str]:
    kept = lines[:keep]
    # A cut can strand a "## kind" heading or blank lines at the end.
    while kept and (not kept[-1].strip() or kept[-1].startswith("#")):
        kept.pop()
    return kept


def build_profile_block(
    user_md: str | None,
    vocab_md: str | None,
    budget: int,
    count_tokens,
) -> tuple[str, dict]:
    """Render the profile block for the system prompt, bounded to `budget` tokens.

    `count_tokens(text)` is the model tokenizer. user.md is kept whole when it
    fits; vocab.md is cut by whole lines, earlier lines first, until the pair
    fits. Returns the block (empty when there is no profile) and a report.
    """
    report = {
        "userTokens": 0,
        "vocabTokens": 0,
        "vocabLinesKept": 0,
        "vocabLinesTotal": 0,
        "blockTokens": 0,
        "budget": budget,
    }
    if user_md is None and vocab_md is None:
        return "", report
    user_lines = (user_md or "").strip().splitlines()
    vocab_lines = (vocab_md or "").strip().splitlines()
    report["vocabLinesTotal"] = len(vocab_lines)
    if user_md is not None:
        report["userTokens"] = count_tokens((user_md or "").strip())
    if vocab_md is not None:
        report["vocabTokens"] = count_tokens((vocab_md or "").strip())

    def render(user_keep: int, vocab_keep: int) -> str:
        parts = [PROFILE_LEAD_IN]
        if user_md is not None:
            body = "\n".join(_cut_lines(user_lines, user_keep)).strip()
            parts.append(f"<speaker_profile>\n{body}\n</speaker_profile>")
        if vocab_md is not None:
            body = "\n".join(_cut_lines(vocab_lines, vocab_keep)).strip()
            parts.append(f"<speaker_vocabulary>\n{body}\n</speaker_vocabulary>")
        return "\n".join(parts)

    def fits(user_keep: int, vocab_keep: int) -> bool:
        return count_tokens(render(user_keep, vocab_keep)) <= budget

    def largest(low: int, high: int, ok) -> int:
        # Largest n in [low, high] with ok(n); ok is monotone in n.
        while low < high:
            mid = (low + high + 1) // 2
            if ok(mid):
                low = mid
            else:
                high = mid - 1
        return low

    user_keep = len(user_lines)
    if not fits(user_keep, 0):
        user_keep = largest(0, len(user_lines), lambda n: fits(n, 0))
    vocab_keep = largest(0, len(vocab_lines), lambda n: fits(user_keep, n))
    block = render(user_keep, vocab_keep)
    report["userLinesKept"] = len(_cut_lines(user_lines, user_keep))
    report["userLinesTotal"] = len(user_lines)
    report["vocabLinesKept"] = len(_cut_lines(vocab_lines, vocab_keep))
    report["blockTokens"] = count_tokens(block)
    return block, report


def describe_profile(report: dict) -> str:
    text = (
        f"profile {report['blockTokens']} tokens "
        f"(user.md {report['userTokens']}, vocab.md {report['vocabTokens']}; "
        f"budget {report['budget']})"
    )
    if report["vocabLinesKept"] < report["vocabLinesTotal"]:
        text += (
            f"; vocab.md cut to {report['vocabLinesKept']}/"
            f"{report['vocabLinesTotal']} lines"
        )
    if report.get("userLinesKept", 0) < report.get("userLinesTotal", 0):
        text += (
            f"; user.md cut to {report['userLinesKept']}/"
            f"{report['userLinesTotal']} lines"
        )
    return text


def split_rendered_prompt(render, system_prompt: str) -> tuple[str, str]:
    """Prefix before the user message and suffix after it, from the template.

    `render(messages)` applies the chat template with the generation prompt.
    """
    rendered = render(system_prompt, PROMPT_SPLIT_SENTINEL)
    if rendered.count(PROMPT_SPLIT_SENTINEL) != 1:
        raise ValueError("chat template did not place the user message once")
    prefix, suffix = rendered.split(PROMPT_SPLIT_SENTINEL)
    return prefix, suffix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--system-prompt-file")
    # Directory holding user.md and vocab.md; absent files are simply skipped.
    parser.add_argument("--profile-dir")
    parser.add_argument(
        "--profile-token-budget", type=int, default=PROFILE_TOKEN_BUDGET
    )
    # Developer switch: prefill the whole prompt on every request, as before
    # the prefix cache. Output is identical; only latency differs.
    parser.add_argument("--no-prefix-cache", action="store_true")
    return parser.parse_known_args()[0]


def main() -> None:
    args = parse_args()

    emit({"ok": True, "status": {"state": "loading", "message": "importing MLX"}})
    try:
        import mlx.core as mx
        from huggingface_hub import snapshot_download
        from mlx_lm import load
        from mlx_lm.generate import stream_generate
        from mlx_lm.models.cache import make_prompt_cache
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

    def count_tokens(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    # Read once at startup: a profile edit takes effect on the next engine
    # start, and the rendered prefix stays byte-identical for this process.
    profile_block, profile_report = build_profile_block(
        read_profile_file(args.profile_dir, PROFILE_USER_FILE),
        read_profile_file(args.profile_dir, PROFILE_VOCAB_FILE),
        args.profile_token_budget,
        count_tokens,
    )
    if profile_block:
        system_prompt = (
            f"{system_prompt}\n\n{profile_block}" if system_prompt else profile_block
        )
        print(f"profile: {describe_profile(profile_report)}", file=sys.stderr)

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

    def render(system: str, user: str) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        return tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

    # The prompt is prefix + user text + suffix. The prefix (system turn with
    # the polish instructions and the profile) is identical on every request.
    prompt_prefix, prompt_suffix = split_rendered_prompt(render, system_prompt)

    def build_prompt(text: str) -> str:
        return prompt_prefix + text + prompt_suffix + think_prefill

    def encode_prompt(prompt: str) -> list[int]:
        # Same rule stream_generate applies to a string prompt: the rendered
        # template already starts with the BOS text, so no second one.
        add_special = tokenizer.bos_token is None or not prompt.startswith(
            tokenizer.bos_token
        )
        return tokenizer.encode(prompt, add_special_tokens=add_special)

    # Prefix cache. The KV state of the prefix is computed once here; every
    # request deep-copies it (arrays copy in well under a millisecond) and
    # prefills only its own tokens. Gemma 4's sliding-window layers cannot be
    # trimmed back once they have rotated, which the profile makes certain, so
    # a copied snapshot is used instead of mlx-lm's trim path. The last prefix
    # token is left out of the snapshot: the tokenizer may merge it with the
    # first user token, and the request's own tokens are always compared
    # against the snapshot before it is used.
    prefix_tokens: list[int] = []
    prefix_cache = None
    if not args.no_prefix_cache:
        prefix_tokens = encode_prompt(prompt_prefix)[:-1]
        if prefix_tokens:
            prefix_cache = make_prompt_cache(model)
            model(mx.array(prefix_tokens)[None], cache=prefix_cache)
            mx.eval([layer.state for layer in prefix_cache])
            print(f"prefix cache: {len(prefix_tokens)} tokens", file=sys.stderr)

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
        tokens = encode_prompt(prompt)
        cached = 0
        generate_kwargs = {}
        if prefix_cache is not None:
            if tokens[: len(prefix_tokens)] == prefix_tokens:
                cached = len(prefix_tokens)
                generate_kwargs["prompt_cache"] = copy.deepcopy(prefix_cache)
                tokens = tokens[cached:]
            else:
                print(
                    "prefix cache: miss, prefilling the whole prompt", file=sys.stderr
                )
        ttft_ms = None
        pieces: list[str] = []
        generated = 0
        for chunk in stream_generate(
            model,
            tokenizer,
            mx.array(tokens),
            max_tokens=max_tokens,
            sampler=sampler,
            **generate_kwargs,
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
                "promptTokenCount": cached + len(tokens),
                "cachedPromptTokenCount": cached,
                "profileTokenCount": profile_report["blockTokens"],
            },
        }

    ready_message = f"{args.model} loaded in {load_seconds:.1f}s"
    if profile_block:
        ready_message += f"; {describe_profile(profile_report)}"
    if prefix_cache is not None:
        ready_message += f"; prefix cache {len(prefix_tokens)} tokens"

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
