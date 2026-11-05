#!/usr/bin/env python3
"""JSONL correction sidecar: load the pinned MLX correction model, polish transcripts.

Speaks the same line protocol the engine already used for the correction stage:
  stdin:  {"command":"warmup"} | {"command":"run","inputText":"…","context":{…}}
          {"command":"status"} | {"command":"shutdown"} | {"command":"prewarm"}
          {"command":"stream_commit","inputText":"…","context":{…}} | {"command":"stream_reset"}
          (`context.app_category` picks a refiner's writing mode; see WRITING_MODE_BY_CATEGORY)
  stdout: {"ok":true,"status":{"state":"ready","message":"…"}}
          {"ok":true,"outputText":"…","diagnostics":{…}}
          {"ok":true,"stream":{…}}   (after a stream_commit job; carries no outputText;
          with `context.draft_id` also draftId, draftSeq, draftText, draftComplete: live text)
          {"ok":false,"error":"…"}

Streaming refine (refiner mode only): while the speaker is still talking the engine sends
stream_commit with the envelope of the transcript so far (closed speech segments). The sidecar
refines that transcript in the background, keeping the prompt in a rollback-able model state and
the output as a draft. A newer commit replaces one that has not started; a run aborts a running
commit between forward passes. The run that follows is the greedy output of its own prompt
whatever the draft says (stream_decode.py), so the inserted text does not depend on streaming.
The state lives in this process (MLX, stream_refiner.Engine) or, with the native refiner engine,
in its per-dictation session (stream_decode.PeEngine).
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
from difflib import SequenceMatcher

# Gemma 4 opens a thought channel by default even when the template omits
# `<|think|>`. Prefilling a closed thought channel puts the model straight into
# the answer channel, which is what a formatter needs: short, literal output.
NO_THINK_PREFILL = "<|channel>thought\nNo thinking needed.\n<channel|>"

# Hard ceiling on generated tokens, so a runaway expansion cannot decode until
# the model's own limit and feel frozen. The per-request budget below already
# scales with the spoken text; the ceiling is only a backstop. It was 256 until
# 2026-09-15, which silently cut every dictation longer than about 230 words
# (20 of 145 real holdout dictations) to its first 256 tokens.
OUTPUT_TOKEN_CEILING = 2048
OUTPUT_TOKEN_FLOOR = 48
# The native engine's context (pe.mm CTX). A streamed request whose prompt plus output budget
# would pass it is refined in parts (StreamSession freezes the older sentences), on either
# backend, so the text does not depend on which one serves it.
STREAM_CONTEXT_TOKENS = 2048
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

# A model directory holding this file is a trained refiner (ml/, phonon-bench
# format) rather than an instruction-following polisher: it gets the exact
# system prompt and vocabulary line it was trained on, the bare transcript as
# the user turn, and the word-count guard. Byte-identical to
# ml/research/bench_v0 prompts.py; a different wording is a different model.
REFINER_MARKER_FILE = "phonon_refiner.json"
REFINER_SYSTEM_PROMPT = (
    "Rewrite the raw dictation transcript into the exact text the speaker intended. "
    "Fix recognition errors and technical terms, keep the speaker's wording and casing style, "
    "do not add or drop content."
)
REFINER_VOCAB_PREFIX = "Vocabulary the speaker may use: "
# Dictionary lines that become vocabulary, most confident first. Term lines are
# comma-separated; pair lines are `spoken => canonical` joined by `;` and
# contribute the canonical side.
REFINER_VOCAB_LINES = (
    ("screen_confirmed_terms", False),
    ("phonetic_matches", True),
    ("exact_replacements", True),
    ("likely_terms", False),
    ("canonical_terms", False),
)
DICTIONARY_PATTERN = re.compile(
    r"<phonon_dictionary>(.*?)</phonon_dictionary>", re.DOTALL
)
# Word-count guard and output cap (ml/AGENTS.md: on for every reported number).
GUARD_TOK_MULT = 1.5
GUARD_TOK_ADD = 32
GUARD_MAX_RATIO = 2.0
GUARD_MIN_RATIO = 0.4
GUARD_MIN_WORDS = 8
# Writing modes (SPEC, "Intended text and writing modes"). A refiner trained
# with mode lines lists them in its marker as "writing_modes": {mode: line};
# its system prompt then carries the line of the target app's mode between the
# fixed instructions and the vocabulary line, as in training. A refiner without
# the key gets no mode line (its prompt is unchanged). The marker may override
# this category -> mode table with "writing_modes_by_app_category".
WRITING_MODE_BY_CATEGORY = {
    "chat": "casual",  # personal chat: Discord, iMessage, WhatsApp
    "work_chat": "formal",  # Slack, Teams, Google Chat
    "email": "formal",
    "ai_prompt": "prompt",
    "code": "prompt",  # terminals and editors: mostly prompts to a coding agent
    "docs": "minimal",
    "other": "minimal",
}
# A refiner directory holding this file runs on the native refiner engine (a
# Metal megakernel with a speculative drafter, `pe serve`) instead of MLX:
# {"binary": path, "target": GGUF path, "draft": GGUF path, "draft_kind": str},
# paths relative to the directory. The prompt, budget and guard are unchanged.
# "binary" is optional: a downloaded refiner has none and runs on the engine
# bundled with the app (`--engine-binary`, shaders in `--engine-shaders`).
REFINER_ENGINE_FILE = "phonon_engine.json"
# The engine must turn this into real text before it serves anything;
# otherwise the MLX weights in the same directory take over. The rule matches
# the engine's own prime check (`llm_prime_passed`): no collapse, no runaway,
# and at least half of the words kept (a refiner may rewrite "fluid voice").
ENGINE_PROBE_TEXT = "hello fluid voice startup prime"


# The engine's JSON-lines channel. isolate_protocol() moves it off fds 0 and 1,
# so no library can read a request or print into a reply line.
PROTOCOL_IN = None
PROTOCOL_OUT = None


def stream_refine_enabled() -> bool:
    """Streaming refine: the engine sets PHONON_STREAM_REFINE (1/true/on) from settings.json
    `stream_refine` or its own environment."""
    value = os.environ.get("PHONON_STREAM_REFINE", "")
    return value.strip().lower() in ("1", "true", "on")


def emit(obj: dict) -> None:
    out = PROTOCOL_OUT or sys.stdout
    out.write(json.dumps(obj, ensure_ascii=False) + "\n")
    out.flush()


def isolate_protocol() -> None:
    """Keep private copies of the engine's stdin and stdout for the protocol,
    then point fd 0 at /dev/null and fd 1 at stderr. transformers asks on stdin
    whether to trust a repository's custom code and prints the question on
    stdout; on the protocol pipes that question swallowed the engine's warmup
    request and corrupted the next reply line, and startup hung."""
    global PROTOCOL_IN, PROTOCOL_OUT
    PROTOCOL_IN = os.fdopen(os.dup(0), "r", encoding="utf-8")
    PROTOCOL_OUT = os.fdopen(os.dup(1), "w", encoding="utf-8")
    devnull = os.open(os.devnull, os.O_RDONLY)
    os.dup2(devnull, 0)
    os.close(devnull)
    sys.stdout.flush()
    os.dup2(2, 1)
    sys.stdin = open(os.devnull, encoding="utf-8")  # noqa: SIM115 - open for the process's lifetime
    sys.stdout = sys.stderr


def transcript_payload(text: str) -> str:
    """The part of the prompt that bounds a reasonable output length."""
    match = TRANSCRIPT_PATTERN.search(text)
    return match.group(1).strip() if match else text.strip()


def with_transcript(text: str, words: list[str]) -> str:
    """The envelope `text` with its transcript replaced by `words` (dictionary kept)."""
    spoken = " ".join(words)
    if not TRANSCRIPT_PATTERN.search(text):
        return spoken
    return TRANSCRIPT_PATTERN.sub(
        lambda _: f"<transcript>\n{spoken}\n</transcript>", text, count=1
    )


def clean_output(text: str) -> str:
    return LEAKED_MARKER_PATTERN.sub("", text).strip()


def refiner_vocabulary(text: str) -> list[str]:
    """Distinct dictionary terms from the engine's envelope, most confident first,
    capped by the adaptive list budget min(80, max(30, 3 x words)) the refiner
    was evaluated with."""
    match = DICTIONARY_PATTERN.search(text)
    if not match:
        return []
    fields: dict[str, str] = {}
    for line in match.group(1).splitlines():
        key, sep, value = line.partition(":")
        if sep:
            fields[key.strip()] = value.strip()
    terms: list[str] = []
    for key, pairs in REFINER_VOCAB_LINES:
        value = fields.get(key, "")
        if pairs:
            terms += [pair.partition("=>")[2] for pair in value.split(";")]
        else:
            terms += value.split(",")
    seen: set[str] = set()
    vocab = []
    for term in (t.strip() for t in terms):
        if term and term not in seen:
            seen.add(term)
            vocab.append(term)
    words = len(transcript_payload(text).split())
    return vocab[: min(80, max(30, 3 * words))]


TERM_SHAPE = re.compile(r"[A-Z].*[A-Z0-9]|.*\d.*|.+-.+")
WORD_PARTS = re.compile(r"^(\W*)(.*?)(\W*)$", re.DOTALL)


def _norm(word: str) -> str:
    return "".join(c for c in word.lower() if c.isalnum())


def restore_terms(spoken: str, output: str, vocab: list[str]) -> str:
    """Put back a one-word dictionary term the speaker said where the refiner wrote another
    term in its place ("native mtp" -> "native MTK", "the mlp" -> "the MMLU-Pro"; MTP and
    MLP in the dictionary; owner dictation 2026-11-04 3:25 pm, a long one). Only a
    one-for-one swap whose new word reads as a term (two capitals, a digit or a hyphen) and
    was not said; everything else in the output is left as it is."""
    terms = {
        _norm(t): t.strip()
        for t in reversed(vocab)
        if t.strip() and " " not in t.strip()
    }
    if not terms:
        return output
    said = [_norm(w) for w in spoken.split()]
    said_set = set(said)
    spans = [m.span() for m in re.finditer(r"\S+", output)]
    words = [output[a:b] for a, b in spans]
    normed = [_norm(w) for w in words]
    fixes: dict[int, str] = {}
    ops = SequenceMatcher(a=said, b=normed, autojunk=False).get_opcodes()
    for op, i1, i2, j1, j2 in ops:
        if op != "replace" or i2 - i1 != j2 - j1:
            continue
        for i, j in zip(range(i1, i2), range(j1, j2)):
            term = terms.get(said[i])
            lead, core, trail = WORD_PARTS.match(words[j]).groups()
            if (
                term
                and normed[j] not in said_set
                and normed[j] != _norm(term)
                and TERM_SHAPE.fullmatch(core)
            ):
                fixes[j] = lead + term + trail
    if not fixes:
        return output
    out, last = [], 0
    for j, (a, b) in enumerate(spans):
        if j in fixes:
            out += [output[last:a], fixes[j]]
            last = b
    return "".join(out) + output[last:]


def refiner_system_prompt(vocab: list[str], mode_line: str = "") -> str:
    system = REFINER_SYSTEM_PROMPT
    if mode_line:
        system += "\n" + mode_line
    if not vocab:
        return system
    return system + "\n" + REFINER_VOCAB_PREFIX + ", ".join(vocab)


def read_writing_modes(model_dir: str) -> tuple[dict[str, str], dict[str, str]]:
    """The refiner's mode lines and the app category -> mode table; ({}, {}) for a
    refiner trained without modes or an unreadable marker."""
    try:
        with open(os.path.join(model_dir, REFINER_MARKER_FILE)) as handle:
            marker = json.load(handle)
    except (OSError, ValueError):
        return {}, {}
    lines = marker.get("writing_modes") if isinstance(marker, dict) else None
    if not isinstance(lines, dict):
        return {}, {}
    lines = {k: v for k, v in lines.items() if isinstance(v, str) and v.strip()}
    table = dict(WRITING_MODE_BY_CATEGORY)
    override = marker.get("writing_modes_by_app_category")
    if isinstance(override, dict):
        table.update({k: v for k, v in override.items() if isinstance(v, str)})
    return lines, table


def writing_mode(
    lines: dict[str, str], table: dict[str, str], context
) -> tuple[str | None, str]:
    """(mode, line) for a request's `context`: the app category's mode, the
    "other" mode when the request names no app, no mode for a refiner without
    mode lines. A `writing_mode` the user picked (Settings > Advanced > Writing style)
    wins over the category when the refiner has that mode."""
    if not lines:
        return None, ""
    forced = context.get("writing_mode") if isinstance(context, dict) else None
    if isinstance(forced, str) and forced in lines:
        return forced, lines[forced]
    category = context.get("app_category") if isinstance(context, dict) else None
    mode = table.get(category or "other") or table.get("other")
    return (mode, lines[mode]) if mode in lines else (None, "")


def draft_request(context) -> dict | None:
    """Live text: a stream_commit whose context has `draft_id` (and the engine's `draft_seq`,
    the commit's place in the stream) wants the draft's text back, tagged with both. Absent,
    the reply carries no text and costs nothing extra."""
    if not isinstance(context, dict):
        return None
    wanted = context.get("draft_id")
    if not isinstance(wanted, str) or not wanted:
        return None
    tag = {"draftId": wanted}
    seq = context.get("draft_seq")
    if isinstance(seq, int) and not isinstance(seq, bool):
        tag["draftSeq"] = seq
    return tag


def draft_fields(
    wanted: dict | None,
    ids: list[int] | None,
    budget: int,
    decode,
    spoken: str,
    head: list[list[int]] = (),
    vocab: list[str] = (),
) -> dict:
    """Live-text keys for a stream_commit reply: the background draft cleaned and guarded
    exactly like a run's output (run_streamed), so the overlay and the paste share one path.
    `head`: outputs of a long dictation's frozen parts, which come before the draft.
    `vocab`: the request's dictionary terms (restore_terms).
    `draftComplete` is false when the draft stopped at the output budget instead of the end
    of its answer."""
    if wanted is None or ids is None:
        return {}
    text, _ = guard_output(spoken, decode_pieces(decode, [*head, ids]))
    text = restore_terms(spoken, text, list(vocab))
    return {**wanted, "draftText": text, "draftComplete": len(ids) < budget}


def decode_pieces(decode, pieces: list[list[int]]) -> str:
    """A streamed answer's parts (frozen heads, then the tail) as one text."""
    return " ".join(t for t in (clean_output(decode(p)) for p in pieces) if t)


def refiner_token_budget(input_tokens: int) -> int:
    return max(
        8, min(OUTPUT_TOKEN_CEILING, int(GUARD_TOK_MULT * input_tokens) + GUARD_TOK_ADD)
    )


def guard_output(spoken: str, output: str) -> tuple[str, bool]:
    """The spoken text instead of a runaway or a collapsed output."""
    spoken_words = len(spoken.split())
    output_words = len(output.split())
    if spoken_words == 0:
        return output, False
    if output_words > GUARD_MAX_RATIO * spoken_words:
        return spoken, True
    if spoken_words > GUARD_MIN_WORDS and output_words < GUARD_MIN_RATIO * spoken_words:
        return spoken, True
    return output, False


def read_engine_config(model_dir: str) -> dict | None:
    path = os.path.join(model_dir, REFINER_ENGINE_FILE)
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def engine_command(
    model_dir: str,
    config: dict,
    engine_binary: str | None = None,
    engine_shaders: str | None = None,
) -> tuple[list[str], dict | None] | None:
    """The `pe serve` command and environment for this refiner, or None when no
    engine binary exists. A local directory that names its own binary keeps it
    (its shaders sit next to it); otherwise the app's bundled engine runs, with
    its shader directory passed as PE_DIR."""
    env = None
    if config.get("binary"):
        binary = os.path.join(model_dir, config["binary"])
    elif engine_binary:
        binary = engine_binary
        if engine_shaders:
            env = dict(os.environ, PE_DIR=engine_shaders)
    else:
        return None
    if not os.path.isfile(binary):
        return None
    command = [binary, os.path.join(model_dir, config["target"]), "serve"]
    if config.get("draft"):
        command += [
            "--draft",
            config.get("draft_kind", "dflash2n"),
            os.path.join(model_dir, config["draft"]),
        ]
    return command, env


def engine_probe_passed(output: str) -> bool:
    spoken = re.findall(r"[a-z0-9]+", ENGINE_PROBE_TEXT)
    words = re.findall(r"[a-z0-9]+", output.lower())
    kept = sum(1 for word in spoken if word in words)
    return len(spoken) - 1 <= len(words) <= len(spoken) + 2 and 2 * kept >= len(spoken)


def has_mlx_weights(model_dir: str) -> bool:
    """config.json plus every safetensors shard the index names (or one file)."""
    if not os.path.isfile(os.path.join(model_dir, "config.json")):
        return False
    index = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.isfile(index):
        try:
            with open(index, encoding="utf-8") as handle:
                shards = set(json.load(handle).get("weight_map", {}).values())
        except (OSError, ValueError):
            return False
        return bool(shards) and all(
            os.path.isfile(os.path.join(model_dir, shard)) for shard in shards
        )
    return any(
        name.endswith(".safetensors") and os.path.isfile(os.path.join(model_dir, name))
        for name in os.listdir(model_dir)
    )


def snapshot_is_complete(model_dir: str) -> bool:
    """Every file this directory needs is on disk. An interrupted first download
    leaves a snapshot with some files missing; that one is downloaded again."""
    config = None
    try:
        config = read_engine_config(model_dir)
    except (OSError, ValueError):
        return False
    if config is not None:
        for key in ("target", "draft"):
            if config.get(key) and not os.path.isfile(
                os.path.join(model_dir, config[key])
            ):
                return False
    return has_mlx_weights(model_dir)


def _directory_bytes(path: str) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.lstat(os.path.join(root, name)).st_size
            except OSError:
                pass
    return total


def _gigabytes(size: int) -> str:
    return f"{size / 1e9:.1f} GB"


def resolve_model_dir(model: str, revision: str) -> str:
    """A local directory as it is; otherwise the pinned snapshot, from the cache
    when it is complete (no network at all), else downloaded with progress."""
    if os.path.isdir(model):
        return model
    from huggingface_hub import snapshot_download

    try:
        cached = snapshot_download(model, revision=revision, local_files_only=True)
        if snapshot_is_complete(cached):
            return cached
    except Exception:  # noqa: BLE001, S110 - not cached yet
        pass
    return download_with_progress(model, revision)


def download_with_progress(model: str, revision: str) -> str:
    """snapshot_download, reporting bytes on disk once a second. Progress is the
    growth of the repository's blob directory in the Hugging Face cache, so it
    counts only bytes that actually arrived."""
    import threading

    from huggingface_hub import HfApi, constants, snapshot_download

    # Xet assembles each file outside the blob directory and moves it in at
    # the end, so byte progress would stall on the first large file; plain
    # HTTPS writes the blob as bytes arrive (and was faster in a 135 MB test).
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    constants.HF_HUB_DISABLE_XET = True
    total = 0
    try:
        info = HfApi().model_info(model, revision=revision, files_metadata=True)
        total = sum(sibling.size or 0 for sibling in info.siblings or [])
    except Exception:  # noqa: BLE001, S110 - progress without a total is still progress
        pass
    blobs = os.path.join(
        constants.HF_HUB_CACHE, "models--" + model.replace("/", "--"), "blobs"
    )
    start = _directory_bytes(blobs)
    stop = threading.Event()

    def report() -> None:
        while not stop.wait(1.0):
            done = max(0, _directory_bytes(blobs) - start)
            status = {"state": "loading"}
            if total:
                status["message"] = (
                    f"Downloading the correction model (first launch only): "
                    f"{_gigabytes(min(done, total))} of {_gigabytes(total)}"
                )
                status["progress"] = min(1.0, done / total)
            else:
                status["message"] = (
                    f"Downloading the correction model (first launch only): "
                    f"{_gigabytes(done)}"
                )
            emit({"ok": True, "status": status})

    reporter = threading.Thread(target=report, daemon=True)
    reporter.start()
    try:
        return snapshot_download(model, revision=revision)
    finally:
        stop.set()
        reporter.join(timeout=2)


def think_prefill_for(tokenizer) -> str:
    # The prefill closes Gemma 4's thought channel. Any model whose chat
    # template has no channel mechanism would receive it as literal prompt
    # text, so only prefill when the template actually speaks that protocol.
    template = getattr(tokenizer, "chat_template", None) or ""
    return NO_THINK_PREFILL if "channel" in template else ""


def render_chat(tokenizer, system: str, user: str) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    kwargs = {}
    # Qwen3-style templates open a thinking block unless told not to; a
    # formatter wants the answer straight away.
    if "enable_thinking" in (getattr(tokenizer, "chat_template", None) or ""):
        kwargs["enable_thinking"] = False
    return tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False, **kwargs
    )


class RefinerEngine:
    """The native refiner engine as a child process: one JSON line per request
    on its stdin, one reply line on its stdout. It exits when stdin closes, so
    it never outlives this sidecar."""

    def __init__(
        self,
        model_dir: str,
        config: dict,
        command: list[str] | None = None,
        env: dict | None = None,
    ):
        import subprocess

        if command is None:
            spec = engine_command(model_dir, config)
            if spec is None:
                raise RuntimeError("no refiner engine binary")
            command, env = spec
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
        )
        ready = self._reply()
        if not ready.get("ready"):
            raise RuntimeError(f"refiner engine did not start: {ready}")
        self.init_seconds = float(ready.get("init_s", 0.0))

    def close(self) -> None:
        try:
            self.process.kill()
            self.process.wait(timeout=5)
        except Exception:  # noqa: BLE001, S110 - already gone
            pass

    def _reply(self) -> dict:
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"refiner engine exited (code {self.process.poll()})")
        return json.loads(line)

    def call(self, request: dict) -> dict:
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        return self._reply()

    def set_prefix(self, text: str) -> int:
        return int(self.call({"prefix": text})["prefix_tokens"])

    def generate(self, prompt: str, max_tokens: int) -> dict:
        return self.call({"max_tokens": max_tokens, "prompt": prompt})


def register_vendored_models() -> None:
    """Make mlx-lm load architectures newer than the pinned release (models/)."""
    import importlib
    import importlib.util

    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    if not os.path.isdir(here):
        return
    # These files live inside the signed app bundle; a __pycache__ written next
    # to them breaks the bundle's code signature.
    sys.dont_write_bytecode = True
    for file in sorted(os.listdir(here)):
        if not file.endswith(".py"):
            continue
        name = f"mlx_lm.models.{file[:-3]}"
        try:
            importlib.import_module(name)
            continue
        except ModuleNotFoundError:
            pass
        spec = importlib.util.spec_from_file_location(name, os.path.join(here, file))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)


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
    # Developer comparisons: send only the text inside <transcript> to a model
    # trained on bare transcripts. Shipped builds never pass this.
    parser.add_argument("--transcript-only", action="store_true")
    # The native refiner engine bundled with the app and its shader directory.
    # Used for a refiner whose phonon_engine.json names no binary of its own.
    parser.add_argument("--engine-binary")
    parser.add_argument("--engine-shaders")
    return parser.parse_known_args()[0]


def main() -> None:
    args = parse_args()
    isolate_protocol()

    # Status messages are shown as they are in the startup window; the ready
    # message carries the developer detail (model, engine, prefix cache).
    emit(
        {
            "ok": True,
            "status": {"state": "loading", "message": "Loading the correction model"},
        }
    )
    try:
        import huggingface_hub  # noqa: F401 - fail here, not mid-download
        import mlx.core as mx
        from mlx_lm import load
        from mlx_lm.generate import stream_generate
        from mlx_lm.models.cache import make_prompt_cache
        from mlx_lm.sample_utils import make_sampler

        register_vendored_models()
    except Exception as error:  # noqa: BLE001 - report to the engine, never crash silently
        emit({"ok": False, "error": f"correction runtime import failed: {error}"})
        return

    print(f"model: {args.model}@{args.revision[:7]}", file=sys.stderr)
    try:
        # A local directory (developer model comparisons, the user's own
        # refiner) is used as it is.
        local_dir = resolve_model_dir(args.model, args.revision)
    except Exception as error:  # noqa: BLE001
        emit({"ok": False, "error": f"correction model download failed: {error}"})
        return
    refiner = os.path.isfile(os.path.join(local_dir, REFINER_MARKER_FILE))
    mode_lines, mode_table = read_writing_modes(local_dir) if refiner else ({}, {})
    engine_config = read_engine_config(local_dir) if refiner else None
    engine_spec = (
        engine_command(
            local_dir, engine_config, args.engine_binary, args.engine_shaders
        )
        if engine_config is not None
        else None
    )

    started = time.perf_counter()
    engine = None
    engine_fallback = None
    model = None
    if engine_spec is not None:
        emit(
            {
                "ok": True,
                "status": {
                    "state": "loading",
                    "message": "Starting the correction engine",
                },
            }
        )
        try:
            from mlx_lm.utils import load_tokenizer

            tokenizer = load_tokenizer(local_dir)
            engine = RefinerEngine(local_dir, engine_config, *engine_spec)
            # One literal request before anything else: a Mac whose GPU the
            # engine has never been tuned on must produce real text here, or
            # the MLX weights in the same directory take over.
            probe_prompt = render_chat(
                tokenizer, REFINER_SYSTEM_PROMPT, ENGINE_PROBE_TEXT
            ) + think_prefill_for(tokenizer)
            probe = engine.generate(
                probe_prompt,
                refiner_token_budget(
                    len(tokenizer.encode(ENGINE_PROBE_TEXT, add_special_tokens=False))
                ),
            )
            if not probe.get("ok") or not engine_probe_passed(probe.get("text", "")):
                raise RuntimeError(f"probe request failed: {probe}")
        except Exception as error:  # noqa: BLE001
            traceback.print_exc(file=sys.stderr)
            if engine is not None:
                engine.close()
                engine = None
            if not has_mlx_weights(local_dir):
                emit({"ok": False, "error": f"refiner engine start failed: {error}"})
                return
            engine_fallback = str(error)
            emit(
                {
                    "ok": True,
                    "status": {
                        "state": "loading",
                        "message": (
                            "The native engine is unavailable on this Mac; "
                            "loading the MLX correction model"
                        ),
                    },
                }
            )
    if engine_config is not None and engine_spec is None:
        engine_fallback = "no engine binary"
    if engine is None:
        emit(
            {
                "ok": True,
                "status": {
                    "state": "loading",
                    "message": "Loading the correction model weights",
                },
            }
        )
        try:
            model, tokenizer = load(local_dir)
        except Exception as error:  # noqa: BLE001
            emit({"ok": False, "error": f"correction model load failed: {error}"})
            traceback.print_exc(file=sys.stderr)
            return
    load_seconds = time.perf_counter() - started

    system_prompt = ""
    if args.system_prompt_file and not refiner:
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
    # A refiner was trained without the profile block; its per-request
    # vocabulary line carries the dictionary instead.
    profile_dir = None if refiner else args.profile_dir
    profile_block, profile_report = build_profile_block(
        read_profile_file(profile_dir, PROFILE_USER_FILE),
        read_profile_file(profile_dir, PROFILE_VOCAB_FILE),
        args.profile_token_budget,
        count_tokens,
    )
    if profile_block:
        system_prompt = (
            f"{system_prompt}\n\n{profile_block}" if system_prompt else profile_block
        )
        print(f"profile: {describe_profile(profile_report)}", file=sys.stderr)

    sampler = make_sampler(temp=0.0)

    think_prefill = think_prefill_for(tokenizer)

    # mlx-lm's TokenizerWrapper.detokenizer is a property that builds a brand new
    # streaming detokenizer on every access, and building one walks this model's
    # 262144-entry vocabulary in Python. stream_generate reads it once per call,
    # so a resident sidecar was paying ~134 ms of table building on every
    # utterance, which was most of the correction stage's time to first token.
    # A detokenizer's reset() clears the whole of its mutable state (offset,
    # pending bytes, text, tokens) and the id-to-piece table it rebuilds is
    # immutable, so handing back one reset instance is indistinguishable from
    # handing back a fresh one.
    if engine is None:
        shared_detokenizer = tokenizer.detokenizer

        def detokenizer_property(_wrapper):
            shared_detokenizer.reset()
            return shared_detokenizer

        type(tokenizer).detokenizer = property(detokenizer_property)

    def render(system: str, user: str) -> str:
        return render_chat(tokenizer, system, user)

    # The prompt is prefix + user text + suffix. The prefix (system turn with
    # the polish instructions and the profile) is identical on every request.
    # A refiner's system turn ends in the per-request vocabulary line, so only
    # the template up to the end of its fixed instructions is shared.
    if refiner:
        prompt_prefix = split_rendered_prompt(
            lambda system, _user: render(system, "-"),
            REFINER_SYSTEM_PROMPT + PROMPT_SPLIT_SENTINEL,
        )[0]
    else:
        prompt_prefix, prompt_suffix = split_rendered_prompt(render, system_prompt)

    def build_prompt(text: str, mode_line: str = "") -> str:
        if refiner:
            system = refiner_system_prompt(refiner_vocabulary(text), mode_line)
            return render(system, transcript_payload(text)) + think_prefill
        if args.transcript_only:
            text = transcript_payload(text)
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
    engine_prefix_tokens = 0
    if engine is not None:
        # The engine keeps its own prefix state (llama.cpp KV + conv, drafter
        # rows) and checks each request's tokens against it the same way.
        if not args.no_prefix_cache:
            engine_prefix_tokens = engine.set_prefix(prompt_prefix)
            print(
                f"prefix cache: {engine_prefix_tokens} tokens (engine)", file=sys.stderr
            )
    elif not args.no_prefix_cache:
        prefix_tokens = encode_prompt(prompt_prefix)[:-1]
        if prefix_tokens:
            prefix_cache = make_prompt_cache(model)
            model(mx.array(prefix_tokens)[None], cache=prefix_cache)
            mx.eval([layer.state for layer in prefix_cache])
            print(f"prefix cache: {len(prefix_tokens)} tokens", file=sys.stderr)

    def output_budget(text: str) -> int:
        spoken = transcript_payload(text)
        spoken_tokens = len(tokenizer.encode(spoken)) if spoken else 0
        if refiner:
            return refiner_token_budget(spoken_tokens)
        # A clean-up pass returns roughly the input again; allow headroom for
        # expanded orthography, then stop.
        budget = math.ceil(spoken_tokens * 1.8) + 24
        return max(OUTPUT_TOKEN_FLOOR, min(OUTPUT_TOKEN_CEILING, budget))

    def run_engine(text: str, mode: tuple = (None, "")) -> dict:
        prompt = build_prompt(text, mode[1])
        max_tokens = output_budget(text)
        started_at = time.perf_counter()
        reply = engine.generate(prompt, max_tokens)
        latency_ms = (time.perf_counter() - started_at) * 1000
        if stream is not None:
            stream_reset(lost=True)  # a one-shot request ends the engine's session
        if not reply.get("ok"):
            raise RuntimeError(f"refiner engine: {reply}")
        output, guard_fired = guard_output(
            transcript_payload(text), clean_output(reply["text"])
        )
        output = restore_terms(
            transcript_payload(text), output, refiner_vocabulary(text)
        )
        generated = int(reply["tokens"])
        return {
            "ok": True,
            "outputText": output,
            "diagnostics": {
                "latencyMilliseconds": latency_ms,
                # the engine returns the whole output at once
                "timeToFirstTokenMilliseconds": latency_ms,
                "tokensPerSecond": (
                    generated / (latency_ms / 1000) if latency_ms > 0 else 0.0
                ),
                "generatedTokenCount": generated,
                "inputCharacterCount": len(text),
                "outputCharacterCount": len(output),
                "maxOutputTokenBudget": max_tokens,
                "promptTokenCount": int(reply["prompt_tokens"]),
                "cachedPromptTokenCount": (
                    engine_prefix_tokens if reply.get("prefix_hit") else 0
                ),
                "profileTokenCount": profile_report["blockTokens"],
                "refiner": refiner,
                "writingMode": mode[0],
                "guardFired": guard_fired,
                "engine": "native",
                "draftedTokenCount": int(reply.get("drafted", 0)),
                "acceptedDraftTokenCount": int(reply.get("accepted", 0)),
            },
        }

    # Streaming refine (refiner mode): a rollback-able model state fed by stream_commit while
    # the speaker talks: a second cache over the MLX weights, or the native engine's session.
    # Only the refiner's prompt is streamed; a Gemma polisher ignores stream_commit. Set up
    # only when the engine turned streaming refine on (PHONON_STREAM_REFINE).
    stream = None
    stream_prefix: list[int] = []
    if refiner and stream_refine_enabled():
        try:  # run as sidecar/polish_server.py (sidecar dir on sys.path) or as a package
            from stream_decode import (
                PE_LADDER,
                Aborted,
                EngineError,
                PeEngine,
                StreamSession,
            )
        except ImportError:
            from .stream_decode import (
                PE_LADDER,
                Aborted,
                EngineError,
                PeEngine,
                StreamSession,
            )

        class EnvelopePrompter:
            """Refiner prompt pieces for an engine envelope (vocabulary + transcript), with
            the writing-mode line of the request being served (set before each job)."""

            mode_line = ""

            def prompt_tokens(self, text: str) -> list[int]:
                return encode_prompt(build_prompt(text, self.mode_line))

            def open_prompt_tokens(self, text: str) -> list[int]:
                system = refiner_system_prompt(refiner_vocabulary(text), self.mode_line)
                rendered = render(
                    system, transcript_payload(text) + PROMPT_SPLIT_SENTINEL
                )
                return encode_prompt(rendered.split(PROMPT_SPLIT_SENTINEL)[0])

            def budget(self, text: str) -> int:
                return output_budget(text)

            def text_tokens(self, text: str) -> list[int]:
                return tokenizer.encode(
                    " " + transcript_payload(text), add_special_tokens=False
                )

            # Long dictations (StreamSession freezes a head when a request stops fitting).
            def fits(self, text: str) -> bool:
                return (
                    len(self.prompt_tokens(text)) + self.budget(text)
                    <= STREAM_CONTEXT_TOKENS
                )

            def words(self, text: str) -> list[str]:
                return transcript_payload(text).split()

            def with_words(self, text: str, words: list[str]) -> str:
                return with_transcript(text, words)

        stream_prefix = encode_prompt(prompt_prefix)[:-1]
        eos = getattr(tokenizer, "eos_token_ids", None) or {tokenizer.eos_token_id}
        if engine is not None:
            # The session opens from the same prefix tokens the engine's one-shot prefix cache
            # holds, so an open restores that state instead of prefilling it.
            stream_engine = PeEngine(engine.call, stream_prefix, set(eos))
        else:
            try:
                from stream_refiner import Engine
            except ImportError:
                from .stream_refiner import Engine

            stream_engine = Engine(model, tokenizer)
            stream_engine.extend(stream_prefix)
        # Draft sizes per backend: the native engine verifies up to 8 rows for the price of 2
        # (real-time replay, 40 held-out dictations, 2 runs each: >= 5 s p90 118-126 ms with
        # the MLX ladder, 99.5-100.7 ms with PE_LADDER).
        stream = StreamSession(
            stream_engine,
            EnvelopePrompter(),
            ladder=PE_LADDER if engine is not None else None,
        )

    def stream_reset(lost: bool = False) -> None:
        """Forget the dictation's draft; keep the prefix state. `lost`: the state may no
        longer match the bookkeeping (an error mid-request), so drop it."""
        if stream is None:
            return
        stream.reset()
        if lost and engine is not None:
            stream.e.invalidate()  # the next forward opens a fresh session
        elif lost:
            stream.e.rollback(0)
            stream.e.extend(stream_prefix)
        else:
            stream.e.rollback(len(stream_prefix))

    def stream_commit(
        text: str, should_abort, mode: tuple = (None, ""), wanted: dict | None = None
    ) -> dict:
        started_at = time.perf_counter()
        aborted = False
        info: dict = {}
        stream.p.mode_line = mode[1]
        try:
            info = stream.commit(text, should_abort=should_abort)
        except Aborted:
            aborted = True
            # Keep only the part of the state that matches this commit's prompt prefix.
            stream.e.rollback(
                stream.e.common_prefix(stream.p.open_prompt_tokens(stream._tail(text)))
            )
        reply = {
            "commits": stream.commits,
            "aborted": aborted,
            "milliseconds": (time.perf_counter() - started_at) * 1000,
            "draftTokenCount": len(stream.draft or []),
            "forwardPasses": info.get("passes", []),
        }
        # Live text: an aborted pass left the previous draft, which the engine already has.
        if not aborted:
            draft_spoken_words = stream.head + stream.p.words(stream.draft_text or text)
            draft_spoken = (
                " ".join(draft_spoken_words)
                if stream.draft_text is not None
                else transcript_payload(text)
            )
            reply.update(
                draft_fields(
                    wanted,
                    stream.draft,
                    output_budget(stream.draft_text or text),
                    tokenizer.decode,
                    draft_spoken,
                    head=stream.head_out,
                    vocab=refiner_vocabulary(text),
                )
            )
        return {"ok": True, "stream": reply}

    def run_streamed(text: str, mode: tuple = (None, "")) -> dict:
        max_tokens = output_budget(text)
        started_at = time.perf_counter()
        commits, draft_tokens = stream.commits, len(stream.draft or [])
        cached = len(stream.e.tokens)
        # A draft made under another mode line still drafts; the answer is this prompt's.
        stream.p.mode_line = mode[1]
        try:
            ids, info = stream.final(text)
        except EngineError as error:
            # e.g. the session's context cap: the one-shot request continues past it
            print(f"stream: {error}; one-shot refine", file=sys.stderr)
            head_out = list(stream.head_out)
            tail_text = stream._tail(text) if head_out else text
            stream_reset(lost=True)
            res = run(tail_text, streamed=False, mode=mode)
            if head_out and res.get("ok"):
                head_str = decode_pieces(tokenizer.decode, head_out)
                combined = " ".join(
                    t for t in (head_str, res["outputText"]) if t
                )
                output, guard_fired = guard_output(transcript_payload(text), combined)
                output = restore_terms(
                    transcript_payload(text), output, refiner_vocabulary(text)
                )
                res["outputText"] = output
                res["diagnostics"]["inputCharacterCount"] = len(text)
                res["diagnostics"]["outputCharacterCount"] = len(output)
                res["diagnostics"]["guardFired"] = guard_fired
            return res
        latency_ms = (time.perf_counter() - started_at) * 1000
        prompt_tokens = len(stream.p.prompt_tokens(text))
        pieces = stream.pieces(ids)
        stream_reset()
        output = decode_pieces(tokenizer.decode, pieces)
        output, guard_fired = guard_output(transcript_payload(text), output)
        output = restore_terms(
            transcript_payload(text), output, refiner_vocabulary(text)
        )
        return {
            "ok": True,
            "outputText": output,
            "diagnostics": {
                "latencyMilliseconds": latency_ms,
                "timeToFirstTokenMilliseconds": latency_ms,
                "tokensPerSecond": (
                    len(ids) / (latency_ms / 1000) if latency_ms > 0 else 0.0
                ),
                "generatedTokenCount": len(ids),
                "inputCharacterCount": len(text),
                "outputCharacterCount": len(output),
                "maxOutputTokenBudget": max_tokens,
                "promptTokenCount": prompt_tokens,
                "cachedPromptTokenCount": min(cached, prompt_tokens),
                "profileTokenCount": profile_report["blockTokens"],
                "refiner": refiner,
                "writingMode": mode[0],
                "guardFired": guard_fired,
                "engine": "native session" if engine is not None else "mlx",
                "streamCommits": commits,
                "streamDraftTokenCount": draft_tokens,
                "releaseForwardPasses": info.get("passes", []),
                # tokens of background work thrown away because the transcript was revised
                "streamWaste": info.get("waste"),
                # parts of a long dictation refined before the last one (0: one request)
                "streamFrozenParts": len(pieces) - 1,
            },
        }

    def run(text: str, streamed: bool = True, mode: tuple = (None, "")) -> dict:
        if (
            streamed
            and stream is not None
            and (
                stream.commits
                or len(stream.p.words(text)) > stream.FREEZE_WORDS
                or not stream.p.fits(text)
            )
        ):
            return run_streamed(text, mode)
        if engine is not None:
            return run_engine(text, mode)
        prompt = build_prompt(text, mode[1])
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
        guard_fired = False
        if refiner:
            output, guard_fired = guard_output(transcript_payload(text), output)
            output = restore_terms(
                transcript_payload(text), output, refiner_vocabulary(text)
            )
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
                "refiner": refiner,
                "writingMode": mode[0],
                "guardFired": guard_fired,
            },
        }

    ready_message = f"{args.model} loaded in {load_seconds:.1f}s"
    if refiner:
        ready_message += "; refiner prompt and word-count guard"
    if mode_lines:
        ready_message += "; writing modes " + ", ".join(sorted(mode_lines))
    if engine is not None:
        ready_message += "; native refiner engine"
        if engine_config.get("draft"):
            ready_message += (
                f" with {engine_config.get('draft_kind', 'dflash2n')} drafter"
            )
        if engine_prefix_tokens:
            ready_message += f"; prefix cache {engine_prefix_tokens} tokens"
    elif engine_fallback is not None:
        ready_message += f"; MLX refiner (native engine unavailable: {engine_fallback})"
    if profile_block:
        ready_message += f"; {describe_profile(profile_report)}"
    if prefix_cache is not None:
        ready_message += f"; prefix cache {len(prefix_tokens)} tokens"
    if stream is not None:
        ready_message += "; streaming refine"

    def handle(request) -> bool:
        """One command; False ends the loop."""
        if isinstance(request, str):
            emit({"ok": False, "error": f"bad json: {request[:80]}"})
            return True
        command = request.get("command")
        if command == "shutdown":
            return False
        if command == "status":
            emit({"ok": True, "status": {"state": "ready", "message": ready_message}})
            return True
        if command == "warmup":
            try:
                run("warmup pass", streamed=False)
            except Exception as error:  # noqa: BLE001
                emit({"ok": False, "error": f"warmup failed: {error}"})
                traceback.print_exc(file=sys.stderr)
                return True
            emit({"ok": True, "status": {"state": "ready", "message": ready_message}})
            return True
        if command == "prewarm":
            # Dictation started. After a minute or two idle the native engine
            # answers its next request ~3x slower (284 vs 96 ms after 120 s,
            # M4 Max); a short request now, while the user is still talking,
            # takes that cost instead (the real one then ran in 107-119 ms).
            if engine is not None:
                try:
                    run("warmup pass", streamed=False)
                except Exception as error:  # noqa: BLE001
                    print(f"prewarm failed: {error}", file=sys.stderr)
            emit({"ok": True, "status": {"state": "warm"}})
            return True
        if command == "stream_reset":
            stream_reset()
            return True
        if command == "run":
            text = request.get("inputText") or ""
            if not text.strip():
                stream_reset()
                emit({"ok": False, "error": "empty inputText"})
                return True
            mode = writing_mode(mode_lines, mode_table, request.get("context"))
            try:
                emit(run(text, mode=mode))
            except Exception as error:  # noqa: BLE001
                stream_reset(lost=True)
                emit({"ok": False, "error": f"correction failed: {error}"})
                traceback.print_exc(file=sys.stderr)
            return True
        emit({"ok": False, "error": f"unknown command: {command}"})
        return True

    try:  # run as sidecar/polish_server.py (sidecar dir on sys.path) or imported as a package
        from linereader import LineReader
    except ImportError:
        from .linereader import LineReader

    # The protocol channel (isolate_protocol moved it off fd 0), read unbuffered so a
    # running background commit can see a newer command between forward passes.
    reader = LineReader((PROTOCOL_IN or sys.stdin).fileno())
    queue: list = []
    pending_commit = None

    def is_commit(request) -> bool:
        return isinstance(request, dict) and request.get("command") == "stream_commit"

    def take(messages) -> None:
        """Queue messages; commits are coalesced to the newest, and any other command
        (a run above all) supersedes commits that have not started."""
        nonlocal pending_commit
        for message in messages:
            if is_commit(message):
                pending_commit = message if stream is not None else None
            else:
                if isinstance(message, dict) and message.get("command") in (
                    "run",
                    "stream_reset",
                    "prewarm",
                ):
                    pending_commit = None
                queue.append(message)

    while True:
        if not queue:
            take(reader.poll(None if pending_commit is None else 0.0))
            if reader.eof and not queue and pending_commit is None:
                break
        if queue:
            if not handle(queue.pop(0)):
                break
            continue
        if pending_commit is None:
            continue
        request, pending_commit = pending_commit, None
        text = request.get("inputText") or ""
        if not text.strip():
            continue

        def should_abort() -> bool:
            take(reader.poll(0.0))
            return bool(queue) or pending_commit is not None

        context = request.get("context")
        mode = writing_mode(mode_lines, mode_table, context)
        try:
            emit(stream_commit(text, should_abort, mode, draft_request(context)))
        except Exception as error:  # noqa: BLE001
            # A failed background commit must not surface as a correction error.
            stream_reset(lost=True)
            emit({"ok": True, "stream": {"error": str(error)}})
            traceback.print_exc(file=sys.stderr)


if __name__ == "__main__":
    main()
