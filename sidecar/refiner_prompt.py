"""The refiner's prompt, one definition for the Mac sidecar.

A trained refiner gets the exact prompt it was trained on (phonon-research, the ctx refiner
builders: base instruction, writing-mode line, then optional lines in this order):

    <instruction>
    Writing mode: ...                              (refiners trained with modes)
    Vocabulary the speaker may use: a, b, c        (when the dictionary has terms)
    On screen: <screen text>                       (refiners trained with context lines, when the app has screen text)
    Before cursor: <text before the cursor>        (same, when the focused field has text before the cursor)

as the system turn, and the bare transcript as the user turn. The context lines are shaped like
the training data: the screen text's first SCREEN_MAX_WORDS words and the last BEFORE_MAX_WORDS
words before the cursor, all whitespace (line breaks too) collapsed to single spaces; a line
whose text is empty is left out. With a Before-cursor line the transcript first goes through
the join rule (join_rule below); the model's output is never changed by it.

fixtures/refiner_prompt_cases.json holds prompts the training-side builders produced from
synthetic inputs; tests/test_refiner_prompt.py checks this module against them byte for byte,
and the Rust (crates/phonon-llm refiner_prompt.rs), Swift (ios/App/RefinerPrompt.swift) and
Kotlin (android RefinerPrompt.kt) definitions against the same file.
"""

from __future__ import annotations

import re

INSTRUCTION = (
    "Rewrite the raw dictation transcript into the exact text the speaker intended. "
    "Fix recognition errors and technical terms, keep the speaker's wording and casing style, "
    "do not add or drop content."
)
MODE_LINES = {
    "minimal": "Writing mode: minimal (notes and documents).",
    "casual": "Writing mode: casual (a chat message).",
    "formal": "Writing mode: formal (an email or document).",
    "prompt": "Writing mode: prompt (an instruction to an AI assistant).",
}
VOCAB_PREFIX = "Vocabulary the speaker may use: "
SCREEN_PREFIX = "On screen: "
BEFORE_PREFIX = "Before cursor: "
# Training screens were at most 350 words; the text before the cursor at most the last 40.
SCREEN_MAX_WORDS = 350
BEFORE_MAX_WORDS = 40

_CLOSERS = "'\")]"


def screen_text(text: str | None) -> str:
    """The On-screen line's text: the first SCREEN_MAX_WORDS words, whitespace collapsed."""
    return " ".join((text or "").split()[:SCREEN_MAX_WORDS])


def before_text(text: str | None) -> str:
    """The Before-cursor line's text: the last BEFORE_MAX_WORDS words, whitespace collapsed."""
    return " ".join((text or "").split()[-BEFORE_MAX_WORDS:])


def end_class(before: str | None) -> str:
    """How the text before the cursor ends: word / comma / sentence-end / colon / newline /
    other (none without text). Trailing spaces and tabs are ignored; a line break after them is
    a newline ending; then trailing closing quotes and brackets are ignored."""
    if before is None:
        return "none"
    b = before.rstrip(" \t")
    if b.endswith(("\n", "\r")):
        return "newline"
    b = b.rstrip().rstrip(_CLOSERS)
    if not b:
        return "other"
    c = b[-1]
    if c in ".?!":
        return "sentence-end"
    if c in ",;":
        return "comma"
    if c == ":":
        return "colon"
    if c.isalnum():
        return "word"
    return "other"


def join_rule(before: str | None, raw: str) -> str:
    """The transcript as it goes into the prompt after text that does not end a sentence (a
    word, `,` or `;`): its first letter lower-cased, unless the first word is "I" / I'm / I'll
    ..., or has another capital after its first letter (API, GitHub, OK). Anything else, and
    no Before-cursor text, leaves it unchanged."""
    if end_class(before) not in ("word", "comma") or not raw:
        return raw
    lead = len(raw) - len(raw.lstrip())
    if lead >= len(raw):
        return raw
    c = raw[lead]
    if not (c.isalpha() and c.isupper()):
        return raw
    core = re.sub(r"[^\w'’]+$", "", raw[lead:].split(None, 1)[0])
    if core == "I" or core.startswith(("I'", "I’")):
        return raw
    if any(ch.isupper() for ch in core[1:]):
        return raw
    return raw[:lead] + c.lower() + raw[lead + 1 :]


def system_prompt(
    vocab: list[str], mode_line: str = "", screen: str | None = None, before: str | None = None
) -> str:
    """The system turn. `screen` and `before` are the raw texts the app has (None or empty:
    no line); a refiner trained without context lines is never given them."""
    system = INSTRUCTION
    if mode_line:
        system += "\n" + mode_line
    if vocab:
        system += "\n" + VOCAB_PREFIX + ", ".join(vocab)
    shown = screen_text(screen)
    if shown:
        system += "\n" + SCREEN_PREFIX + shown
    typed = before_text(before)
    if typed:
        system += "\n" + BEFORE_PREFIX + typed
    return system


def user_turn(transcript: str, before: str | None = None) -> str:
    """The user turn: the transcript, through the join rule when there is a Before-cursor line."""
    raw = transcript.strip()
    return join_rule(before, raw) if before_text(before) else raw
