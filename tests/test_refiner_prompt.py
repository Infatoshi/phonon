"""The sidecar's refiner prompt against the training-side builders' own output
(fixtures/refiner_prompt_cases.json, generated from synthetic inputs)."""

import json
from pathlib import Path

import pytest

from sidecar.polish_server import (
    REFINER_SYSTEM_PROMPT,
    REFINER_VOCAB_PREFIX,
    RequestLines,
    read_context_lines,
    refiner_system_prompt,
    request_lines,
)
from sidecar.refiner_prompt import (
    BEFORE_MAX_WORDS,
    INSTRUCTION,
    MODE_LINES,
    SCREEN_MAX_WORDS,
    VOCAB_PREFIX,
    join_rule,
    system_prompt,
    user_turn,
)

FIXTURES = json.loads(
    (Path(__file__).resolve().parents[1] / "fixtures/refiner_prompt_cases.json").read_text()
)
CASES = FIXTURES["cases"]


def test_the_fixed_strings_are_the_training_ones():
    assert INSTRUCTION == FIXTURES["instruction"] == REFINER_SYSTEM_PROMPT
    assert MODE_LINES == FIXTURES["modes"]
    assert VOCAB_PREFIX == FIXTURES["vocabulary_prefix"] == REFINER_VOCAB_PREFIX
    assert SCREEN_MAX_WORDS == FIXTURES["screen_max_words"]
    assert BEFORE_MAX_WORDS == FIXTURES["before_max_words"]


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_prompt_is_byte_identical_to_the_training_builder(case):
    system = system_prompt(
        case["vocabulary"],
        MODE_LINES[case["mode"]],
        case["screen_text"],
        case["before_cursor"],
    )
    assert system == case["system"]
    assert user_turn(case["transcript"], case["before_cursor"]) == case["user"]


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_the_sidecar_request_path_builds_the_same_prompt(case, tmp_path):
    (tmp_path / "phonon_refiner.json").write_text(
        json.dumps(
            {"writing_modes": MODE_LINES, "context_lines": ["on_screen", "before_cursor"]}
        )
    )
    context = {"writing_mode": case["mode"]}
    if case["screen_text"] is not None:
        context["screen_text"] = case["screen_text"]
    if case["before_cursor"] is not None:
        context["textbox"] = {"before_text": case["before_cursor"], "after_text": "x"}
    envelope = f"<transcript>\n{case['transcript']}\n</transcript>"
    lines = request_lines(
        MODE_LINES,
        {"other": "minimal"},
        context,
        read_context_lines(str(tmp_path)),
        envelope,
    )
    assert lines.mode == case["mode"]
    system = refiner_system_prompt(case["vocabulary"], lines.line, lines.screen, lines.before)
    assert system == case["system"]
    assert user_turn(lines.spoken, lines.before) == case["user"]


@pytest.mark.parametrize("before, raw, want", FIXTURES["join_rule_cases"])
def test_join_rule_cases(before, raw, want):
    assert join_rule(before, raw) == want


def test_join_rule_has_all_31_training_cases():
    assert len(FIXTURES["join_rule_cases"]) == 31


def test_a_refiner_without_context_lines_never_gets_them(tmp_path):
    (tmp_path / "phonon_refiner.json").write_text(json.dumps({"writing_modes": MODE_LINES}))
    assert read_context_lines(str(tmp_path)) == frozenset()
    context = {
        "app_category": "chat",
        "screen_text": "Inbox Drafts",
        "textbox": {"before_text": "and then"},
    }
    lines = request_lines(MODE_LINES, {"chat": "casual"}, context, frozenset(), "Hello")
    assert (lines.mode, lines.line, lines.screen, lines.before) == (
        "casual",
        MODE_LINES["casual"],
        "",
        "",
    )
    assert user_turn("Hello", lines.before) == "Hello"


def test_only_the_part_that_starts_the_dictation_gets_the_before_cursor_line():
    lines = RequestLines("minimal", MODE_LINES["minimal"], "", "and then", "One two. Three four.")
    assert lines.for_part("One two.").before == "and then"
    assert lines.for_part("One two. Three four.").before == "and then"
    assert lines.for_part("Three four.").before == ""
