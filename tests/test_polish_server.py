import json
import sys
import time

import pytest

from sidecar.polish_server import (
    PROFILE_LEAD_IN,
    REFINER_ENGINE_FILE,
    REFINER_MARKER_FILE,
    REFINER_SYSTEM_PROMPT,
    RefinerEngine,
    build_profile_block,
    describe_profile,
    draft_fields,
    draft_request,
    engine_command,
    engine_only_files,
    engine_probe_passed,
    guard_output,
    has_mlx_weights,
    read_engine_config,
    read_writing_modes,
    refiner_system_prompt,
    refiner_token_budget,
    refiner_vocabulary,
    resolve_model_dir,
    restore_terms,
    snapshot_is_complete,
    split_rendered_prompt,
    with_transcript,
    writing_mode,
)


def word_tokens(text: str) -> int:
    return len(text.split())


USER_MD = "# user.md\n\n- Elliot Arledge, handle infatoshi.\n- Machines: anvil, gamer."
VOCAB_MD = (
    "# vocab.md\n\n## project\n- Phonon [phone on]\n- KernelBench [colonel bench]\n"
    "\n## product\n- Claude Code [clawed code]\n- Codex [codecs]"
)


def test_no_profile_files_means_no_block():
    block, report = build_profile_block(None, None, 1500, word_tokens)
    assert block == ""
    assert report["blockTokens"] == 0


def test_profile_block_keeps_both_files_when_they_fit():
    block, report = build_profile_block(USER_MD, VOCAB_MD, 1500, word_tokens)
    assert block.startswith(PROFILE_LEAD_IN)
    assert "<speaker_profile>\n" + USER_MD + "\n</speaker_profile>" in block
    assert "<speaker_vocabulary>\n" + VOCAB_MD + "\n</speaker_vocabulary>" in block
    assert report["vocabLinesKept"] == report["vocabLinesTotal"]
    assert report["blockTokens"] == word_tokens(block)
    assert "cut" not in describe_profile(report)


def test_profile_block_is_byte_stable():
    first, _ = build_profile_block(USER_MD, VOCAB_MD, 1500, word_tokens)
    second, _ = build_profile_block(USER_MD, VOCAB_MD, 1500, word_tokens)
    assert first == second


def test_vocab_is_cut_by_whole_lines_within_budget():
    lead = word_tokens(PROFILE_LEAD_IN)
    user_cost = word_tokens(f"<speaker_profile>\n{USER_MD}\n</speaker_profile>")
    # Room for the user file plus roughly the first vocabulary section only.
    budget = lead + user_cost + 12
    block, report = build_profile_block(USER_MD, VOCAB_MD, budget, word_tokens)
    assert word_tokens(block) <= budget
    assert "<speaker_profile>\n" + USER_MD in block
    assert "Phonon [phone on]" in block
    assert "Codex [codecs]" not in block
    assert report["vocabLinesKept"] < report["vocabLinesTotal"]
    # A cut never strands a heading at the end of the vocabulary.
    body = block.split("<speaker_vocabulary>\n")[1].split("\n</speaker_vocabulary>")[0]
    assert not body.splitlines()[-1].startswith("#")
    assert "vocab.md cut to" in describe_profile(report)


def test_user_file_is_cut_before_vocab_when_it_alone_exceeds_budget():
    lead = word_tokens(PROFILE_LEAD_IN)
    user_cost = word_tokens(f"<speaker_profile>\n{USER_MD}\n</speaker_profile>")
    budget = lead + user_cost - 2
    block, report = build_profile_block(USER_MD, VOCAB_MD, budget, word_tokens)
    assert word_tokens(block) <= budget
    assert "<speaker_profile>" in block
    assert report["userLinesKept"] < report["userLinesTotal"]
    assert report["vocabLinesKept"] == 0


def test_only_one_file_present_renders_only_that_file():
    block, _ = build_profile_block(None, VOCAB_MD, 1500, word_tokens)
    assert "<speaker_profile>" not in block
    assert "<speaker_vocabulary>" in block


def test_split_rendered_prompt_returns_stable_prefix_and_suffix():
    def render(system: str, user: str) -> str:
        return f"<bos><|turn>system\n{system}<turn|>\n<|turn>user\n{user}<turn|>\n<|turn>model\n"

    prefix, suffix = split_rendered_prompt(render, "Fix the text.")
    assert prefix == "<bos><|turn>system\nFix the text.<turn|>\n<|turn>user\n"
    assert suffix == "<turn|>\n<|turn>model\n"
    assert prefix + "hello" + suffix == render("Fix the text.", "hello")


ENVELOPE = (
    "<phonon_dictionary>\ncanonical_terms: cuBLAS, Triton\nlikely_terms: cuBLAS, vLLM, Triton\n"
    "screen_confirmed_terms: vLLM\nphonetic_matches: pie torch => PyTorch\n"
    "exact_replacements: phonon => Phonon\n</phonon_dictionary>\n"
    "<transcript>\nmove the PyTorch loop to triton and check coo blast\n</transcript>"
)


def test_refiner_vocabulary_orders_by_confidence_and_dedupes():
    assert refiner_vocabulary(ENVELOPE) == [
        "vLLM",
        "PyTorch",
        "Phonon",
        "cuBLAS",
        "Triton",
    ]


def test_refiner_vocabulary_is_empty_without_an_envelope():
    assert refiner_vocabulary("just a bare transcript") == []


def test_refiner_vocabulary_caps_at_the_adaptive_budget():
    terms = ", ".join(f"term{i}" for i in range(200))
    text = f"<phonon_dictionary>\nlikely_terms: {terms}\n</phonon_dictionary>\n<transcript>\nhi\n</transcript>"
    assert len(refiner_vocabulary(text)) == 30
    long_text = text.replace("hi", " ".join(["word"] * 20))
    assert len(refiner_vocabulary(long_text)) == 60


def test_refiner_system_prompt_matches_the_training_format():
    assert refiner_system_prompt([]) == REFINER_SYSTEM_PROMPT
    assert refiner_system_prompt(["vLLM", "cuBLAS"]) == (
        REFINER_SYSTEM_PROMPT + "\nVocabulary the speaker may use: vLLM, cuBLAS"
    )
    # intent_v0 mix_build.py: refiner prompt, mode line, then the vocabulary line
    line = "Writing mode: casual (a chat message)."
    assert refiner_system_prompt([], line) == REFINER_SYSTEM_PROMPT + "\n" + line
    assert refiner_system_prompt(["vLLM"], line) == (
        REFINER_SYSTEM_PROMPT + "\n" + line + "\nVocabulary the speaker may use: vLLM"
    )


MODE_LINES = {
    "minimal": "Writing mode: minimal (notes and documents).",
    "casual": "Writing mode: casual (a chat message).",
    "formal": "Writing mode: formal (an email or document).",
    "prompt": "Writing mode: prompt (an instruction to an AI assistant).",
}


def test_a_refiner_without_modes_gets_no_mode_line(tmp_path):
    assert read_writing_modes(str(tmp_path)) == ({}, {})
    (tmp_path / REFINER_MARKER_FILE).write_text(json.dumps({"name": "legacy"}))
    lines, table = read_writing_modes(str(tmp_path))
    assert (lines, table) == ({}, {})
    assert writing_mode(lines, table, {"app_category": "chat"}) == (None, "")
    (tmp_path / REFINER_MARKER_FILE).write_text("{not json")
    assert read_writing_modes(str(tmp_path)) == ({}, {})


def test_writing_mode_follows_the_app_category(tmp_path):
    (tmp_path / REFINER_MARKER_FILE).write_text(
        json.dumps({"writing_modes": MODE_LINES})
    )
    lines, table = read_writing_modes(str(tmp_path))
    pick = lambda context: writing_mode(lines, table, context)[0]
    assert pick({"app_category": "chat"}) == "casual"
    assert pick({"app_category": "work_chat"}) == "casual"
    assert pick({"app_category": "email"}) == "formal"
    assert pick({"app_category": "ai_prompt"}) == "prompt"
    assert pick({"app_category": "code"}) == "prompt"
    assert pick({"app_category": "docs"}) == "minimal"
    assert pick({"app_category": "other"}) == "minimal"
    # no app, an unknown category, or no context: the "other" mode
    assert pick({}) == pick({"app_category": "game"}) == pick(None) == "minimal"
    assert (
        writing_mode(lines, table, {"app_category": "chat"})[1] == MODE_LINES["casual"]
    )


def test_a_picked_writing_style_overrides_the_app(tmp_path):
    (tmp_path / REFINER_MARKER_FILE).write_text(
        json.dumps({"writing_modes": MODE_LINES})
    )
    lines, table = read_writing_modes(str(tmp_path))
    pick = lambda context: writing_mode(lines, table, context)
    assert pick({"app_category": "chat", "writing_mode": "formal"}) == (
        "formal",
        MODE_LINES["formal"],
    )
    assert pick({"writing_mode": "casual"})[0] == "casual"
    # a mode the refiner does not have, or junk: the category decides
    assert pick({"app_category": "chat", "writing_mode": "shouty"})[0] == "casual"
    assert pick({"app_category": "email", "writing_mode": 3})[0] == "formal"
    # a refiner without modes still gets no mode line
    assert writing_mode({}, {}, {"writing_mode": "formal"}) == (None, "")


def test_the_marker_can_remap_categories(tmp_path):
    marker = {
        "writing_modes": {
            "minimal": MODE_LINES["minimal"],
            "prompt": MODE_LINES["prompt"],
        },
        "writing_modes_by_app_category": {"code": "minimal", "chat": "casual"},
    }
    (tmp_path / REFINER_MARKER_FILE).write_text(json.dumps(marker))
    lines, table = read_writing_modes(str(tmp_path))
    assert writing_mode(lines, table, {"app_category": "code"})[0] == "minimal"
    assert writing_mode(lines, table, {"app_category": "ai_prompt"})[0] == "prompt"
    # a mode the refiner has no line for gives no line
    assert writing_mode(lines, table, {"app_category": "chat"}) == (None, "")


def test_refiner_token_budget():
    assert refiner_token_budget(0) == 32
    assert refiner_token_budget(100) == 182


def test_guard_falls_back_on_runaway_and_collapse():
    spoken = "one two three four five six seven eight nine ten"
    assert guard_output(spoken, spoken + " " + spoken + " more") == (spoken, True)
    assert guard_output(spoken, "one two") == (spoken, True)
    assert guard_output(spoken, "one two three four five six seven eight") == (
        "one two three four five six seven eight",
        False,
    )
    assert guard_output("hi there", "") == ("", False)


def test_a_said_dictionary_term_survives_a_swap_for_another_term():
    # owner dictation 2026-11-04 3:25 pm: MTP and MLP in the dictionary
    vocab = ["MLP", "MTP", "MoE", "DSpark"]
    spoken = "then we would test native mtp versus d spark, the mlp or the moe"
    out = "Then we would test native MTK versus DSpark, the MMLU-Pro or the MoE."
    assert restore_terms(spoken, out, vocab) == (
        "Then we would test native MTP versus DSpark, the MLP or the MoE."
    )
    # a term that was said stays; a rewrite into ordinary words stays; no vocab, no change
    assert restore_terms("mtp or mtk", "MTP or MTK.", vocab) == "MTP or MTK."
    assert restore_terms("the mtp path", "the fast path", vocab) == "the fast path"
    assert restore_terms(spoken, out, []) == out
    # line breaks in the output are kept
    assert (
        restore_terms("hi sam the mtp", "Hi Sam,\nthe MTK", vocab) == "Hi Sam,\nthe MTP"
    )


def test_a_long_dictations_parts_share_the_envelope():
    text = "<phonon_dictionary>\nlikely_terms: MTP\n</phonon_dictionary>\n<transcript>\na b c\n</transcript>"
    part = with_transcript(text, ["b", "c"])
    assert part.startswith("<phonon_dictionary>\nlikely_terms: MTP")
    assert part.endswith("<transcript>\nb c\n</transcript>")
    assert with_transcript("a b c", ["c"]) == "c"


def test_live_text_is_sent_only_when_the_engine_asks():
    assert draft_request(None) is None
    assert draft_request({"app_category": "chat"}) is None  # the default path: no text
    assert draft_request({"draft_id": ""}) is None
    assert draft_request({"draft_id": 7}) is None
    assert draft_request({"draft_id": "p3", "draft_seq": True}) == {"draftId": "p3"}
    wanted = draft_request({"app_category": "chat", "draft_id": "p3", "draft_seq": 5})
    assert wanted == {"draftId": "p3", "draftSeq": 5}

    def decode(ids):
        return " ".join(f"w{i}" for i in ids) + "<turn|>"

    spoken = "w1 w2 w3"
    assert draft_fields(None, [1, 2, 3], 40, decode, spoken) == {}
    assert draft_fields(wanted, None, 40, decode, spoken) == {}
    # cleaned like a run's output; complete when it ended before the budget
    assert draft_fields(wanted, [1, 2, 3], 40, decode, spoken) == {
        "draftId": "p3",
        "draftSeq": 5,
        "draftText": "w1 w2 w3",
        "draftComplete": True,
    }
    # stopped at the output budget: shown, but marked incomplete
    assert draft_fields(wanted, [1, 2, 3], 3, decode, spoken)["draftComplete"] is False
    # the run's guard applies too: a collapsed draft shows the spoken text
    long_spoken = " ".join(f"w{i}" for i in range(1, 21))
    assert (
        draft_fields(wanted, [1], 40, decode, long_spoken)["draftText"] == long_spoken
    )


# Stands in for `pe serve`: same handshake and reply shapes, echoes the prompt.
FAKE_ENGINE = """
import json, sys
print(json.dumps({"ready": True, "init_s": 0.5}), flush=True)
for line in sys.stdin:
    request = json.loads(line)
    if "prefix" in request:
        print(json.dumps({"ok": True, "prefix_tokens": len(request["prefix"].split()) - 1}), flush=True)
        continue
    print(json.dumps({"ok": True, "text": request["prompt"].upper(), "prompt_tokens": 3,
                      "tokens": request["max_tokens"], "prefix_hit": True, "drafted": 7,
                      "accepted": 5, "ms": 1.0}), flush=True)
"""


def test_engine_config_is_read_only_when_present(tmp_path):
    assert read_engine_config(str(tmp_path)) is None
    config = {"binary": "engine/bin/pe", "target": "target.gguf"}
    (tmp_path / REFINER_ENGINE_FILE).write_text(json.dumps(config))
    assert read_engine_config(str(tmp_path)) == config


def test_refiner_engine_round_trip_and_exit_on_close():
    engine = RefinerEngine("", {}, command=[sys.executable, "-c", FAKE_ENGINE])
    assert engine.init_seconds == 0.5
    assert engine.set_prefix("a b c d") == 3
    reply = engine.generate("héllo", 12)
    assert reply["text"] == "HÉLLO"
    assert reply["tokens"] == 12
    engine.process.stdin.close()
    assert engine.process.wait(timeout=5) == 0


def test_a_hung_engine_is_killed_at_the_ready_timeout():
    # An engine that never says it is ready (a hung GPU, a stuck shader build) must
    # not hold up startup: it is killed and the caller falls back to MLX.
    hang = "import time\ntime.sleep(60)\n"
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="did not answer within 0.5 s"):
        RefinerEngine("", {}, command=[sys.executable, "-c", hang], ready_timeout=0.5)
    assert time.monotonic() - started < 10


def test_a_hung_probe_times_out_and_kills_the_engine():
    ready_then_hang = (
        "import json, sys, time\n"
        'print(json.dumps({"ready": True, "init_s": 0.1}), flush=True)\n'
        "sys.stdin.readline()\n"
        "time.sleep(60)\n"
    )
    engine = RefinerEngine("", {}, command=[sys.executable, "-c", ready_then_hang])
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="did not answer"):
        engine.generate("hello", 8, timeout=0.5)
    assert time.monotonic() - started < 10
    assert engine.process.poll() is not None


def test_an_engine_ready_within_the_timeout_starts():
    slow = "import time\ntime.sleep(0.3)\n" + FAKE_ENGINE
    engine = RefinerEngine(
        "", {}, command=[sys.executable, "-c", slow], ready_timeout=10
    )
    assert engine.init_seconds == 0.5
    engine.close()


def test_engine_command_prefers_the_directory_binary(tmp_path):
    own = tmp_path / "engine" / "bin" / "pe"
    own.parent.mkdir(parents=True)
    own.write_text("")
    bundled = tmp_path / "Helpers" / "pe"
    bundled.parent.mkdir()
    bundled.write_text("")
    config = {
        "binary": "engine/bin/pe",
        "target": "t.gguf",
        "draft": "d.gguf",
        "draft_kind": "dflash2n",
    }
    command, env = engine_command(str(tmp_path), config, str(bundled), "/shaders")
    assert command == [
        str(own),
        str(tmp_path / "t.gguf"),
        "serve",
        "--draft",
        "dflash2n",
        str(tmp_path / "d.gguf"),
    ]
    assert env is None


def test_engine_command_uses_the_bundled_engine_for_a_download(tmp_path):
    bundled = tmp_path / "pe"
    bundled.write_text("")
    config = {"target": "t.gguf", "draft": "d.gguf"}
    command, env = engine_command(str(tmp_path), config, str(bundled), "/shaders")
    assert command[0] == str(bundled)
    assert command[-3:] == ["--draft", "dflash2n", str(tmp_path / "d.gguf")]
    assert env["PE_DIR"] == "/shaders"
    assert engine_command(str(tmp_path), config) is None
    assert engine_command(str(tmp_path), config, str(tmp_path / "missing")) is None


def test_a_user_refiner_directory_is_used_as_it_is(tmp_path):
    # The owner's personal setup: a local refiner with its own engine config is
    # never swapped for the pinned download, and keeps its own engine binary.
    (tmp_path / REFINER_ENGINE_FILE).write_text(
        json.dumps({"binary": "engine/bin/pe", "target": "t.gguf"})
    )
    assert resolve_model_dir(str(tmp_path), "local") == str(tmp_path)
    own = tmp_path / "engine" / "bin" / "pe"
    own.parent.mkdir(parents=True)
    own.write_text("")
    command, _ = engine_command(
        str(tmp_path), read_engine_config(str(tmp_path)), "/bundled/pe", "/shaders"
    )
    assert command[0] == str(own)


ISOLATION_CHILD = """
import sys
sys.path.insert(0, ".")
from sidecar import polish_server
polish_server.isolate_protocol()
try:
    answer = input("Do you wish to run the custom code? [y/N] ")
except EOFError:
    answer = "<eof>"
print("library chatter")
polish_server.emit({"ok": True, "answer": answer})
polish_server.emit({"ok": True, "line": next(iter(polish_server.PROTOCOL_IN)).strip()})
"""


def test_libraries_cannot_touch_the_protocol_pipes():
    import subprocess

    result = subprocess.run(
        [sys.executable, "-c", ISOLATION_CHILD],
        check=False,
        input='{"command": "warmup"}\n',
        capture_output=True,
        text=True,
        timeout=30,
    )
    replies = [json.loads(line) for line in result.stdout.splitlines()]
    # The prompt read /dev/null, not the engine's warmup line, which is still
    # there for the protocol loop; the chatter went to stderr.
    assert replies == [
        {"ok": True, "answer": "<eof>"},
        {"ok": True, "line": '{"command": "warmup"}'},
    ]
    assert "library chatter" in result.stderr
    assert "custom code" in result.stderr


def test_engine_probe_needs_the_fixture_words():
    assert engine_probe_passed("Hello, Fluid Voice startup prime.")
    assert engine_probe_passed("hello FluentBit startup prime")
    assert not engine_probe_passed("")
    assert not engine_probe_passed("hello fluid")
    assert not engine_probe_passed("the quick brown fox jumps")
    assert not engine_probe_passed("hello fluid voice startup prime " + "and more " * 4)


def test_snapshot_completeness(tmp_path):
    assert not snapshot_is_complete(str(tmp_path))
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors").write_text("")
    assert snapshot_is_complete(str(tmp_path))
    engine = {"target": "t.gguf", "draft": "d.gguf"}
    (tmp_path / REFINER_ENGINE_FILE).write_text(json.dumps(engine))
    assert not snapshot_is_complete(str(tmp_path))
    (tmp_path / "t.gguf").write_text("")
    (tmp_path / "d.gguf").write_text("")
    assert snapshot_is_complete(str(tmp_path))
    index = {"weight_map": {"a": "model-00001.safetensors"}}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    assert not has_mlx_weights(str(tmp_path))
    (tmp_path / "model-00001.safetensors").write_text("")
    assert has_mlx_weights(str(tmp_path))


def test_mlx_cache_is_capped_before_the_model_loads():
    """Issue #27: without a cap MLX kept ~1 GB of freed buffers per dictation."""
    import inspect

    from sidecar import polish_server

    assert polish_server.POLISH_MLX_CACHE_LIMIT == 1 << 30
    source = inspect.getsource(polish_server.main)
    cap = source.index("mx.set_cache_limit(POLISH_MLX_CACHE_LIMIT)")
    assert cap < source.index("resolve_model_dir(")


def test_stopped_at_cap_tells_a_cut_from_an_end_token():
    from sidecar.polish_server import ENGINE_CONTEXT_TOKENS, stopped_at_cap

    assert not stopped_at_cap(112, 160)  # ended on its own (the end token is dropped)
    assert stopped_at_cap(160, 160)  # used the whole output budget
    # the native engine's context: prompt plus output reached 2 * CTX (seen live: 2317 + 1779)
    assert stopped_at_cap(1779, 2048, 2317, ENGINE_CONTEXT_TOKENS)
    assert not stopped_at_cap(1779, 2048, 2316, ENGINE_CONTEXT_TOKENS)
    assert not stopped_at_cap(1779, 2048, 2317)  # MLX: no fixed context


def test_split_point_prefers_the_sentence_end_nearest_the_middle():
    from sidecar.polish_server import split_point

    words = [
        "one",
        "two.",
        "three",
        "four",
        "five",
        "six.",
        "seven",
        "eight",
        "nine",
        "ten",
    ]
    assert split_point(words) == 6  # after "six." (the middle is 5; "two." is at 2)
    assert (
        split_point(["a", "b", "c", "d", "e", "f"]) == 3
    )  # no sentence end: the middle
    assert (
        split_point(["a", "b", "c", "d."]) == 2
    )  # the final word's period is not a split


def test_engine_only_files_skips_mlx_safetensors_for_native_refiner():
    from sidecar.model_pins import load_pins

    pins = load_pins()
    refiner = "Infatoshi/phonon-refiner-1.7b"
    rev = pins[refiner]["revision"]
    only = engine_only_files(refiner, rev, pins)
    assert only is not None
    assert "model.safetensors" not in only
    assert "model.safetensors.index.json" not in only
    assert "phonon-refiner-1.7b-Q4_K_S-e4.gguf" in only
    assert "phonon-refiner-1.7b-dflash2-drafter.gguf" in only
    assert "tokenizer.json" in only
    assert sum(pins[refiner]["files"][f]["size"] for f in only) == 1_221_230_710

    gemma = "mlx-community/gemma-4-e2b-it-4bit"
    assert engine_only_files(gemma, pins[gemma]["revision"], pins) is None

    with pytest.raises(ValueError):
        engine_only_files(refiner, "main", pins)


def test_snapshot_completeness_for_engine_allowlist(tmp_path):
    engine = {"target": "t.gguf", "draft": "d.gguf"}
    (tmp_path / REFINER_ENGINE_FILE).write_text(json.dumps(engine))
    (tmp_path / "t.gguf").write_text("")
    (tmp_path / "d.gguf").write_text("")
    (tmp_path / "tokenizer.json").write_text("{}")
    only = [REFINER_ENGINE_FILE, "t.gguf", "d.gguf", "tokenizer.json"]
    assert snapshot_is_complete(str(tmp_path), only=only)
    assert not snapshot_is_complete(str(tmp_path))  # MLX mode still requires safetensors
    assert not snapshot_is_complete(str(tmp_path), only=[*only, "config.json"])


def _fake_refiner_repo(tmp_path):
    import hashlib

    repo = "Org/refiner"
    rev = "c" * 40
    files = {
        REFINER_ENGINE_FILE: json.dumps({"target": "target.gguf", "draft": "draft.gguf"}).encode(),
        REFINER_MARKER_FILE: json.dumps({"writing_modes": {"minimal": "m"}}).encode(),
        "config.json": b'{"model_type": "lfm2"}',
        "tokenizer.json": b"{}",
        "tokenizer_config.json": b"{}",
        "chat_template.jinja": b"{{ messages }}",
        "target.gguf": b"gguf-target",
        "draft.gguf": b"gguf-draft",
        "model.safetensors": b"mlx-weights" * 50,
        "model.safetensors.index.json": json.dumps(
            {"weight_map": {"w": "model.safetensors"}}
        ).encode(),
    }
    pins = {
        repo: {
            "revision": rev,
            "files": {
                name: {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                for name, data in files.items()
            },
        }
    }
    hub = tmp_path / "hub"
    repo_dir = hub / f"models--{repo.replace('/', '--')}"
    snap = repo_dir / "snapshots" / rev
    blobs = repo_dir / "blobs"
    blobs.mkdir(parents=True)
    return repo, rev, files, pins, hub, snap, blobs


def test_engine_only_download_offline_reuse_and_lazy_mlx_fallback(tmp_path, monkeypatch):
    import hashlib
    import os
    import types

    from sidecar import model_pins, polish_server

    repo, rev, files, pins, hub, snap, blobs = _fake_refiner_repo(tmp_path)
    marker_dir = tmp_path / "markers"
    monkeypatch.setattr(model_pins, "load_pins", lambda: pins)
    monkeypatch.setattr(model_pins, "default_marker_dir", lambda: str(marker_dir))

    downloads = []
    local_checks = []
    emitted = []
    monkeypatch.setattr(polish_server, "emit", emitted.append)

    def materialize(names):
        snap.mkdir(parents=True, exist_ok=True)
        for name in names:
            data = files[name]
            digest = hashlib.sha256(data).hexdigest()
            blob = blobs / digest
            blob.write_bytes(data)
            dst = snap / name
            if not os.path.lexists(dst):
                os.symlink(os.path.relpath(blob, snap), dst)

    def fake_snapshot_download(model, revision=None, local_files_only=False, allow_patterns=None):
        assert model == repo and revision == rev
        if local_files_only:
            local_checks.append(sorted(allow_patterns) if allow_patterns else None)
            if not snap.is_dir():
                raise FileNotFoundError("not cached")
            return str(snap)
        chosen = list(allow_patterns) if allow_patterns is not None else sorted(files)
        downloads.append(sorted(chosen))
        materialize(chosen)
        return str(snap)

    class FakeHfApi:
        def model_info(self, model, revision=None, files_metadata=True):
            return types.SimpleNamespace(
                siblings=[
                    types.SimpleNamespace(rfilename=name, size=len(data))
                    for name, data in files.items()
                ]
            )

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(
            snapshot_download=fake_snapshot_download,
            HfApi=FakeHfApi,
            constants=types.SimpleNamespace(HF_HUB_CACHE=str(hub), HF_HUB_DISABLE_XET=False),
        ),
    )

    hashed = []
    real_sha = model_pins.sha256_of
    monkeypatch.setattr(
        model_pins,
        "sha256_of",
        lambda p: hashed.append(os.path.basename(p)) or real_sha(p),
    )

    # 1. First launch with --engine-binary: fetches only engine + tokenizer files (never model.safetensors)
    out = resolve_model_dir(repo, rev, engine_only=True)
    assert out == str(snap)
    assert len(downloads) == 1
    assert "model.safetensors" not in downloads[0]
    assert "model.safetensors.index.json" not in downloads[0]
    assert not (snap / "model.safetensors").exists()
    assert not has_mlx_weights(str(snap))
    expected_engine_count = len(files) - 2
    assert len(hashed) == expected_engine_count

    # 2. Offline second launch: cache + marker hit with 0 downloads and 0 re-hashing
    hashed.clear()
    out2 = resolve_model_dir(repo, rev, engine_only=True)
    assert out2 == str(snap)
    assert len(downloads) == 1
    assert hashed == []

    # 3. Lazy MLX fallback: fetches and sha256-verifies the remaining MLX weights, keeping engine marker hits
    out3 = resolve_model_dir(repo, rev, engine_only=False)
    assert out3 == str(snap)
    assert len(downloads) == 2
    assert has_mlx_weights(str(snap))
    assert sorted(hashed) == ["model.safetensors", "model.safetensors.index.json"]
    marker = json.loads((marker_dir / "Org--refiner.json").read_text())
    assert set(marker["files"]) == set(files)


def test_main_engine_only_and_lazy_mlx_fallback(tmp_path, monkeypatch):
    import os
    import types

    from sidecar import polish_server

    def pipe_in(payload: bytes):
        r, w = os.pipe()
        os.write(w, payload)
        os.close(w)
        return os.fdopen(r, "r", encoding="utf-8")

    repo, rev, _files, _pins, _hub, snap, _blobs = _fake_refiner_repo(tmp_path)
    snap.mkdir(parents=True)
    (snap / REFINER_ENGINE_FILE).write_text(json.dumps({"target": "target.gguf"}))
    (snap / REFINER_MARKER_FILE).write_text(json.dumps({"writing_modes": {"minimal": "m"}}))
    (snap / "config.json").write_text("{}")
    (snap / "target.gguf").write_text("gguf")
    pe = tmp_path / "pe"
    pe.write_text("")

    calls = []

    def fake_resolve(model, revision, *, engine_only=False):
        calls.append(engine_only)
        if not engine_only:
            (snap / "model.safetensors").write_text("mlx")
        return str(snap)

    monkeypatch.setattr(polish_server, "resolve_model_dir", fake_resolve)
    monkeypatch.setattr(polish_server, "isolate_protocol", lambda: None)
    p1 = pipe_in(b'{"command":"status"}\n{"command":"shutdown"}\n')
    monkeypatch.setattr(polish_server, "PROTOCOL_IN", p1)
    emitted = []
    monkeypatch.setattr(polish_server, "emit", emitted.append)
    monkeypatch.setattr(polish_server, "register_vendored_models", lambda: None)

    class FailingEngine:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("metal unavailable")

    monkeypatch.setattr(polish_server, "RefinerEngine", FailingEngine)

    class FakeArray:
        def __getitem__(self, _idx):
            return self

    class FakeModel:
        def __call__(self, *_a, **_k):
            return None

    class FakeTokenizer:
        chat_template = ""
        bos_token = "<s>"
        eos_token_id = 0
        detokenizer = types.SimpleNamespace(reset=lambda: None)

        def encode(self, text, add_special_tokens=False):
            return [1, 2]

        def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False, **kw):
            return f"{messages[0]['content']}|PHONON_USER_TURN|{messages[1]['content']}"

    fake_mx = types.SimpleNamespace(
        set_cache_limit=lambda _n: None,
        array=lambda _x: FakeArray(),
        eval=lambda _x: None,
    )
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "mlx", types.SimpleNamespace(core=fake_mx))
    monkeypatch.setitem(sys.modules, "mlx.core", fake_mx)
    monkeypatch.setitem(
        sys.modules,
        "mlx_lm",
        types.SimpleNamespace(load=lambda _d: (FakeModel(), FakeTokenizer())),
    )
    monkeypatch.setitem(
        sys.modules,
        "mlx_lm.generate",
        types.SimpleNamespace(stream_generate=lambda *a, **k: iter([])),
    )
    monkeypatch.setitem(
        sys.modules,
        "mlx_lm.models",
        types.SimpleNamespace(cache=types.SimpleNamespace(make_prompt_cache=lambda _m: [])),
    )
    monkeypatch.setitem(
        sys.modules,
        "mlx_lm.models.cache",
        types.SimpleNamespace(make_prompt_cache=lambda _m: []),
    )
    monkeypatch.setitem(
        sys.modules,
        "mlx_lm.sample_utils",
        types.SimpleNamespace(make_sampler=lambda **_k: None),
    )
    monkeypatch.setitem(
        sys.modules,
        "mlx_lm.utils",
        types.SimpleNamespace(load_tokenizer=lambda _d: FakeTokenizer()),
    )

    monkeypatch.setattr(
        sys,
        "argv",
        ["polish_server.py", "--model", repo, "--revision", rev, "--engine-binary", str(pe)],
    )
    polish_server.main()
    p1.close()
    assert calls == [True, False]
    assert any(
        m.get("status", {}).get("message", "").endswith("downloading the MLX correction model")
        for m in emitted
    )
    assert any(m.get("status", {}).get("state") == "ready" for m in emitted)

    # Non-engine runs fetch everything up front (engine_only=False once)
    calls.clear()
    emitted.clear()
    p2 = pipe_in(b'{"command":"status"}\n{"command":"shutdown"}\n')
    monkeypatch.setattr(polish_server, "PROTOCOL_IN", p2)
    monkeypatch.setattr(sys, "argv", ["polish_server.py", "--model", repo, "--revision", rev])
    polish_server.main()
    p2.close()
    assert calls == [False]
    assert any(m.get("status", {}).get("state") == "ready" for m in emitted)
