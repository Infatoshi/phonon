import json
import sys

from sidecar.polish_server import (
    PROFILE_LEAD_IN,
    REFINER_ENGINE_FILE,
    REFINER_SYSTEM_PROMPT,
    RefinerEngine,
    build_profile_block,
    describe_profile,
    engine_command,
    engine_probe_passed,
    guard_output,
    has_mlx_weights,
    read_engine_config,
    refiner_system_prompt,
    refiner_token_budget,
    refiner_vocabulary,
    resolve_model_dir,
    snapshot_is_complete,
    split_rendered_prompt,
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
