import json
import sys

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
    with_transcript,
    snapshot_is_complete,
    split_rendered_prompt,
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
    pick = lambda context: writing_mode(lines, table, context)[0]  # noqa: E731
    assert pick({"app_category": "chat"}) == "casual"
    assert pick({"app_category": "work_chat"}) == "formal"
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
    pick = lambda context: writing_mode(lines, table, context)  # noqa: E731
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
    # earlier (higher-confidence) dictionary terms win when two terms normalize to the same key
    assert (
        restore_terms("run vllm", "run VLM", ["vLLM", "VLLM"]) == "run vLLM"
    )
    assert (
        restore_terms("run vllm", "run VLM", ["VLLM", "vLLM"]) == "run VLLM"
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


def _run_polish_server_main(
    monkeypatch,
    tmp_path,
    capsys,
    *,
    use_engine=False,
    events=(),
    engine_handler=None,
    mlx_chunks=None,
    stream_engine_cls=None,
):
    import types
    from sidecar import polish_server

    (tmp_path / REFINER_MARKER_FILE).write_text(json.dumps({"name": "test-refiner"}))
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors").write_text("")
    if use_engine:
        (tmp_path / REFINER_ENGINE_FILE).write_text(
            json.dumps({"binary": "pe", "target": "t.gguf"})
        )
        (tmp_path / "pe").write_text("")
        (tmp_path / "t.gguf").write_text("")

    monkeypatch.setenv("PHONON_STREAM_REFINE", "1")
    monkeypatch.setattr(polish_server, "isolate_protocol", lambda: None)
    monkeypatch.setattr(
        polish_server, "PROTOCOL_IN", types.SimpleNamespace(fileno=lambda: 0)
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "polish_server.py",
            "--model",
            str(tmp_path),
            "--revision",
            "local",
            "--no-prefix-cache",
        ],
    )

    class FakeTokenizer:
        bos_token = "<bos>"
        eos_token_id = 99
        eos_token_ids = {99}
        chat_template = "fake"

        def __init__(self):
            self.detokenizer = types.SimpleNamespace(reset=lambda: None)

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            sys_msg = messages[0]["content"]
            usr_msg = messages[1]["content"]
            return f"<bos><sys>{sys_msg}</sys><usr>{usr_msg}</usr><mdl>"

        def encode(self, text, add_special_tokens=True):
            # Simple deterministic word/piece encoding
            return [abs(hash(w)) % 80 + 10 for w in text.split()]

        def decode(self, ids):
            return " ".join(f"tok{i}" for i in ids)

    fake_tok = FakeTokenizer()
    fake_model = lambda x, cache=None: None  # noqa: E731

    def fake_stream_generate(model, tokenizer, prompt, max_tokens=100, **kw):
        text_out = mlx_chunks(prompt, max_tokens) if mlx_chunks else "mlx one shot"
        yield types.SimpleNamespace(text=text_out, generation_tokens=3)

    fake_mlx_lm = types.ModuleType("mlx_lm")
    fake_mlx_lm.load = lambda d: (fake_model, fake_tok)
    fake_mlx_gen = types.ModuleType("mlx_lm.generate")
    fake_mlx_gen.stream_generate = fake_stream_generate
    fake_mlx_cache = types.ModuleType("mlx_lm.models.cache")
    fake_mlx_cache.make_prompt_cache = lambda m: []
    fake_mlx_sample = types.ModuleType("mlx_lm.sample_utils")
    fake_mlx_sample.make_sampler = lambda temp=0.0: None
    fake_mlx_utils = types.ModuleType("mlx_lm.utils")
    fake_mlx_utils.load_tokenizer = lambda d: fake_tok

    monkeypatch.setitem(sys.modules, "mlx_lm", fake_mlx_lm)
    monkeypatch.setitem(sys.modules, "mlx_lm.generate", fake_mlx_gen)
    monkeypatch.setitem(sys.modules, "mlx_lm.models.cache", fake_mlx_cache)
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", fake_mlx_sample)
    monkeypatch.setitem(sys.modules, "mlx_lm.utils", fake_mlx_utils)
    monkeypatch.setattr(polish_server, "register_vendored_models", lambda: None)

    if use_engine:
        class FakeRefinerEngine:
            init_seconds = 0.01

            def __init__(self, *a, **kw):
                self.calls = []

            def set_prefix(self, p):
                return 4

            def generate(self, prompt, max_tokens):
                if ENGINE_PROBE_TEXT in prompt:
                    return {
                        "ok": True,
                        "text": "Hello, Fluid Voice startup prime.",
                        "tokens": 5,
                        "prompt_tokens": 10,
                    }
                if engine_handler:
                    return engine_handler("generate", prompt, max_tokens)
                return {
                    "ok": True,
                    "text": "native one shot",
                    "tokens": 3,
                    "prompt_tokens": 10,
                }

            def call(self, req):
                if engine_handler:
                    return engine_handler("call", req, None)
                return {"ok": False, "error": "unhandled"}

            def close(self):
                pass

        from sidecar.polish_server import ENGINE_PROBE_TEXT
        monkeypatch.setattr(polish_server, "RefinerEngine", FakeRefinerEngine)

    if stream_engine_cls is not None:
        import sidecar.stream_refiner as sr
        monkeypatch.setattr(sr, "Engine", stream_engine_cls)

    event_list = list(events)

    class FakeLineReader:
        def __init__(self, _fd):
            self.eof = False

        def poll(self, timeout):
            if event_list:
                batch = event_list.pop(0)
                return batch if isinstance(batch, list) else [batch]
            self.eof = True
            return []

    import sidecar.linereader as lr
    monkeypatch.setattr(lr, "LineReader", FakeLineReader)

    polish_server.main()
    return [
        json.loads(line)
        for line in capsys.readouterr().out.splitlines()
        if line.strip()
    ]


def test_prewarm_resets_stale_stream_state_and_clears_pending_commit(
    monkeypatch, tmp_path, capsys
):
    sid = {"n": 0, "toks": []}

    def engine_handler(kind, arg1, arg2):
        if kind == "call":
            op = arg1.get("op")
            if op == "open":
                sid["n"] += 1
                sid["toks"] = list(arg1["prefix"])
                return {"ok": True, "session": sid["n"], "n": len(sid["toks"]), "ms": 0.1}
            if op == "rollback":
                del sid["toks"][arg1["n"] :]
                return {"ok": True, "n": arg1["n"]}
            if op == "forward":
                t, j = arg1["tokens"], arg1["logits_from"]
                sid["toks"] += t
                return {
                    "ok": True,
                    "ids": [99 for _ in range(j, len(t))],
                    "n": len(sid["toks"]),
                    "path": "mk",
                    "ms": 0.1,
                }
        return {
            "ok": True,
            "text": "one shot result",
            "tokens": 3,
            "prompt_tokens": 10,
        }

    # 1. stream_commit sets stream.commits = 1 and stream.draft
    # 2. Next batch has a stream_commit followed immediately by prewarm -> pending_commit is cleared by prewarm,
    #    and prewarm calls run_engine which calls stream_reset(lost=True), resetting stream.commits to 0.
    # 3. A subsequent run() sees stream.commits == 0 and runs one-shot instead of using stale stream state.
    events = [
        [{"command": "stream_commit", "inputText": "first dictation words"}],
        [
            {"command": "stream_commit", "inputText": "superseded commit"},
            {"command": "prewarm"},
        ],
        [{"command": "run", "inputText": "second dictation words"}],
        [{"command": "shutdown"}],
    ]
    replies = _run_polish_server_main(
        monkeypatch,
        tmp_path,
        capsys,
        use_engine=True,
        events=events,
        engine_handler=engine_handler,
    )
    commits = [r["stream"] for r in replies if "stream" in r]
    # Only the first stream_commit ran; the second was cleared when prewarm arrived
    assert len(commits) == 1
    assert commits[0]["commits"] == 1
    run_replies = [r for r in replies if "outputText" in r]
    assert len(run_replies) == 1
    # Because prewarm called stream_reset(lost=True), the subsequent run used one-shot native engine
    assert run_replies[0]["diagnostics"]["engine"] == "native"
    assert run_replies[0]["outputText"] == "one shot result"


def test_newer_stream_commit_aborts_in_flight_commit_and_preserves_tail_prefix(
    monkeypatch, tmp_path, capsys
):
    from sidecar.stream_decode import RollbackDecoder

    rollbacks = []

    class FakeMlxStreamEngine(RollbackDecoder):
        def __init__(self, model, tokenizer):
            super().__init__({99})

        def _truncate(self, n_keep):
            rollbacks.append(n_keep)

        def _pass(self, toks, row0, tie_from, tie_margin):
            return [99 for _ in range(row0, len(toks))], None

    # 1. Start first commit; during its should_abort poll, a second stream_commit arrives -> aborts first!
    # 2. The main loop then polls with timeout=0.0 ([]) and runs the second commit; during its should_abort
    #    poll ([]), nothing arrives -> completes!
    events = [
        [{"command": "stream_commit", "inputText": "first part of speech"}],
        [{"command": "stream_commit", "inputText": "first part of speech extended"}],
        [],
        [],
        [{"command": "shutdown"}],
    ]
    replies = _run_polish_server_main(
        monkeypatch,
        tmp_path,
        capsys,
        use_engine=False,
        events=events,
        stream_engine_cls=FakeMlxStreamEngine,
    )
    commits = [r["stream"] for r in replies if "stream" in r]
    assert len(commits) == 2
    assert commits[0]["aborted"] is True
    assert commits[1]["aborted"] is False


def test_mlx_run_streamed_engine_error_falls_back_to_mlx_one_shot_and_preserves_frozen_head(
    monkeypatch, tmp_path, capsys
):
    from sidecar.stream_decode import EngineError, RollbackDecoder

    class FailingTailMlxEngine(RollbackDecoder):
        def __init__(self, model, tokenizer):
            super().__init__({99})
            self.spec_calls = 0

        def _truncate(self, n_keep):
            pass

        def _pass(self, toks, row0, tie_from, tie_margin):
            return [99 for _ in range(row0, len(toks))], None

        def speculative(self, prompt, max_tokens, proposer, **kw):
            self.spec_calls += 1
            if self.spec_calls == 1:
                # Freezing the first head (130 words) succeeds -> return 130 tokens so guard_output passes
                return [100 + i for i in range(130)], {"passes": [5], "accepted": 130}
            # Tail fails with EngineError on MLX backend (where engine is None)
            raise EngineError("context cap reached on MLX")

    # 210 words (> FREEZE_WORDS = 200) with sentence boundaries, and commits == 0!
    # Also tests that `run` enters `run_streamed` when len(words) > FREEZE_WORDS even with stream.commits == 0.
    long_words = ["word."] * 150 + ["tailword"] * 60
    long_text = " ".join(long_words)
    events = [
        [{"command": "run", "inputText": long_text}],
        [{"command": "shutdown"}],
    ]
    replies = _run_polish_server_main(
        monkeypatch,
        tmp_path,
        capsys,
        use_engine=False,
        events=events,
        mlx_chunks=lambda prompt, max_tokens: " ".join(["tailout"] * 80),
        stream_engine_cls=FailingTailMlxEngine,
    )
    run_replies = [r for r in replies if "outputText" in r]
    assert len(run_replies) == 1
    assert run_replies[0]["diagnostics"]["guardFired"] is False
    out = run_replies[0]["outputText"]
    # Frozen head ("tok100 tok101 ...") is preserved and combined with the one-shot MLX fallback for the tail
    assert out.startswith("tok100 tok101 ")
    assert "tailout" in out
