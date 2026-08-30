from sidecar.polish_server import (
    PROFILE_LEAD_IN,
    build_profile_block,
    describe_profile,
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
