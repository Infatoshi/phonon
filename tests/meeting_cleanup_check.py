"""Python reference for the app's meeting-cleanup check (bar/Sources/MeetingCleanup.swift,
`MeetingCleanupValidator.check`). The Swift checker and this file must give the same verdict on
every answer: tests/test_meeting_cleanup_check.py runs both on fixtures/meeting_cleanup_cases.json,
and `MeetingCleanupLiveCheck` compares the Swift side with answers scored here.

usage:
  meeting_cleanup_check.py [score] CHUNKS.jsonl GEN.jsonl > ANSWERS.jsonl
      {"id","text","verdict"} per answer; verdict counts to stderr.
  meeting_cleanup_check.py halves CHUNKS.jsonl ANSWERS.jsonl > HALVES.jsonl
      the second pass's inputs: {"id": "<chunk id>#h0" / "#h1", "text"} for each rejected answer.
  meeting_cleanup_check.py combine CHUNKS.jsonl GEN.jsonl HALF_GEN.jsonl > FINAL.jsonl
      per chunk: first verdict, the halves' verdicts, the final one; the kept-raw rate to stderr.
  CHUNKS: {"id","text"} with "Speaker N: words" lines; GEN and HALF_GEN: {"id","text"} answers.
"""

import collections
import json
import re
import sys

MIN_CHUNK_RECALL = 0.85
LONG_LINE_WORDS = 12
MIN_LONG_LINE_RECALL = 0.5
MAX_NEW_WORDS_PER_100 = 10.0
MAX_LENGTH_RATIO = 2.2

FILLERS = {
    "um", "uh", "uhm", "hmm", "mm", "mhm", "mmhmm", "mm-hmm", "uh-huh", "huh", "ah", "eh", "er", "erm", "hm",
    "mmm", "oh", "yeah", "okay", "ok", "right", "yes", "so", "like", "well", "mm-mm", "yep", "sure", "alright",
}  # fmt: skip
LABEL = re.compile(r"^\s*\**(Speaker \d+)\**\s*:\s*\**")
TLDR = re.compile(r"^\s*\**\s*TL;?\s*DR", re.IGNORECASE)


def tokens(text):
    """Lower case, everything but a-z 0-9 and apostrophes a space (Swift `tokens`)."""
    return re.sub(r"[^a-z0-9' ]", " ", text.lower()).split()


def filler_only(text):
    return all(t in FILLERS for t in tokens(text))


def labelled(line):
    m = LABEL.match(line)
    if not m:
        return None
    return m.group(1), line[m.end() :].strip(" \t")


def parse_input(text):
    return [x for x in (labelled(line) for line in text.split("\n")) if x is not None]


def parse(output):
    lines = []
    for raw in re.split(r"\r\n|\n|\r", output):
        line = raw.strip(" \t")
        if not line or TLDR.match(line):
            continue
        x = labelled(line)
        if x is None:
            return None
        lines.append(x)
    return lines


def recall(source, output):
    if not source:
        return 1.0
    have = collections.Counter(output)
    kept = 0
    for w in source:
        if have[w] > 0:
            have[w] -= 1
            kept += 1
    return kept / len(source)


def overlap(a, b):
    return sum(
        (collections.Counter(tokens(a)) & collections.Counter(tokens(b))).values()
    )


def can_stand_for(answer_line, input_line):
    """Same label, and a non-empty answer line of fillers only never stands for a line that held
    content (that line was dropped, not cleaned)."""
    if answer_line[0] != input_line[0]:
        return False
    text = answer_line[1]
    return not (text and filler_only(text) and not filler_only(input_line[1]))


def align(inp, answer):
    """Answer line k -> input index, strictly increasing, each answer line able to stand for its
    input line, every skipped input line filler-only; of those, the one sharing the most words
    with the input. None when there is none. Ties take the match over the skip, scanning from
    the end (Swift `align`, same order)."""
    n, m = len(inp), len(answer)
    NEG = None
    best = [[NEG] * (n + 1) for _ in range(m + 1)]
    best[0][0] = 0
    for j in range(1, n + 1):
        best[0][j] = (
            best[0][j - 1]
            if best[0][j - 1] is not None and filler_only(inp[j - 1][1])
            else NEG
        )
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            options = []
            if best[i][j - 1] is not None and filler_only(inp[j - 1][1]):
                options.append(best[i][j - 1])
            if best[i - 1][j - 1] is not None and can_stand_for(
                answer[i - 1], inp[j - 1]
            ):
                options.append(
                    best[i - 1][j - 1] + overlap(answer[i - 1][1], inp[j - 1][1])
                )
            best[i][j] = max(options) if options else NEG
    if best[m][n] is None:
        return None
    out = [0] * m
    i, j = m, n
    while i > 0:
        match = (
            best[i - 1][j - 1] is not None
            and can_stand_for(answer[i - 1], inp[j - 1])
            and best[i - 1][j - 1] + overlap(answer[i - 1][1], inp[j - 1][1])
            == best[i][j]
        )
        if match:
            out[i - 1] = j - 1
            i, j = i - 1, j - 1
        else:
            j -= 1
    return out


def structural_reason(inp, answer):
    """Why no alignment exists, by the greedy walk the check used before (its reason names)."""
    nxt = 0
    answered = set()
    for label, _ in answer:
        while nxt < len(inp) and inp[nxt][0] != label:
            if not filler_only(inp[nxt][1]):
                return "speaker order"
            nxt += 1
        if nxt >= len(inp):
            return "extra speaker line"
        answered.add(nxt)
        nxt += 1
    for k, (_, t) in enumerate(inp):
        if k not in answered and not filler_only(t):
            return "speaker line lost"
    return "speaker order"


def check(inp, output):
    """-> ("ok", cleaned per input line or None) or (reason, None)."""
    answer = parse(output)
    if answer is None:
        return "unlabelled line", None
    at = align(inp, answer)
    if at is None:
        return structural_reason(inp, answer), None
    cleaned = [None] * len(inp)
    for (_, text), k in zip(answer, at):
        if text:
            cleaned[k] = text
    input_words = [w for _, t in inp for w in tokens(t)]
    output_words = [w for t in cleaned if t is not None for w in tokens(t)]
    if len(output_words) > MAX_LENGTH_RATIO * max(1, len(input_words)):
        return "too long", None
    content = [w for w in input_words if w not in FILLERS]
    if recall(content, output_words) < MIN_CHUNK_RECALL:
        return "content lost", None
    for k, (_, t) in enumerate(inp):
        words = [w for w in tokens(t) if w not in FILLERS]
        if (
            len(words) >= LONG_LINE_WORDS
            and recall(words, tokens(cleaned[k] or "")) < MIN_LONG_LINE_RECALL
        ):
            return "line cut short", None
    ic, oc = collections.Counter(input_words), collections.Counter(output_words)
    added = sum(max(0, n - ic[w]) for w, n in oc.items())
    if 100 * added / max(1, len(output_words)) > MAX_NEW_WORDS_PER_100:
        return "words added", None
    return "ok", cleaned


def halves(inp):
    """The second pass's split (Swift `MeetingCleanupChunker.halves`): the line boundary nearest
    half the words, the earliest on a tie. None for a one-line chunk."""
    if len(inp) < 2:
        return None
    total = sum(len(t.split()) for _, t in inp)
    best, best_gap, words = 1, None, 0
    for cut in range(1, len(inp)):
        words += len(inp[cut - 1][1].split())
        gap = abs(2 * words - total)
        if best_gap is None or gap < best_gap:
            best, best_gap = cut, gap
    return inp[:best], inp[best:]


def render(lines):
    return "\n".join(f"{label}: {text}" for label, text in lines)


def combine(first, second, original):
    """Both halves' verdicts as the chunk's (Swift `CleanupOutcome.combine`)."""
    if first == "ok" and second == "ok":
        return "ok"
    if first == "ok" or second == "ok":
        return "mixed"
    return original


def load(path):
    return [json.loads(line) for line in open(path)]


def main():
    args = sys.argv[1:]
    mode = (
        args.pop(0) if args and args[0] in ("score", "halves", "combine") else "score"
    )
    chunks = {row["id"]: row["text"] for row in load(args[0])}
    counts = collections.Counter()
    if mode == "score":
        for g in load(args[1]):
            verdict, _ = check(parse_input(chunks[g["id"]]), g["text"])
            counts[verdict] += 1
            print(json.dumps({"id": g["id"], "text": g["text"], "verdict": verdict}))
    elif mode == "halves":
        # the inputs of the second pass: each rejected answer's chunk as two half chunks
        for a in load(args[1]):
            split = (
                halves(parse_input(chunks[a["id"]])) if a["verdict"] != "ok" else None
            )
            for k, part in enumerate(split or []):
                counts["halves"] += 1
                print(json.dumps({"id": f"{a['id']}#h{k}", "text": render(part)}))
    else:
        # per chunk after the second pass; GEN rows may repeat a chunk id across seeds, so pass
        # one seed's files at a time
        half_answers = {row["id"]: row["text"] for row in load(args[2])}
        for g in load(args[1]):
            inp = parse_input(chunks[g["id"]])
            verdict, _ = check(inp, g["text"])
            final, retried = verdict, None
            split = halves(inp) if verdict != "ok" else None
            if split:
                parts = [
                    check(part, half_answers[f"{g['id']}#h{k}"])[0]
                    for k, part in enumerate(split)
                ]
                final, retried = combine(parts[0], parts[1], verdict), parts
            counts[final] += 1
            print(
                json.dumps(
                    {
                        "id": g["id"],
                        "first": verdict,
                        "halves": retried,
                        "verdict": final,
                    }
                )
            )
        n = sum(counts.values())
        kept_raw = n - counts["ok"]
        print(
            f"kept raw (any line) {kept_raw}/{n} = {100 * kept_raw / max(1, n):.1f}%; "
            f"wholly raw {kept_raw - counts['mixed']}/{n}",
            file=sys.stderr,
        )
    print(dict(counts), file=sys.stderr)


if __name__ == "__main__":
    main()
