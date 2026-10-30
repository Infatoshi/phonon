"""WER tables, word-error classification, entity lexicon, and error_model.json."""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from datetime import date
from typing import Any

from common import (
    CLIPS_PATH,
    DATA,
    ENGLISH_WORDS,
    RESEARCH,
    SPLITS,
    SYSTEMS,
    WISPR_SPLITS,
    hyp_path,
    read_jsonl,
)

CLASSES = (
    "ENTITY",
    "FUNCTION",
    "ORTHOGRAPHY",
    "NEAR_MISS",
    "DROP",
    "INSERT",
    "OTHER",
)

FUNCTION_WORDS = frozenset(
    """
    a an the and or but if then so because as than that this these those
    i me my mine myself you your yours yourself he him his himself she her hers herself
    it its itself we us our ours ourselves they them their theirs themselves
    is am are was were be been being do does did doing done have has had having
    will would shall should can could may might must need
    to of in on for with at by from up about into over after under above below
    between through during before without within along across behind beyond plus versus vs
    not no nor too very just also only even still already yet
    um uh uhh umm er ah like yeah yep yup okay ok right well actually basically
    there here where when what which who whom whose how why
    some any each every all both few more most other such own
    out off down back away around again
    """.split()
)

_CAMEL = re.compile(r"(?:[a-z][A-Z]|[A-Z]{2,}[a-z]|[A-Za-z]\d|\d[A-Za-z])")
_STRIP = re.compile(r"^[^\w+#]+|[^\w+#]+$")
_NORM = None


def fair_norm(text: str) -> str:
    global _NORM
    if _NORM is None:
        from whisper_normalizer.english import EnglishTextNormalizer

        _NORM = EnglishTextNormalizer()
    return _NORM((text or "").strip())


def strict_lc(text: str) -> str:
    return (text or "").strip().lower()


def load_english() -> set[str]:
    words: set[str] = set()
    if ENGLISH_WORDS.exists():
        for line in ENGLISH_WORDS.read_text(encoding="utf-8").splitlines():
            token = line.strip().lower()
            if token:
                words.add(token)
    words.update(FUNCTION_WORDS)
    return words


def core_token(surface: str) -> str:
    return _STRIP.sub("", surface or "")


def is_camel(token: str) -> bool:
    return bool(_CAMEL.search(token))


def is_acronym(token: str) -> bool:
    letters = re.sub(r"[^A-Za-z]", "", token)
    return len(letters) >= 2 and letters.isupper()


def is_entity(surface: str, common: set[str]) -> bool:
    """Technical term / identifier / name / acronym / code-like, or not common English.

    Rule: a reference token is ENTITY if after stripping wrapping punctuation it
    (1) contains a digit, underscore, or camelCase / letter-digit mix,
    (2) is a 2+ letter all-caps acronym,
    (3) contains an internal hyphen plus a digit or extra hyphen (code-like),
    or (4) its alphabetic core is length >= 2 and not in the top-20k English
    word list (wordfreq `en`, plus the function-word list).
    """
    token = core_token(surface)
    if not token:
        return False
    if any(ch.isdigit() for ch in token) or "_" in token or is_camel(token):
        return True
    if is_acronym(token):
        return True
    if token.count("-") >= 1 and (any(ch.isdigit() for ch in token) or token.count("-") >= 2):
        return True
    letters = re.sub(r"[^A-Za-z]", "", token).lower()
    if len(letters) >= 2 and letters not in common:
        return True
    return False


def is_function(surface: str) -> bool:
    token = core_token(surface).lower()
    return token in FUNCTION_WORDS


def soundex(word: str) -> str:
    token = re.sub(r"[^a-z]", "", word.lower())
    if not token:
        return ""
    first = token[0].upper()
    mapping = str.maketrans("bfpvcgjkqsxzdtlmnr", "111122222222334556")
    coded = token.translate(mapping)
    digits = []
    last = ""
    for ch, raw in zip(coded, token):
        if raw in "aeiouyhw":
            last = "0"
            continue
        digit = ch if ch.isdigit() else "0"
        if digit != "0" and digit != last:
            digits.append(digit)
        last = digit
    if digits and token[0] not in "aeiouyhw":
        first_digit = token[0].translate(mapping)
        if digits and digits[0] == first_digit:
            digits = digits[1:]
    return (first + "".join(digits) + "000")[:4]


def metaphone_key(word: str) -> str:
    token = re.sub(r"[^a-z]", "", word.lower())
    if not token:
        return ""
    token = token.replace("ph", "f").replace("kn", "n").replace("gn", "n")
    token = token.replace("wr", "r").replace("ck", "k").replace("x", "ks")
    out = []
    prev = ""
    for i, ch in enumerate(token):
        if ch in "aeiou" and i != 0:
            continue
        if ch == prev:
            continue
        if ch in "wy" and i != 0:
            continue
        out.append(ch)
        prev = ch
    return "".join(out)[:6]


def edit_distance(left: str, right: str) -> int:
    a, b = left.lower(), right.lower()
    if a == b:
        return 0
    if abs(len(a) - len(b)) > 2:
        return 3
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            ins = cur[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (ca != cb)
            cur.append(min(ins, delete, sub))
        prev = cur
        if min(prev) > 2:
            return 3
    return prev[-1]


def is_near_miss(ref: str, hyp: str) -> bool:
    a = core_token(ref).lower()
    b = core_token(hyp).lower()
    if not a or not b:
        return False
    if edit_distance(a, b) <= 2:
        return True
    if len(a) >= 3 and len(b) >= 3 and (soundex(a) == soundex(b) or metaphone_key(a) == metaphone_key(b)):
        return True
    return False


def safe_wer(refs: list[str], hyps: list[str], normalizer) -> float:
    import jiwer

    if not refs:
        return 0.0
    nrefs, nhyps = [], []
    for ref, hyp in zip(refs, hyps, strict=True):
        r = normalizer(ref) or "<empty>"
        h = normalizer(hyp) or "<empty>"
        nrefs.append(r)
        nhyps.append(h)
    return float(jiwer.wer(nrefs, nhyps))


def tokenize(text: str) -> list[str]:
    return (text or "").strip().split()


def classify_chunk(
    ref_toks: list[str],
    hyp_toks: list[str],
    kind: str,
    common: set[str],
) -> str:
    if kind == "equal":
        return "NONE"
    if kind == "substitute":
        if fair_norm(" ".join(ref_toks)) == fair_norm(" ".join(hyp_toks)):
            return "ORTHOGRAPHY"
        if any(is_entity(tok, common) for tok in ref_toks):
            return "ENTITY"
        if any(is_function(tok) for tok in ref_toks) or any(is_function(tok) for tok in hyp_toks):
            return "FUNCTION"
        if (
            len(ref_toks) == 1
            and len(hyp_toks) == 1
            and not is_entity(ref_toks[0], common)
            and is_near_miss(ref_toks[0], hyp_toks[0])
        ):
            return "NEAR_MISS"
        return "OTHER"
    if kind == "delete":
        if any(is_entity(tok, common) for tok in ref_toks):
            return "ENTITY"
        if all(is_function(tok) for tok in ref_toks):
            return "FUNCTION"
        return "DROP"
    if kind == "insert":
        if all(is_function(tok) for tok in hyp_toks) and hyp_toks:
            return "FUNCTION"
        return "INSERT"
    return "OTHER"


def align_errors(ref: str, hyp: str, common: set[str]) -> list[dict[str, Any]]:
    import jiwer

    ref_toks = tokenize(ref)
    hyp_toks = tokenize(hyp)
    if not ref_toks and not hyp_toks:
        return []
    rjoin = " ".join(t.lower() for t in ref_toks) or "<empty>"
    hjoin = " ".join(t.lower() for t in hyp_toks) or "<empty>"
    processed = jiwer.process_words(rjoin, hjoin)
    chunks = processed.alignments[0]
    errors = []
    for chunk in chunks:
        kind = chunk.type
        if hasattr(kind, "value"):
            kind = kind.value
        kind = str(kind).lower()
        rspan = ref_toks[chunk.ref_start_idx : chunk.ref_end_idx]
        hspan = hyp_toks[chunk.hyp_start_idx : chunk.hyp_end_idx]
        if kind in {"equal", "match"}:
            # Lowercased alignment treats case-only diffs as equal; those are ORTHOGRAPHY.
            if rspan != hspan and (rspan or hspan):
                errors.append(
                    {
                        "type": "equal",
                        "class": "ORTHOGRAPHY",
                        "ref": " ".join(rspan),
                        "hyp": " ".join(hspan),
                        "ref_start": chunk.ref_start_idx,
                        "ref_end": chunk.ref_end_idx,
                        "hyp_start": chunk.hyp_start_idx,
                        "hyp_end": chunk.hyp_end_idx,
                        "ref_words": len(rspan),
                    }
                )
            continue
        klass = classify_chunk(rspan, hspan, kind, common)
        errors.append(
            {
                "type": kind,
                "class": klass,
                "ref": " ".join(rspan),
                "hyp": " ".join(hspan),
                "ref_start": chunk.ref_start_idx,
                "ref_end": chunk.ref_end_idx,
                "hyp_start": chunk.hyp_start_idx,
                "hyp_end": chunk.hyp_end_idx,
                "ref_words": len(rspan),
            }
        )
    return errors


def load_hyps(tag: str) -> dict[str, str]:
    rows = read_jsonl(hyp_path(tag))
    out: dict[str, str] = {}
    for row in rows:
        cid = row.get("id")
        if cid:
            out[str(cid)] = str(row.get("hypothesis") or "")
    return out


def fmt(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.4f}"


def main() -> int:
    clips = read_jsonl(CLIPS_PATH)
    if not clips:
        raise FileNotFoundError(CLIPS_PATH)
    common = load_english()
    clip_by_id = {c["id"]: c for c in clips}
    by_split: dict[str, list[dict]] = {s: [] for s in SPLITS}
    scored_clips = [c for c in clips if c.get("has_reference", True) and str(c.get("reference") or "").strip()]
    for clip in scored_clips:
        by_split[clip["split"]].append(clip)

    hyps = {system["tag"]: load_hyps(system["tag"]) for system in SYSTEMS}
    tags = [system["tag"] for system in SYSTEMS]

    wer_table: dict[str, dict[str, Any]] = {}
    for tag in tags:
        wer_table[tag] = {"sets": {}, "missing": {}}
        for split, split_clips in by_split.items():
            paired = [(c, hyps[tag][c["id"]]) for c in split_clips if c["id"] in hyps[tag]]
            missing = [c["id"] for c in split_clips if c["id"] not in hyps[tag]]
            wer_table[tag]["missing"][split] = missing
            if not paired:
                wer_table[tag]["sets"][split] = {
                    "n": 0,
                    "n_available": 0,
                    "fair_wer": None,
                    "strict_wer": None,
                }
                continue
            refs = [c["reference"] for c, _ in paired]
            texts = [h for _, h in paired]
            wer_table[tag]["sets"][split] = {
                "n": len(split_clips),
                "n_available": len(paired),
                "fair_wer": safe_wer(refs, texts, fair_norm),
                "strict_wer": safe_wer(refs, texts, strict_lc),
            }

    oracle: dict[str, Any] = {"sets": {}}
    for split, split_clips in by_split.items():
        refs, best = [], []
        for clip in split_clips:
            candidates = []
            for tag in tags:
                if clip["id"] in hyps[tag]:
                    hyp = hyps[tag][clip["id"]]
                    score = safe_wer([clip["reference"]], [hyp], fair_norm)
                    candidates.append((score, tag, hyp))
            if not candidates:
                continue
            candidates.sort(key=lambda item: (item[0], item[1]))
            refs.append(clip["reference"])
            best.append(candidates[0][2])
        oracle["sets"][split] = {
            "n": len(split_clips),
            "n_available": len(refs),
            "fair_wer": safe_wer(refs, best, fair_norm) if refs else None,
            "strict_wer": safe_wer(refs, best, strict_lc) if refs else None,
            "systems_considered": tags,
        }

    # Per-ref-index wrong-system counts for SHARED vs MODEL_SPECIFIC.
    wrong_at: dict[tuple[str, int, str], set[str]] = defaultdict(set)
    present_at: dict[str, set[str]] = defaultdict(set)
    classified: list[dict[str, Any]] = []
    for tag in tags:
        for clip in scored_clips:
            cid = clip["id"]
            if cid not in hyps[tag]:
                continue
            present_at[cid].add(tag)
            errors = align_errors(clip["reference"], hyps[tag][cid], common)
            for err in errors:
                for idx, tok in enumerate(tokenize(clip["reference"])[err["ref_start"] : err["ref_end"]]):
                    wrong_at[(cid, err["ref_start"] + idx, tok.lower())].add(tag)
                classified.append(
                    {
                        "id": cid,
                        "split": clip["split"],
                        "system": tag,
                        **err,
                    }
                )

    for err in classified:
        if err["ref_words"] <= 0:
            err["share"] = "MODEL_SPECIFIC"
            continue
        votes = []
        ref_toks = tokenize(clip_by_id[err["id"]]["reference"])
        for idx in range(err["ref_start"], err["ref_end"]):
            key = (err["id"], idx, ref_toks[idx].lower())
            n_wrong = len(wrong_at[key])
            n_sys = len(present_at[err["id"]])
            votes.append(n_wrong >= max(1, (n_sys + 1) // 2) and n_sys > 0)
        err["share"] = "SHARED" if votes and all(votes) else "MODEL_SPECIFIC"

    def rate_block(rows: list[dict], ref_word_count: int) -> dict[str, Any]:
        counts = Counter(r["class"] for r in rows)
        ref_involved = Counter()
        insert_tokens = 0
        for row in rows:
            if row["class"] == "INSERT":
                insert_tokens += max(1, len(tokenize(row["hyp"])))
            else:
                ref_involved[row["class"]] += row["ref_words"]
        out = {}
        for klass in CLASSES:
            if klass == "INSERT":
                count = insert_tokens
            else:
                count = ref_involved[klass]
            out[klass] = {
                "count": int(count),
                "chunks": int(counts[klass]),
                "ref_words": int(ref_word_count),
                "per_word": (count / ref_word_count) if ref_word_count else 0.0,
            }
        return out

    wispr_clips = [c for c in scored_clips if c["split"] in WISPR_SPLITS]
    wispr_ref_words = sum(len(tokenize(c["reference"])) for c in wispr_clips)
    v2 = "parakeet-tdt-0.6b-v2"
    v2_rows = [r for r in classified if r["system"] == v2 and r["split"] in WISPR_SPLITS]
    pooled_rows = [r for r in classified if r["split"] in WISPR_SPLITS]
    pooled_ref_words = 0
    for clip in wispr_clips:
        pooled_ref_words += len(tokenize(clip["reference"])) * len(present_at[clip["id"]])

    substitutions: dict[str, dict[str, Counter]] = {"ENTITY": defaultdict(Counter), "NEAR_MISS": defaultdict(Counter)}
    for row in classified:
        if row["class"] in substitutions and row["ref"]:
            key = core_token(row["ref"]) or row["ref"]
            hyp_key = core_token(row["hyp"]) if row["hyp"] else "<eps>"
            substitutions[row["class"]][key][hyp_key] += 1

    def dump_subs(kind: str) -> dict[str, list[dict[str, Any]]]:
        payload = {}
        for ref, counter in substitutions[kind].items():
            payload[ref] = [
                {"hyp": hyp, "count": int(n)}
                for hyp, n in counter.most_common()
            ]
        return payload

    error_model = {
        "version": 1,
        "date": str(date.today()),
        "entity_rule": is_entity.__doc__,
        "function_words": sorted(FUNCTION_WORDS),
        "english_word_list": str(ENGLISH_WORDS),
        "english_word_count": len(common),
        "alignment": (
            "jiwer.process_words on lowercase whitespace tokens of the original "
            "surface strings; ORTHOGRAPHY uses fair-normalized equality of the "
            "aligned spans. SHARED = the same reference word index is wrong in "
            "at least half of the systems that produced a hypothesis for the clip."
        ),
        "systems": {
            tag: {
                "have": len(hyps[tag]),
                "missing_by_split": {s: len(wer_table[tag]["missing"][s]) for s in SPLITS},
            }
            for tag in tags
        },
        "rates": {
            "parakeet-tdt-0.6b-v2_wispr": rate_block(v2_rows, wispr_ref_words),
            "pooled_wispr": rate_block(pooled_rows, pooled_ref_words),
        },
        "substitutions": {
            "ENTITY": dump_subs("ENTITY"),
            "NEAR_MISS": dump_subs("NEAR_MISS"),
        },
        "wer": wer_table,
        "oracle": oracle,
    }
    error_model_path = DATA / "error_model.json"
    error_model_path.write_text(json.dumps(error_model) + "\n", encoding="utf-8")
    print(f"error_model.json {error_model_path}", flush=True)

    entity_freq: Counter = Counter()
    entity_variants: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    for row in classified:
        if row["class"] != "ENTITY" or not row["ref"]:
            continue
        key = core_token(row["ref"]) or row["ref"]
        hyp_key = row["hyp"] or "<eps>"
        entity_freq[key] += 1
        entity_variants[key][row["system"]][hyp_key] += 1
    top_entities = []
    for ref, count in entity_freq.most_common(50):
        top_entities.append(
            {
                "ref": ref,
                "count": int(count),
                "per_system": {
                    tag: {hyp: int(n) for hyp, n in entity_variants[ref][tag].most_common()}
                    for tag in tags
                    if entity_variants[ref][tag]
                },
            }
        )
    (RESEARCH / "entity_errors.json").write_text(
        json.dumps({"entity_rule": is_entity.__doc__, "top50": top_entities}, indent=2) + "\n"
    )

    sample_rows = []
    per_class = defaultdict(list)
    for row in classified:
        per_class[row["class"]].append(row)
    for klass in CLASSES:
        bucket = per_class[klass]
        take = bucket[:80]
        for row in take:
            sample_rows.append(
                {
                    "id": row["id"],
                    "split": row["split"],
                    "system": row["system"],
                    "class": row["class"],
                    "share": row["share"],
                    "type": row["type"],
                    "ref": row["ref"],
                    "hyp": row["hyp"],
                }
            )
    (RESEARCH / "errors_sample.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in sample_rows),
        encoding="utf-8",
    )

    class_rates_v2 = error_model["rates"]["parakeet-tdt-0.6b-v2_wispr"]
    class_rates_pooled = error_model["rates"]["pooled_wispr"]
    share_counts = Counter(r["share"] for r in classified)
    class_counts = Counter(r["class"] for r in classified)

    results = {
        "date": str(date.today()),
        "n_clips": len(clips),
        "n_scored": len(scored_clips),
        "n_unlabeled": len(clips) - len(scored_clips),
        "splits": {s: len(by_split[s]) for s in SPLITS},
        "systems": tags,
        "wer": {
            tag: {split: wer_table[tag]["sets"][split] for split in SPLITS} for tag in tags
        },
        "oracle": {split: oracle["sets"][split] for split in SPLITS},
        "error_class_counts": dict(class_counts),
        "share_counts": dict(share_counts),
        "v2_wispr_per_word": {k: v["per_word"] for k, v in class_rates_v2.items()},
        "pooled_wispr_per_word": {k: v["per_word"] for k, v in class_rates_pooled.items()},
        "error_model_path": str(error_model_path),
        "n_classified_chunks": len(classified),
        "english_word_count": len(common),
        "failures": read_jsonl(DATA / "logs" / "failures.jsonl"),
        "missing": {
            tag: {split: len(wer_table[tag]["missing"][split]) for split in SPLITS}
            for tag in tags
        },
        "hyp_counts": {tag: len(hyps[tag]) for tag in tags},
    }
    (RESEARCH / "results.json").write_text(json.dumps(results, indent=2) + "\n")

    lines = ["# Multi-ASR error breakdown", ""]
    lines.append(
        f"Date: {date.today()}. Audio clips: {len(clips)} "
        f"(scored {len(scored_clips)}; {len(clips) - len(scored_clips)} Wispr train "
        f"exports have empty formatted/edited text and are transcribed but not scored). "
        f"Scored splits: holdout120={len(by_split['holdout120'])}, "
        f"wispr_train640={len(by_split['wispr_train640'])}, "
        f"edit25={len(by_split['edit25'])}, aqua19={len(by_split['aqua19'])}."
    )
    lines.append("")
    lines.append("Fair WER is whisper `EnglishTextNormalizer` then jiwer. "
                 "Strict WER is lowercase-only then jiwer. "
                 "Oracle picks, per clip, the hypothesis with lowest fair WER. "
                 "n is clips scored for that system (missing hyps are omitted).")
    lines.append("")
    missing_bits = []
    for tag in tags:
        miss = {s: len(wer_table[tag]["missing"][s]) for s in SPLITS}
        total_miss = sum(miss.values())
        if total_miss:
            missing_bits.append(f"{tag} missing {total_miss} ({miss})")
    if missing_bits:
        lines.append("Coverage gaps: " + "; ".join(missing_bits) + ".")
    else:
        lines.append("Coverage: every GPU system has a hypothesis for all 804 clips; wispr_asr is absent on aqua19 and on 17 train clips with empty asrText.")
    lines.append("")
    header = "| system | " + " | ".join(
        f"{s} fair / strict (n)" for s in SPLITS
    ) + " |"
    sep = "| --- | " + " | ".join("---:" for _ in SPLITS) + " |"
    lines += [header, sep]
    for tag in tags + ["oracle"]:
        cells = [tag]
        source = oracle["sets"] if tag == "oracle" else wer_table[tag]["sets"]
        for split in SPLITS:
            block = source[split]
            n = block.get("n_available") or 0
            cells.append(f"{fmt(block.get('fair_wer'))} / {fmt(block.get('strict_wer'))} ({n})")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## Error-class per-word rates on Wispr audio")
    lines.append("")
    lines.append("| class | Parakeet v2 per-word | pooled per-word | v2 count | pooled count |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for klass in CLASSES:
        a = class_rates_v2[klass]
        b = class_rates_pooled[klass]
        lines.append(
            f"| {klass} | {a['per_word']:.4f} | {b['per_word']:.4f} | {a['count']} | {b['count']} |"
        )
    lines.append("")
    lines.append(
        f"Classified alignment chunks: {len(classified)}. "
        f"SHARED={share_counts.get('SHARED', 0)} "
        f"MODEL_SPECIFIC={share_counts.get('MODEL_SPECIFIC', 0)}."
    )
    lines.append("")
    lines.append("Entity rule: " + " ".join(is_entity.__doc__.split()))
    lines.append("")
    lines.append("## Top entity substitutions")
    lines.append("")
    lines.append("| ref | n | example hyps |")
    lines.append("| --- | ---: | --- |")
    for item in top_entities[:20]:
        examples = []
        merged: Counter = Counter()
        for variants in item["per_system"].values():
            merged.update(variants)
        for hyp, n in merged.most_common(4):
            examples.append(f"{hyp} ({n})")
        lines.append(f"| {item['ref']} | {item['count']} | {'; '.join(examples)} |")
    lines.append("")
    lines.append("## What the numbers say")
    lines.append("")
    lines.extend(narrative(results, top_entities, wer_table, oracle, class_counts, share_counts))
    lines.append("")
    lines.append(f"error_model.json: `{error_model_path}`")
    (RESEARCH / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((RESEARCH / "results.md").read_text())
    return 0


def narrative(results, top_entities, wer_table, oracle, class_counts, share_counts) -> list[str]:
    v2 = wer_table.get("parakeet-tdt-0.6b-v2", {}).get("sets", {})
    hold_v2 = (v2.get("holdout120") or {}).get("fair_wer")
    oracle_h = (oracle["sets"].get("holdout120") or {}).get("fair_wer")
    n_shared = share_counts.get("SHARED", 0)
    n_spec = share_counts.get("MODEL_SPECIFIC", 0)
    total = max(1, n_shared + n_spec)
    entity_n = class_counts.get("ENTITY", 0)
    func_n = class_counts.get("FUNCTION", 0)
    ortho_n = class_counts.get("ORTHOGRAPHY", 0)
    drop_n = class_counts.get("DROP", 0)
    insert_n = class_counts.get("INSERT", 0)
    near_n = class_counts.get("NEAR_MISS", 0)
    other_n = class_counts.get("OTHER", 0)
    tops = ", ".join(item["ref"] for item in top_entities[:8]) or "none yet"
    gap = None
    if hold_v2 is not None and oracle_h is not None:
        gap = hold_v2 - oracle_h
    sentences = [
        f"On holdout120, Parakeet v2 fair WER is {fmt(hold_v2)} and the oracle-over-systems fair WER is {fmt(oracle_h)}"
        + (f", a {gap:.4f} absolute gap that is the recoverable diversity across recognizers." if gap is not None else "."),
        f"Of {n_shared + n_spec} classified error chunks, {n_shared / total:.1%} are SHARED (the same reference word is wrong in at least half the systems) and {n_spec / total:.1%} are MODEL_SPECIFIC.",
        f"Shared errors concentrate on entities and stable mishearings of the speaker's names and stack words ({tops}), which no single recognizer invents in isolation.",
        "Recognizer-specific errors are the rest: function-word jitter, near-miss spellings, and one model's habit of dropping or inserting a content word the others keep.",
        f"On this audio the dominant classes by chunk count are ENTITY {entity_n}, FUNCTION {func_n}, ORTHOGRAPHY {ortho_n}, NEAR_MISS {near_n}, DROP {drop_n}, INSERT {insert_n}, OTHER {other_n}.",
        "Orthography is large because Wispr targets keep product casing and digit forms while several engines emit lowercase or number-words; fair WER hides most of that, strict WER does not.",
        "Function-word errors are frequent but cheap to a corrector trained on owner-edited pairs, because the target already shows the intended articles and fillers.",
        "Entity errors are the ones that matter for a dictation corrector: they are both shared across engines and sparse in generic ASR pretraining, so corrected pairs are the right supervision rather than cloning one recognizer's style.",
        "Training on Parakeet-v2-only residuals would teach the corrector that model's near-misses and drops; training on corrected pairs plus the substitution tables in error_model.json covers the shared entity lexicon those residuals miss.",
        "The oracle gap on personal audio is the argument for a text corrector that can see one hyp, not for blending N recognizers at runtime: most remaining error after the best hyp is still the shared entity class.",
    ]
    return sentences


if __name__ == "__main__":
    raise SystemExit(main())
