"""Deterministic symbol index: tree-sitter definitions + doc/config/flag terms over a repo tree.

This is the context model's input.  It produces (term, kind, count, n_files, repos, langs)
rows; an LLM only ranks what this produces.

Usage:
  python symbol_index.py --roots /home/user/dev /home/user/phonon \
      --out /data/phonon_index_v0/symbols_local.jsonl --stats /data/phonon_index_v0/index_stats.json
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

MAX_FILE_BYTES = 1_500_000

SKIP_DIRS = {
    ".git", ".venv", "venv", "node_modules", "target", "build", "dist", "__pycache__",
    ".cache", "wandb", "datasets", "data", "runs", "checkpoints", ".mypy_cache", ".ruff_cache",
    "site-packages", "weights", "ckpt", "wavs", "shards", ".tox", ".pytest_cache", ".next",
    "coverage", "htmlcov", "vendor", "cmakefiles", "cmake-build-debug", "cmake-build-release",
    ".idea", ".vscode", "tts_wavs", "chunks", "adapters", ".uv", ".cargo", ".npm", ".yarn",
    ".turbo", "artifacts", "tmp", "temp", ".eggs", "wheels", ".huggingface", "hub", "parquet",
    "audio", "videos", "images", ".ipynb_checkpoints", "__pypackages__", "deriveddata",
    "xcuserdata", "egg-info", "third_party", "llvm-project", "logs", "output", "out",
    ".gradle", ".swiftpm", ".bundle", "testdata", "fixtures", "golden", ".pre-commit",
}

LANG_BY_EXT = {
    ".py": "python", ".pyi": "python",
    ".rs": "rust",
    ".c": "c", ".h": "c",
    ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp", ".hpp": "cpp", ".hh": "cpp", ".hxx": "cpp",
    ".cu": "cuda", ".cuh": "cuda",
    ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    ".ts": "typescript", ".tsx": "tsx",
    ".go": "go",
}
DOC_EXT = {".md", ".markdown", ".rst"}
CFG_EXT = {".toml", ".yaml", ".yml", ".json", ".cfg", ".ini", ".conf"}
SH_EXT = {".sh", ".bash", ".zsh"}
ALL_EXT = set(LANG_BY_EXT) | DOC_EXT | CFG_EXT | SH_EXT

# ------------------------------------------------------------------ node maps
NODES = {
    "python": {"function_definition": "function", "class_definition": "class"},
    "rust": {"function_item": "function", "struct_item": "struct", "enum_item": "enum",
             "trait_item": "trait", "mod_item": "module", "type_item": "type",
             "const_item": "constant", "static_item": "constant", "union_item": "struct",
             "macro_definition": "macro"},
    "go": {"function_declaration": "function", "method_declaration": "method",
           "type_spec": "type", "const_spec": "constant"},
    "javascript": {"function_declaration": "function", "generator_function_declaration": "function",
                   "class_declaration": "class", "method_definition": "method",
                   "variable_declarator": "constant"},
    "c": {"struct_specifier": "struct", "enum_specifier": "enum", "type_definition": "type",
          "preproc_def": "macro", "preproc_function_def": "macro",
          "function_definition": "function", "declaration": "function"},
}
NODES["typescript"] = dict(NODES["javascript"], interface_declaration="interface",
                           type_alias_declaration="type", enum_declaration="enum",
                           abstract_class_declaration="class")
NODES["tsx"] = NODES["typescript"]
NODES["cpp"] = dict(NODES["c"], class_specifier="class", namespace_definition="module",
                    alias_declaration="type", concept_definition="type")
NODES["cuda"] = NODES["cpp"]

C_LIKE = {"c", "cpp", "cuda"}
NAME_TYPES = {"identifier", "type_identifier", "field_identifier", "property_identifier",
              "constant", "primitive_type", "namespace_identifier", "word", "package_identifier"}

# ------------------------------------------------------------------ filters
COMMON = None  # filled from the english list at aggregation time
KEYWORDS = frozenset("""
int float void char bool true false null none nil self this super class def fn func function
return yield async await import from export const let var pub priv private public static final
abstract virtual override impl trait struct enum type interface module package namespace using
include define ifdef ifndef endif pragma template typename sizeof alignas alignof constexpr
noexcept switch case break continue for while do if else elif try catch except finally throw
raise with as in is not and or pass lambda assert del global nonlocal match where crate mod use
mut ref dyn box printf println unwrap expect clone copy drop into default main str string list
dict set tuple len range print open new delete size data value key name args kwargs
""".split())

TERM_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-\./]{1,79}$")
HEAD_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$", re.M)
SPAN_RE = re.compile(r"`([^`\n]{2,60})`")
FLAG_RE = re.compile(r"(?:^|[\s=`\"'(\[])(--[A-Za-z][A-Za-z0-9-]{1,40})\b")
CFG_RE = re.compile(r"^\s*[\"']?([A-Za-z_][A-Za-z0-9_.\-]{1,60})[\"']?\s*[:=]", re.M)
IDENTISH = re.compile(r"[A-Za-z][A-Za-z0-9_\-\./]*")
HEXISH = re.compile(r"^(0x)?[0-9a-f]{6,}$", re.I)

# tier B: mentions.  Technical-looking tokens anywhere in the text, plus command and
# product seeds (these never appear as a definition site, so a definition-only index
# can never produce them). The user's own names and hostnames come from --seeds, a
# file from their profile, one term per line; none are hardcoded here.
MENTION_RE = re.compile(r"[A-Za-z][A-Za-z0-9_\-\.]{2,60}")
GENERIC_SEEDS = (
    r"uv|pytest|ruff|nvidia-smi|nvcc|ncu|nsys|cmake|ninja|rsync|tmux|docker|kubectl|"
    r"huggingface-cli|hyperfine|croc|bun|cargo|rustc|node|npm|pnpm|ffmpeg|"
    r"sox|jq|rg|fd|scp|blackwell|Grok|Opus|Codex|"
    r"Claude|Fable|parakeet|whisper|wispr|jiwer|metaphone|rapidfuzz"
)
SEED_RE = re.compile(rf"\b({GENERIC_SEEDS})\b", re.IGNORECASE)


@functools.lru_cache(maxsize=4)
def seed_regex(extra: tuple[str, ...]) -> re.Pattern:
    """SEED_RE plus the user's own seeds (names, hostnames) from --seeds."""
    if not extra:
        return SEED_RE
    alternatives = "|".join(re.escape(term) for term in extra)
    return re.compile(rf"\b({GENERIC_SEEDS}|{alternatives})\b", re.IGNORECASE)


def load_seeds(path: str) -> tuple[str, ...]:
    """One term per line; blank lines and # comments are skipped."""
    if not path:
        return ()
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return tuple(sorted({ln.strip() for ln in lines if ln.strip() and not ln.startswith("#")}))


def looks_technical(t: str) -> bool:
    """Heading / code-span tokens only survive if they look like a term, not prose."""
    if "_" in t or "." in t or "/" in t or "-" in t:
        return True
    if re.search(r"[a-z][A-Z]", t):
        return True
    if re.fullmatch(r"[A-Z]{2,8}", t):
        return True
    if re.search(r"\d", t):
        return True
    return False


def valid(term: str) -> bool:
    if not term or len(term) > 80 or len(term) < 2:
        return False
    if term.startswith("--"):
        return re.fullmatch(r"--[A-Za-z][A-Za-z0-9-]{1,40}", term) is not None
    if TERM_RE.match(term) is None:
        return False
    alpha = re.sub(r"[^A-Za-z]", "", term)
    if len(alpha) < 2:
        return False
    if HEXISH.match(term):
        return False
    if term.count("_") >= 5 or term.count("/") >= 4:
        return False
    if term.lower() in KEYWORDS:
        return False
    return True


# ------------------------------------------------------------------ parsing
_PARSERS: dict = {}


def parser(lang: str):
    p = _PARSERS.get(lang)
    if p is None:
        from tree_sitter_language_pack import get_parser
        p = get_parser(lang)
        _PARSERS[lang] = p
    return p


def _first_name(node, src):
    n = node.child_by_field_name("name")
    if n is not None:
        return src[n.start_byte:n.end_byte].decode("utf-8", "replace")
    for ch in node.children:
        if ch.type in NAME_TYPES:
            return src[ch.start_byte:ch.end_byte].decode("utf-8", "replace")
    return None


def _c_name(node, src):
    """Descend declarator chain of a C/C++ function definition or declaration."""
    cur = node.child_by_field_name("declarator")
    hops = 0
    while cur is not None and hops < 8:
        if cur.type in ("identifier", "field_identifier", "type_identifier",
                        "qualified_identifier", "operator_name", "destructor_name"):
            txt = src[cur.start_byte:cur.end_byte].decode("utf-8", "replace")
            return txt.split("::")[-1] if "::" in txt else txt
        nxt = cur.child_by_field_name("declarator")
        if nxt is None:
            for ch in cur.children:
                if ch.type in ("identifier", "field_identifier", "qualified_identifier"):
                    nxt = ch
                    break
        cur = nxt
        hops += 1
    return None


def walk_tree(root, src, lang, emit):
    nmap = NODES[lang]
    stack = [root]
    while stack:
        n = stack.pop()
        t = n.type
        kind = nmap.get(t)
        if kind is not None:
            if lang in C_LIKE and t in ("function_definition", "declaration"):
                nm = _c_name(n, src)
                if nm:
                    emit(nm, "function")
            else:
                nm = _first_name(n, src)
                if nm:
                    if lang == "go" and t == "const_spec":
                        emit(nm, "constant")
                    else:
                        emit(nm, kind)
        elif t in ("import_statement", "import_from_statement") and lang == "python":
            for ch in n.children:
                if ch.type in ("dotted_name", "aliased_import"):
                    emit(src[ch.start_byte:ch.end_byte].decode("utf-8", "replace").split(" as ")[0],
                         "module")
        elif t == "use_declaration" and lang == "rust":
            txt = src[n.start_byte:n.end_byte].decode("utf-8", "replace")
            for seg in re.split(r"[^A-Za-z0-9_]+", txt)[1:]:
                if seg and seg not in ("use", "crate", "self", "super", "as"):
                    emit(seg, "module")
        elif t == "preproc_include" and lang in C_LIKE:
            p = n.child_by_field_name("path")
            if p is not None:
                txt = src[p.start_byte:p.end_byte].decode("utf-8", "replace").strip('<>"')
                emit(txt.split("/")[-1], "file")
        elif t == "import_spec" and lang == "go":
            txt = src[n.start_byte:n.end_byte].decode("utf-8", "replace").strip().strip('"')
            if txt:
                emit(txt.split("/")[-1].strip('"'), "module")
        elif t == "package_clause" and lang == "go":
            nm = _first_name(n, src)
            if nm:
                emit(nm, "module")
        stack.extend(n.children)


def scan_file(path: str, ext: str, mentions: bool = False, seeds: tuple[str, ...] = ()):
    """-> list of (term, kind, lang)."""
    out = []
    try:
        raw = Path(path).read_bytes()
    except OSError:
        return out
    if len(raw) > MAX_FILE_BYTES:
        raw = raw[:MAX_FILE_BYTES]
    lang = LANG_BY_EXT.get(ext)
    seen = set()

    def emit(term, kind, lg=None):
        term = (term or "").strip().strip("`\"'.,;:()[]{}<>")
        if not valid(term):
            return
        key = (term, kind)
        if key in seen:
            return
        seen.add(key)
        out.append((term, kind, lg or lang or "text"))

    if lang:
        try:
            tree = parser(lang).parse(raw)
            walk_tree(tree.root_node, raw, lang, emit)
        except Exception:
            pass
        if ext in (".md",):
            pass
    text = None
    if ext in DOC_EXT or ext in CFG_EXT or ext in SH_EXT:
        text = raw.decode("utf-8", "replace")
    if ext in DOC_EXT:
        for m in HEAD_RE.finditer(text):
            for tok in IDENTISH.findall(m.group(1)):
                if looks_technical(tok):
                    emit(tok, "heading", "md")
        for m in SPAN_RE.finditer(text):
            span = m.group(1).strip()
            if span.startswith("--"):
                emit(re.split(r"[=\s]", span)[0], "flag", "md")
                continue
            toks = IDENTISH.findall(span)
            if len(toks) == 1 and looks_technical(toks[0]):
                emit(toks[0], "code_span", "md")
            elif len(toks) > 1:
                for tok in toks:
                    if looks_technical(tok) and len(tok) >= 4:
                        emit(tok, "code_span", "md")
    if ext in CFG_EXT:
        for m in CFG_RE.finditer(text):
            k = m.group(1)
            if looks_technical(k) or len(k) >= 4:
                emit(k, "config_key", "cfg")
    if text is not None:
        for m in FLAG_RE.finditer(text):
            emit(m.group(1), "flag", "cfg" if ext in CFG_EXT else ("md" if ext in DOC_EXT else "sh"))
    if mentions:
        if text is None:
            text = raw.decode("utf-8", "replace")
        for m in FLAG_RE.finditer(text):
            emit(m.group(1), "flag", lang or "text")
        for m in seed_regex(seeds).finditer(text):
            emit(m.group(1), "mention", lang or "text")
        for m in MENTION_RE.finditer(text):
            tok = m.group(0)
            if looks_technical(tok):
                emit(tok, "mention", lang or "text")
    return out


def scan_batch(batch, mentions=False, seeds=()):
    """batch: list of (path, ext, repo). -> {(term,kind): [count, nfiles, {repo:c}, {lang:c}]}"""
    agg = {}
    nf = 0
    for path, ext, repo in batch:
        rows = scan_file(path, ext, mentions, seeds)
        if rows:
            nf += 1
        for term, kind, lang in rows:
            rec = agg.get(term)
            if rec is None:
                rec = agg[term] = [0, Counter(), Counter(), Counter()]
            rec[0] += 1
            rec[1][kind] += 1
            rec[2][repo] += 1
            rec[3][lang] += 1
    return agg, len(batch), nf


# ------------------------------------------------------------------ walking
def is_skipped(name: str) -> bool:
    n = name.lower()
    return n in SKIP_DIRS or n.endswith(".egg-info") or n.endswith(".dist-info")


def iter_files(roots, repo_depth=1, max_per_repo=0):
    files = []
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        per_repo = Counter()
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            dirnames[:] = [d for d in dirnames if not is_skipped(d) and not d.startswith(".")]
            rel = Path(dirpath)
            try:
                relp = rel.relative_to(root).parts
            except ValueError:
                relp = ()
            repo = f"{root.name}/{relp[0]}" if len(relp) >= repo_depth and relp else root.name
            for name in filenames:
                ext = os.path.splitext(name)[1].lower()
                if ext not in ALL_EXT:
                    continue
                if name.endswith(".min.js") or name.endswith(".map") or name.endswith(".lock"):
                    continue
                if max_per_repo and per_repo[repo] >= max_per_repo:
                    continue
                p = str(rel / name)
                try:
                    if os.path.getsize(p) == 0:
                        continue
                except OSError:
                    continue
                per_repo[repo] += 1
                files.append((p, ext, repo))
    return files


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stats", default="")
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--batch", type=int, default=150)
    ap.add_argument("--max-per-repo", type=int, default=0)
    ap.add_argument("--min-count", type=int, default=1)
    ap.add_argument("--mentions", action="store_true")
    ap.add_argument("--seeds", default="",
                    help="file of the user's own seed terms (names, hostnames), one per line")
    a = ap.parse_args()
    seeds = load_seeds(a.seeds)

    t0 = time.perf_counter()
    files = iter_files(a.roots, max_per_repo=a.max_per_repo)
    t_walk = time.perf_counter() - t0
    print(f"walk: {len(files)} files in {t_walk:.1f}s", flush=True)

    batches = [files[i:i + a.batch] for i in range(0, len(files), a.batch)]
    total = {}
    done = 0
    nf_used = 0
    t1 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(scan_batch, b, a.mentions, seeds) for b in batches]
        for fu in as_completed(futs):
            agg, nb, nf = fu.result()
            done += nb
            nf_used += nf
            for term, rec in agg.items():
                cur = total.get(term)
                if cur is None:
                    total[term] = rec
                else:
                    cur[0] += rec[0]
                    cur[1].update(rec[1])
                    cur[2].update(rec[2])
                    cur[3].update(rec[3])
            if done % 20000 < a.batch:
                el = time.perf_counter() - t1
                print(f"  {done}/{len(files)} files  {len(total)} terms  {el:.0f}s "
                      f"({1000*el/max(1,done):.1f}s/1k files)", flush=True)
    t_parse = time.perf_counter() - t1

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    kinds = Counter()
    n = 0
    with out.open("w", encoding="utf-8") as h:
        for term, (count, kc, rc, lc) in sorted(total.items(), key=lambda kv: -kv[1][0]):
            if count < a.min_count:
                continue
            kind = kc.most_common(1)[0][0]
            kinds[kind] += 1
            n += 1
            h.write(json.dumps({"term": term, "kind": kind, "count": count,
                                "repos": dict(rc.most_common(5)),
                                "langs": dict(lc.most_common(4))}, ensure_ascii=False) + "\n")
    stats = {
        "roots": a.roots, "files_seen": len(files), "files_with_terms": nf_used,
        "walk_seconds": round(t_walk, 1), "parse_seconds": round(t_parse, 1),
        "seconds_per_1k_files": round(1000 * t_parse / max(1, len(files)), 2),
        "terms": n, "kinds": dict(kinds.most_common()),
        "workers": a.workers, "mentions": bool(a.mentions), "user_seeds": len(seeds),
    }
    print(json.dumps(stats, indent=1), flush=True)
    if a.stats:
        Path(a.stats).write_text(json.dumps(stats, indent=1) + "\n")
    print(f"INDEX done terms={n} -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
