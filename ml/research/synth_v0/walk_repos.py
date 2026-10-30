"""Offline repo walk: lexicon + topic seeds. CPU only. No /data/phonon_personal."""
from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import DATA_ROOT, LEXICON_PATH, REPO_ROOTS, TOPICS_PATH  # noqa: E402

MAX_FILE_BYTES = 2 * 1024 * 1024
MAX_FILES_PER_SOURCE = 5
MAX_TOPICS = 8000
WORKERS = 16

SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "target",
    "build",
    "dist",
    "__pycache__",
    ".cache",
    "wandb",
    "datasets",
    "data",
    "runs",
    "checkpoints",
    ".mypy_cache",
    ".ruff_cache",
    "site-packages",
    "weights",
    "ckpt",
    "wavs",
    "shards",
    ".tox",
    ".pytest_cache",
    ".next",
    "coverage",
    "htmlcov",
    "vendor",
    "cmakefiles",
    "cmake-build-debug",
    "cmake-build-release",
    ".idea",
    ".vscode",
    "tts_wavs",
    "chunks",
    "adapters",
    ".uv",
    ".cargo",
    ".npm",
    ".yarn",
    ".turbo",
    "artifacts",
    "tmp",
    "temp",
    ".eggs",
    "wheels",
    ".huggingface",
    "hub",
    "parquet",
    "audio",
    "videos",
    "images",
    ".ipynb_checkpoints",
    "__pypackages__",
    "deriveddata",
    "xcuserdata",
    "egg-info",
    "third_party",
    "llvm-project",
    "logs",
    "output",
    "out",
    ".gradle",
    ".swiftpm",
    ".bundle",
    "testdata",
    "fixtures",
    "golden",
    ".pre-commit",
}

ALLOW_EXT = {
    ".py",
    ".pyi",
    ".rs",
    ".go",
    ".c",
    ".h",
    ".cpp",
    ".cc",
    ".cxx",
    ".cu",
    ".cuh",
    ".hpp",
    ".hxx",
    ".hh",
    ".swift",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".mjs",
    ".cjs",
    ".md",
    ".txt",
    ".toml",
    ".yaml",
    ".yml",
    ".json",
    ".sh",
    ".bash",
    ".zsh",
    ".cmake",
    ".mk",
    ".proto",
    ".metal",
    ".mm",
    ".java",
    ".kt",
    ".rb",
    ".lua",
    ".jl",
    ".nix",
    ".sql",
    ".rst",
    ".cfg",
    ".ini",
    ".conf",
    ".service",
    ".r",
    ".svelte",
    ".vue",
}

DOC_NAMES = {
    "readme.md",
    "devlog.md",
    "spec.md",
    "agents.md",
    "goal.md",
    "product.md",
    "labeling.md",
    "claude.md",
    "contributing.md",
    "changelog.md",
}

CAMEL = re.compile(r"\b[a-z]+(?:[A-Z][a-z0-9]+){1,}\b")
PASCAL = re.compile(r"\b[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+){1,}\b")
SNAKE = re.compile(r"\b[A-Za-z][A-Za-z0-9]*_[A-Za-z0-9_]{1,}\b")
ACRONYM = re.compile(r"\b[A-Z]{2,8}\b")
FLAG = re.compile(r"(?:^|[\s=`\"'(])(--[A-Za-z][A-Za-z0-9-]{1,40})\b")
SHORT_FLAG = re.compile(r"(?:^|[\s=`\"'(])(-[A-Za-z])\b")
DEF_NAME = re.compile(
    r"\b(?:def|class|fn|func|function|struct|enum|type|interface|trait|impl|mod|pub\s+fn|"
    r"async\s+fn|pub\s+struct|pub\s+enum)\s+([A-Za-z_][A-Za-z0-9_]*)"
)
PY_IMPORT = re.compile(r"^\s*(?:from|import)\s+([A-Za-z_][A-Za-z0-9_\.]*)", re.M)
RUST_USE = re.compile(r"^\s*use\s+([A-Za-z_][A-Za-z0-9_:]*)", re.M)
GO_IMPORT = re.compile(r'^\s*"([A-Za-z0-9_./-]+)"\s*$', re.M)
SHELL_CMD = re.compile(
    r"\b(uv|pytest|ruff|nvidia-smi|nvcc|ncu|nsys|cmake|ninja|git|rsync|ssh|tmux|"
    r"docker|kubectl|huggingface-cli|overnight-compute|hyperfine|croc|bun|cargo|"
    r"rustc|go|node|npm|pnpm|make|ffmpeg|sox|jq|rg|fd|scp|rsync)\b"
)
HOSTISH = re.compile(r"\b(gpubox|gamer|macbook|blackwell|3090|m4\s*max)\b", re.I)
PERSONISH = re.compile(r"\b(Elliot|Infatoshi|Grok|Opus|Codex|Claude|Fable)\b")
IDENT_TOKEN = re.compile(r"\b[A-Za-z][A-Za-z0-9_\-]{1,80}\b")
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(])")
COMMENT_HASH = re.compile(r"(?:^|\s)#(?!\!|#)\s*(.+)$")
COMMENT_SLASH = re.compile(r"//+\s*(.+)$")
BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.S)
DOCSTRING = re.compile(r'^\s*(?:r|u|f)?["\']{3}(.+?)["\']{3}', re.S | re.I)
LICENSE_LINE = re.compile(r"copyright|spdx-license|licensed under|all rights reserved", re.I)
BOILER = re.compile(r"^(todo|fixme|xxx|note|hack|lint|noqa|type:\s*ignore)\b", re.I)

COMMON_ENGLISH = frozenset(
    """
    a about above after again against all almost also always am among an and any
    anyone anything are around as at away back be because been before being below
    between both but by came can cannot could did do does doing done down during
    each either else enough even ever every everyone everything few first for from
    get go going gone got had has have having he her here hers herself him himself
    his how however i if in into is it its itself just keep knew know known last
    later least less let like likely little long look looking lot made make making
    many may maybe me might more most much must my myself never new next no nor
    not nothing now of off often on once one only onto or other others our ours
    ourselves out over own part per perhaps please put rather really right said
    same saw say saying see seen several she should since so some someone something
    still such than that the their them themselves then there these they this those
    though through to too under until up upon us used using very want was way we
    well went were what when where whether which while who whom whose why will with
    within without would yes yet you your yours yourself yourselves
    able actually already also always around away better big bit both called come
    coming days doing early end even far feel felt few find found full given going
    good got great hard high however instead keep kind last left let little long
    look looking lot low made make making many maybe mean means might need needed
    needs never next often old once only open order part parts place possible put
    quite rather really right run said same saw say saying see seen set several
    show shown small something still sure take taken takes taking thing things
    think thought three time times today together took try trying turn two until
    use used uses using want wanted wants way ways week went whether whole work
    worked working works year years yes yet
    add added adding adds also change changed changes changing check checked
    create created creates creating current default different does doing done
    error errors example examples extra fail failed fails file files final find
    found function functions help info information input inputs issue issues item
    items line lines list lists load loaded loading local log logs main message
    messages method methods mode name names need needed needs note notes number
    numbers option options output outputs path paths point points process read
    reading result results return returned returns run running runs start started
    starting starts state states status step steps stop stopped support supported
    test tests text true type types update updated updates value values version
    versions write writes writing
    above according across after against along already although always among around
    because before behind below besides between beyond both despite during except
    following given including inside instead near outside since through throughout
    toward towards under unless until upon versus via whether while within without
    about again almost also always another any anybody anything anywhere around
    away back both each either else enough ever every everybody everyone everything
    everywhere few first half last later least less many more most much next none
    nothing often once only other others same several some somebody someone
    something sometimes somewhere still such that then there these this those
    together too twice well yet
    the a an and or but if then else when where why how what who whom whose which
    this that these those i you he she it we they me him her us them my your his
    her its our their mine yours hers ours theirs
    is am are was were be been being do does did doing done have has had having
    can could should would may might must will shall need dare used
    not no nor none never nothing nobody nowhere
    yes ok okay sure please thanks thank sorry wait uh um like actually basically
    anyway anyways kinda sort maybe perhaps probably definitely
    zero one two three four five six seven eight nine ten eleven twelve hundred
    thousand million billion
    monday tuesday wednesday thursday friday saturday sunday today tomorrow yesterday
    january february march april may june july august september october november december
    """.split()
)

CODE_KEYWORDS = frozenset(
    """
    int float void char bool true false null none nil self this super class def fn
    func function return yield async await import from export const let var pub
    priv private public static final abstract virtual override impl trait struct
    enum type interface module package namespace using include define ifdef ifndef
    endif pragma template typename sizeof alignas alignof constexpr noexcept
    switch case break continue for while do if else elif try catch except finally
    throw raise with as in is not and or pass lambda assert del global nonlocal
    match where crate mod use fn let mut ref dyn box impl where async await
    printf printk println eprintln unwrap expect clone copy drop into from default
    """.split()
)

GENERIC_SEEDS = {
    "blackwell": "machine",
    "rtx": "product",
}


def load_machine_seeds() -> dict[str, str]:
    """Generic seeds plus the user's own names and hostnames from seeds.json beside
    the data ({term: kind}, e.g. {"mybox": "machine"}). Nothing personal is
    hardcoded here."""
    seeds = dict(GENERIC_SEEDS)
    path = DATA_ROOT / "seeds.json"
    if path.is_file():
        user = json.loads(path.read_text(encoding="utf-8"))
        seeds.update({term.lower(): kind for term, kind in user.items()})
    return seeds


MACHINE_SEEDS = load_machine_seeds()


def is_skipped_dir(name: str) -> bool:
    n = name.lower()
    if n in SKIP_DIRS:
        return True
    if n.endswith(".egg-info") or n.endswith(".dist-info"):
        return True
    return False


def iter_files(roots: list[Path]) -> list[str]:
    out: list[str] = []
    for root in roots:
        if not root.exists():
            continue
        if root.is_file():
            out.append(str(root))
            continue
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            dirnames[:] = [d for d in dirnames if not is_skipped_dir(d) and not d.startswith(".")]
            rel = Path(dirpath)
            parts_l = {p.lower() for p in rel.parts}
            if parts_l & SKIP_DIRS:
                dirnames[:] = []
                continue
            for name in filenames:
                path = rel / name
                ext = path.suffix.lower()
                if ext not in ALLOW_EXT:
                    continue
                if name.endswith(".min.js") or name.endswith(".map"):
                    continue
                try:
                    st = path.stat()
                except OSError:
                    continue
                if not st.st_size or st.st_size > MAX_FILE_BYTES:
                    continue
                out.append(str(path))
    return out


def classify_term(term: str, hint: str) -> str:
    if hint:
        return hint
    if term.startswith("--") or (term.startswith("-") and len(term) <= 3):
        return "flag"
    if re.fullmatch(r"[A-Z]{2,8}", term):
        return "acronym"
    if "/" in term or term.endswith((".py", ".rs", ".cu", ".cuh", ".go", ".ts", ".md", ".toml", ".json")):
        return "file"
    if "_" in term or re.search(r"[a-z][A-Z]", term):
        return "identifier"
    low = term.lower()
    if low in MACHINE_SEEDS:
        return MACHINE_SEEDS[low]
    return "term"


def mangle_score(term: str) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []
    low = term.lower().strip("-")
    alpha = re.sub(r"[^a-z]", "", low)
    if low in COMMON_ENGLISH or alpha in COMMON_ENGLISH or low in CODE_KEYWORDS:
        score -= 3.0
        reasons.append("common_english")
    else:
        score += 2.0
        reasons.append("not_common_english")
    if re.search(r"\d", term):
        score += 2.0
        reasons.append("digits")
    if "_" in term:
        score += 2.0
        reasons.append("underscores")
    if re.search(r"[a-z][A-Z]", term):
        score += 3.0
        reasons.append("camelCase")
    if re.fullmatch(r"[A-Z]{2,8}", term):
        score += 2.0
        reasons.append("acronym")
    if "-" in term and not term.startswith("-"):
        score += 1.0
        reasons.append("hyphenated")
    if term.startswith("--"):
        score += 1.5
        reasons.append("cli_flag")
    if len(alpha) <= 1:
        score -= 4.0
    return score, reasons


def valid_term(term: str) -> bool:
    if not term or len(term) > 80:
        return False
    if term.startswith("--"):
        return 2 <= len(term) <= 40
    if term.startswith("-") and len(term) == 2:
        return term[1].isalpha()
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_\-\.]{0,79}", term) is None:
        return False
    alpha = re.sub(r"[^A-Za-z]", "", term)
    if len(alpha) < 2:
        return False
    low = term.lower()
    if low in COMMON_ENGLISH or low in CODE_KEYWORDS:
        return False
    if term.isdigit():
        return False
    return True


def add_term(bucket: dict, term: str, kind: str, path: str) -> None:
    term = term.strip().strip("`\"'.,;:()[]{}<>")
    if not valid_term(term):
        return
    kind = classify_term(term, kind)
    rec = bucket.get(term)
    if rec is None:
        rec = {"term": term, "kind": kind, "count": 0, "files": Counter()}
        bucket[term] = rec
    rec["count"] += 1
    rec["files"][path] += 1
    if rec["kind"] == "term" and kind != "term":
        rec["kind"] = kind


def extract_manifest_names(text: str, path: str, bucket: dict) -> None:
    name = Path(path).name.lower()
    if name in {"package.json", "composer.json"}:
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return
        for key in ("name",):
            val = data.get(key)
            if isinstance(val, str):
                add_term(bucket, val.split("/")[-1], "product", path)
        for section in ("dependencies", "devDependencies", "peerDependencies"):
            deps = data.get(section) or {}
            if isinstance(deps, dict):
                for dep in list(deps)[:80]:
                    add_term(bucket, str(dep).split("/")[-1], "library", path)
        return
    if name in {"pyproject.toml", "cargo.toml", "go.mod"}:
        for m in re.finditer(r'(?m)^\s*(?:name\s*=\s*"([^"]+)"|module\s+(\S+))', text):
            val = m.group(1) or m.group(2)
            add_term(bucket, val.split("/")[-1], "product", path)
        for m in re.finditer(r'(?m)^\s*"?([A-Za-z0-9_.\-]+)"?\s*=\s*"', text):
            add_term(bucket, m.group(1), "library", path)


def extract_comments_and_docs(text: str, path: str, ext: str) -> list[tuple[str, str]]:
    sentences: list[tuple[str, str]] = []
    name = Path(path).name.lower()
    is_doc = name in DOC_NAMES or name.startswith("readme")
    if is_doc or ext in {".md", ".rst", ".txt"}:
        body = text
        kind = "markdown" if ext in {".md", ".rst"} else "text"
        for raw in SENT_SPLIT.split(body.replace("\n", " ")):
            s = " ".join(raw.split())
            if 40 <= len(s) <= 400 and not LICENSE_LINE.search(s) and not s.startswith("#"):
                s = s.lstrip("#> -")
                if s:
                    sentences.append((s, kind))
        return sentences[:80]
    comments: list[str] = []
    for line in text.splitlines():
        if LICENSE_LINE.search(line):
            continue
        m = COMMENT_HASH.search(line) if ext in {".py", ".sh", ".bash", ".zsh", ".toml", ".yaml", ".yml", ".rb", ".r", ".jl"} else None
        if m:
            comments.append(m.group(1).strip())
            continue
        m = COMMENT_SLASH.search(line)
        if m:
            comments.append(m.group(1).strip())
    for block in BLOCK_COMMENT.findall(text):
        comments.append(" ".join(block.strip("/* ").split()))
    if ext == ".py":
        for ds in DOCSTRING.findall(text[:8000]):
            comments.append(" ".join(ds.split())[:400])
    for c in comments:
        if not c or BOILER.match(c) or LICENSE_LINE.search(c):
            continue
        if 40 <= len(c) <= 400:
            sentences.append((c, "comment"))
    return sentences[:40]


def process_file(path_str: str) -> tuple[dict, list]:
    path = Path(path_str)
    bucket: dict = {}
    topics: list = []
    try:
        raw = path.read_bytes()
    except OSError:
        return bucket, topics
    if b"\x00" in raw[:4096]:
        return bucket, topics
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text = raw.decode("latin-1")
        except UnicodeDecodeError:
            return bucket, topics
    if text.count("\n") < 2 and len(text) > 4000:
        return bucket, topics
    ext = path.suffix.lower()
    rel = path_str
    add_term(bucket, path.name, "file", rel)
    if path.suffix:
        stem = path.stem
        if valid_term(stem):
            add_term(bucket, stem, "module" if ext in {".py", ".rs", ".go", ".ts"} else "file", rel)
    parent = path.parent.name
    if parent and valid_term(parent) and parent.lower() not in SKIP_DIRS:
        add_term(bucket, parent, "module", rel)

    extract_manifest_names(text, rel, bucket)

    for rx, kind in (
        (CAMEL, "identifier"),
        (PASCAL, "identifier"),
        (SNAKE, "identifier"),
        (ACRONYM, "acronym"),
        (FLAG, "flag"),
    ):
        for m in rx.finditer(text):
            add_term(bucket, m.group(0) if rx is not FLAG else m.group(1), kind, rel)
            if len(bucket) > 4000:
                break

    for m in DEF_NAME.finditer(text):
        add_term(bucket, m.group(1), "identifier", rel)
    for m in PY_IMPORT.finditer(text):
        root = m.group(1).split(".")[0]
        add_term(bucket, root, "library", rel)
    for m in RUST_USE.finditer(text):
        root = m.group(1).split("::")[0]
        add_term(bucket, root, "library", rel)
    for m in SHELL_CMD.finditer(text):
        add_term(bucket, m.group(1), "cli_tool", rel)
    for m in HOSTISH.finditer(text):
        add_term(bucket, m.group(1), "machine", rel)
    for m in PERSONISH.finditer(text):
        add_term(bucket, m.group(1), "person", rel)

    for sent, kind in extract_comments_and_docs(text, rel, ext):
        topics.append({"text": sent, "source_file": rel, "kind": kind})
    return bucket, topics


def merge_bucket(dst: dict, src: dict) -> None:
    for term, rec in src.items():
        cur = dst.get(term)
        if cur is None:
            dst[term] = {
                "term": rec["term"],
                "kind": rec["kind"],
                "count": rec["count"],
                "files": Counter(rec["files"]),
            }
            continue
        cur["count"] += rec["count"]
        cur["files"].update(rec["files"])
        if cur["kind"] == "term" and rec["kind"] != "term":
            cur["kind"] = rec["kind"]


def main() -> int:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    files = iter_files(REPO_ROOTS)
    print(f"walk files={len(files)} roots={[str(r) for r in REPO_ROOTS if r.exists()]}", flush=True)
    bucket: dict = {}
    topics: list = []
    seen_topic = set()
    done = 0
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futs = {pool.submit(process_file, p): p for p in files}
        for fut in as_completed(futs):
            done += 1
            try:
                part, part_topics = fut.result()
            except Exception as exc:
                print(f"fail {futs[fut]}: {type(exc).__name__}: {exc}", flush=True)
                continue
            merge_bucket(bucket, part)
            for t in part_topics:
                key = t["text"][:200]
                if key in seen_topic:
                    continue
                seen_topic.add(key)
                topics.append(t)
            if done % 500 == 0 or done == len(files):
                print(
                    f"processed {done}/{len(files)} terms={len(bucket)} topics={len(topics)}",
                    flush=True,
                )
    for seed, kind in MACHINE_SEEDS.items():
        if seed not in bucket:
            bucket[seed] = {"term": seed, "kind": kind, "count": 1, "files": Counter({"seed": 1})}
        else:
            if bucket[seed]["kind"] == "term":
                bucket[seed]["kind"] = kind

    rows = []
    for rec in bucket.values():
        score, reasons = mangle_score(rec["term"])
        files_top = [f for f, _c in rec["files"].most_common(MAX_FILES_PER_SOURCE)]
        rows.append(
            {
                "term": rec["term"],
                "kind": rec["kind"],
                "count": int(rec["count"]),
                "files": files_top,
                "source_file": files_top[0] if files_top else "",
                "mangle_score": round(score, 3),
                "mangle_reasons": reasons,
            }
        )
    rows.sort(key=lambda r: (-r["mangle_score"], -r["count"], r["term"]))
    with LEXICON_PATH.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    topics = topics[:MAX_TOPICS]
    with TOPICS_PATH.open("w", encoding="utf-8") as handle:
        for i, t in enumerate(topics):
            t["id"] = f"topic_{i:05d}"
            handle.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(
        f"wrote {LEXICON_PATH} n={len(rows)} and {TOPICS_PATH} n={len(topics)}",
        flush=True,
    )
    print("top mangle:", [r["term"] for r in rows[:25]], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
