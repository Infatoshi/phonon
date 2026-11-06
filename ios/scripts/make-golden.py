#!/usr/bin/env python3
"""Runs the Mac app's own refiner helpers (sidecar/polish_server.py, extracted with ast so no
model runtime is imported) on fixed cases and writes the answers the Swift port must match."""
import ast, json, re, sys
from difflib import SequenceMatcher
src = open(sys.argv[1]).read()
tree = ast.parse(src)
want_fn = {"_norm", "restore_terms", "guard_output", "clean_output", "refiner_system_prompt", "refiner_token_budget"}
want_const = {"REFINER_SYSTEM_PROMPT", "REFINER_VOCAB_PREFIX", "GUARD_TOK_MULT", "GUARD_TOK_ADD", "GUARD_MAX_RATIO",
              "GUARD_MIN_RATIO", "GUARD_MIN_WORDS", "OUTPUT_TOKEN_CEILING", "LEAKED_MARKER_PATTERN", "TERM_SHAPE", "WORD_PARTS"}
nodes = []
for n in tree.body:
    if isinstance(n, ast.FunctionDef) and n.name in want_fn: nodes.append(n)
    if isinstance(n, ast.Assign) and any(getattr(t, "id", None) in want_const for t in n.targets): nodes.append(n)
ns = {"re": re, "SequenceMatcher": SequenceMatcher}
exec(compile(ast.Module(body=nodes, type_ignores=[]), "polish_server_subset", "exec"), ns)
restore = [
    ("we use native mtp for the decode", "We use native MTK for the decode.", ["MTP", "MLP"]),
    ("the mlp layer is slow", "The MMLU-Pro layer is slow.", ["MLP"]),
    ("ship it to the gpu tonight", "Ship it to the GPU tonight.", ["GPU"]),
    ("run vllm on the box", "Run VLLM-2 on the box.", ["vLLM"]),
    ("hello there friend", "Hello there, friend.", []),
    ("check the cuda kernel and the mtp head", "Check the CUDA kernel and the MTX-9 head.", ["CUDA", "MTP"]),
]
guards = [("one two three four five six seven eight nine ten", "one"),
          ("short one", "a b c d e f g h i j k l m n"),
          ("one two three four five six seven eight nine ten", "one two three four five six seven eight nine ten eleven"),
          ("", "anything"),
          ("a b c d e f g h", "a")]
cleans = ["<think></think>Hello there.", "  <|turn|>Text<channel|> ", "Plain"]
out = {
    "systemPrompt": ns["REFINER_SYSTEM_PROMPT"],
    "systems": [{"vocab": v, "mode": m, "out": ns["refiner_system_prompt"](v, m)} for v, m in
                [([], ""), (["CUDA", "vLLM"], ""), ([], "Writing mode: casual (a chat message)."),
                 (["MTP"], "Writing mode: formal (an email or document).")]],
    "budgets": [{"tokens": t, "out": ns["refiner_token_budget"](t)} for t in [0, 1, 10, 100, 1500, 5000]],
    "guards": [{"spoken": s, "output": o, "out": list(ns["guard_output"](s, o))} for s, o in guards],
    "cleans": [{"in": c, "out": ns["clean_output"](c)} for c in cleans],
    "restores": [{"spoken": s, "output": o, "vocab": v, "out": ns["restore_terms"](s, o, v)} for s, o, v in restore],
}
json.dump(out, open(sys.argv[2], "w"), indent=1)
print("golden cases:", sum(len(v) for v in out.values() if isinstance(v, list)))
