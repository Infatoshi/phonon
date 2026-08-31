"""Subcommand: rollout. Agent loop against an OpenAI-compatible endpoint.

One tool: bash, run inside the persona dir. No network: proxy envs stripped,
curl/wget/ssh/git blocked by a PATH shim. Full trajectory saved as JSONL.
"""

import json
import os
import re
import stat
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

TOOL_TIMEOUT = 15
TOOL_MAX_BYTES = 4096
BLOCKED = ["curl", "wget", "ssh", "git", "scp", "sftp"]
FENCE = re.compile(r"```json\s*(.*?)```", re.DOTALL)

SYSTEM_PROMPT = """You are a vocabulary miner for a personal dictation app.
The current working directory is one developer's machine: cloned repositories
under repos/ and their chat logs with coding agents under logs/ (claude.txt,
codex.txt; one line per user-authored message).

Task: mine the personal vocabulary a speech-to-text system would need for this
user. Find project codenames, machine names, people and handles, model names,
tool names, and domain jargon that the user actually types. Prefer terms that
look invented or long-tail over common English. When the logs show a garbled
spoken version of a term (for example "the neb rl" for "deneb-rl"), record it
under spoken_forms of the canonical term.

Use the bash tool to explore (ls, grep, sort, uniq, awk, head...). Commands
time out after 15 s and output is truncated to 4 KB, so aggregate instead of
dumping whole files. There is no network access.

When you are done, end your final message with a single JSON array in a
```json fence and nothing after it. Each element:
{"term": str, "count": int, "kind": str, "spoken_forms": [str], "evidence": str}
kind is one of: project, machine, person, model, tool, library, jargon, file, other.
count is how often the term appears in the logs. evidence is one short quote or
path showing the term in use. Aim for the 30-60 best terms, ranked best first."""

TOOLS = [{
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Run a bash command in the persona machine directory. "
                       "15 s timeout, output truncated to 4 KB, no network.",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string", "description": "The command to run."}},
            "required": ["command"],
        },
    },
}]


def make_shim(out: Path) -> Path:
    shim = out / ".shim"
    shim.mkdir(parents=True, exist_ok=True)
    for name in BLOCKED:
        p = shim / name
        p.write_text(f"#!/bin/sh\necho '{name}: network access is disabled in persona-gym' >&2\nexit 1\n")
        p.chmod(p.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return shim


def tool_env(shim: Path) -> dict:
    env = {k: v for k, v in os.environ.items() if "proxy" not in k.lower()}
    env["PATH"] = f"{shim}:{env.get('PATH', '/usr/bin:/bin')}"
    return env


def run_bash(cmd: str, cwd: Path, env: dict) -> str:
    try:
        r = subprocess.run(["/bin/bash", "-c", cmd], cwd=cwd, env=env, check=False,
                           capture_output=True, timeout=TOOL_TIMEOUT)
    except subprocess.TimeoutExpired:
        return f"error: command timed out after {TOOL_TIMEOUT}s"
    out = (r.stdout + r.stderr).decode("utf-8", errors="replace")
    if len(out) > TOOL_MAX_BYTES:
        out = out[:TOOL_MAX_BYTES] + f"\n[truncated at {TOOL_MAX_BYTES} bytes]"
    return f"exit {r.returncode}\n{out}" if out.strip() else f"exit {r.returncode} (no output)"


def chat(endpoint: str, model: str, messages: list, timeout: int = 900,
         tool_choice: str = "auto") -> dict:
    body = json.dumps({"model": model, "messages": messages,
                       "tools": TOOLS, "tool_choice": tool_choice}).encode()
    url = endpoint.rstrip("/")
    if not url.endswith("/chat/completions"):
        url += "/v1/chat/completions" if not url.endswith("/v1") else "/chat/completions"
    req = urllib.request.Request(url, data=body, headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', 'none')}",
    })
    last = None
    for attempt in range(2):  # retry once on transport errors
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as e:
            if isinstance(e, urllib.error.HTTPError) and e.code < 500:
                raise
            last = e
            if attempt == 0:
                print(f"[rollout] transport error, retrying: {e}", file=sys.stderr)
                time.sleep(2)
    raise RuntimeError(f"endpoint failed after retry: {last}")


def parse_final(content: str):
    m = FENCE.findall(content or "")
    if not m:
        return None, "no json fence"
    try:
        ans = json.loads(m[-1], strict=False)
    except json.JSONDecodeError as e:
        return None, f"json error: {e}"
    if not isinstance(ans, list):
        return None, "not a list"
    return ans, "ok"


def run_one(persona_dir: Path, endpoint: str, model: str, rdir: Path,
            max_turns: int, shim: Path) -> dict:
    rdir.mkdir(parents=True, exist_ok=True)
    traj_path = rdir / "trajectory.jsonl"
    traj_path.write_text("")

    def log(msg):
        with open(traj_path, "a") as traj:
            traj.write(json.dumps(msg, ensure_ascii=False) + "\n")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "The machine root is the current working directory. Begin."},
    ]
    for m in messages:
        log(m)
    env = tool_env(shim)
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    answer, status, turns = None, "max_turns", 0
    nudged = False
    finalize_at = max(max_turns - 2, 1)
    for turn in range(max_turns):
        turns = turn + 1
        final_call = turn >= finalize_at
        if final_call and messages[-1].get("role") != "user":
            wrap = {"role": "user", "content":
                    "Stop exploring. Reply now with your final answer: a single "
                    "JSON array in a ```json fence and nothing after it."}
            messages.append(wrap)
            log(wrap)
        resp = chat(endpoint, model, messages,
                    tool_choice="none" if final_call else "auto")
        u = resp.get("usage") or {}
        for k in usage:
            usage[k] += u.get(k) or 0
        msg = resp["choices"][0]["message"]
        # Keep reasoning out of the history: replaying it re-renders thousands
        # of thinking tokens into every later prompt and overflows the context.
        clean = {k: msg.get(k) for k in ("role", "content", "tool_calls")
                 if msg.get(k) is not None}
        messages.append(clean)
        log(msg)
        calls = msg.get("tool_calls") or []
        if calls:
            for c in calls:
                try:
                    cmd = json.loads(c["function"]["arguments"]).get("command", "")
                except (json.JSONDecodeError, KeyError):
                    cmd = ""
                out = run_bash(cmd, persona_dir, env) if cmd else "error: no command given"
                tool_msg = {"role": "tool", "tool_call_id": c.get("id", ""), "content": out}
                messages.append(tool_msg)
                log(tool_msg)
            continue
        answer, status = parse_final(msg.get("content"))
        if answer is None and not nudged and turn + 1 < max_turns:
            nudged = True
            nudge = {"role": "user", "content":
                     "Reply with your final answer now: a single JSON array in a ```json fence."}
            messages.append(nudge)
            log(nudge)
            continue
        break
    meta = {"persona": persona_dir.name, "model": model, "turns": turns,
            "status": status, "usage": usage, "n_terms": len(answer) if answer else 0}
    with open(rdir / "meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    with open(rdir / "answer.json", "w") as f:
        json.dump(answer, f, indent=1, ensure_ascii=False)
    return meta


def run(personas: Path, endpoint: str, model: str, out: Path,
        n_per: int = 1, max_turns: int = 24) -> None:
    personas = personas.resolve()
    out = out.resolve()
    shim = make_shim(out)
    pdirs = sorted(d for d in personas.iterdir()
                   if d.is_dir() and d.name not in ("meta", ".cache", ".shim"))
    if not pdirs:
        raise SystemExit(f"no personas found in {personas}")
    for pdir in pdirs:
        for r in range(n_per):
            rdir = out / pdir.name / f"r{r}"
            if (rdir / "meta.json").exists():
                print(f"[rollout] {pdir.name} r{r}: skip (done)", file=sys.stderr)
                continue
            t0 = time.time()
            meta = run_one(pdir, endpoint, model, rdir, max_turns, shim)
            print(f"[rollout] {pdir.name} r{r}: {meta['status']}, {meta['turns']} turns, "
                  f"{meta['n_terms']} terms, {meta['usage']['prompt_tokens']}+"
                  f"{meta['usage']['completion_tokens']} tok, {time.time() - t0:.0f}s",
                  file=sys.stderr)
