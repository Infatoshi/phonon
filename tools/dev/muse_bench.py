"""Benchmark Meta Muse Voice Transcribe on the local Phonon corpus.

Developer tooling, not part of the app: this script UPLOADS your recordings to
Meta's cloud API. Phonon itself never sends audio or text off the Mac.

Sends each speech recording to POST /v1/asr/transcribe twice: plain, and with
`keywords` = the Phonon dictionary (canonical phrases). Writes muse_results.json
and prints a side-by-side of Parakeet raw / Phonon final / Muse / Muse+keywords
for recordings that contain technical tokens, plus WER of each system against
the Phonon final transcript (agreement, not ground truth).

Requires MODEL_API_KEY (Meta Model API) in the environment. Audio never leaves
the machine without it.
"""
import glob
import json
import os
import re
import sys
import time
import urllib.request
import uuid

API = "https://api.meta.ai/v1/asr/transcribe"
SUPPORT = os.path.expanduser("~/Library/Application Support/Phonon")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "muse_results.json")
WORDS = set(w.strip().lower() for w in open("/usr/share/dict/words"))
MAX_KEYWORDS = int(os.environ.get("MUSE_MAX_KEYWORDS", "50"))  # >50 -> empty transcript, 200 -> 503


def multipart(fields: dict, file_field: str, filename: str, data: bytes) -> tuple[bytes, str]:
    b = f"----phonon{uuid.uuid4().hex}"
    body = bytearray()
    for k, (v, ctype) in fields.items():
        body += (f"--{b}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n"
                 f"Content-Type: {ctype}\r\n\r\n{v}\r\n").encode()
    body += (f"--{b}\r\nContent-Disposition: form-data; name=\"{file_field}\"; "
             f"filename=\"{filename}\"\r\nContent-Type: audio/wav\r\n\r\n").encode()
    body += data + f"\r\n--{b}--\r\n".encode()
    return bytes(body), f"multipart/form-data; boundary={b}"


def transcribe(key: str, wav: str, keywords: list[str] | None) -> dict:
    req = {"mode": "PUSH_TO_TALK", "model": "muse-voice-transcribe-1.0", "audioEncoding": "WAV"}
    if keywords:
        req["keywords"] = keywords
    body, ctype = multipart({"request": (json.dumps(req), "application/json")},
                            "audio", "audio.wav", open(wav, "rb").read())
    r = urllib.request.Request(API, data=body, headers={
        "Authorization": f"Bearer {key}", "Content-Type": ctype})
    for attempt in range(4):
        try:
            t0 = time.time()
            with urllib.request.urlopen(r, timeout=120) as resp:
                out = json.loads(resp.read())
            out["_latency_s"] = round(time.time() - t0, 2)
            return out
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="replace")[:300]
            if e.code == 429 and attempt < 3:
                time.sleep(10 * (attempt + 1))
                continue
            return {"error": f"HTTP {e.code}: {msg}"}
    return {"error": "retries exhausted"}


def norm(s: str) -> list[str]:
    s = s.lower().replace("’", "'")
    return re.findall(r"[a-z0-9][a-z0-9'_.\-]*", s)


def wer(ref: list[str], hyp: list[str]) -> float:
    d = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, h in enumerate(hyp, 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r != h))
            prev, d[j] = d[j], cur
    return d[-1] / max(len(ref), 1)


def tech_tokens(text: str) -> list[str]:
    toks = [t.strip(".,;:!?()\"'").lower() for t in text.split()]
    out = set()
    for t in toks:
        if len(t) < 3 or "'" in t or t in WORDS:
            continue
        if t.endswith("s") and t[:-1] in WORDS:
            continue
        if t.endswith("ing") and t[:-3] in WORDS:
            continue
        if re.search(r"[a-z]", t):
            out.add(t)
    return sorted(out)


def main():
    key = os.environ.get("MODEL_API_KEY") or os.environ.get("META_MODEL_API_KEY")
    if not key:
        sys.exit("MODEL_API_KEY not set; get one at https://dev.meta.ai/")
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    d = json.load(open(os.path.join(SUPPORT, "dictionary.json")))
    ents = sorted((e for e in d["entries"] if e.get("phrase")),
                  key=lambda e: (-(e.get("usage_count") or 0), not e.get("starred"), e["phrase"]))
    keywords = list(dict.fromkeys(e["phrase"] for e in ents))[:MAX_KEYWORDS]
    recs = []
    for p in glob.glob(os.path.join(SUPPORT, "Corpus", "*", "*.json")):
        j = json.load(open(p))
        if not j.get("speech_detected"):
            continue
        wav = os.path.join(os.path.dirname(p), j.get("audio_file", "audio.wav"))
        if os.path.exists(wav):
            recs.append((j, wav))
    recs.sort(key=lambda r: -len(tech_tokens(r[0]["final_transcript"])))
    if limit:
        recs = recs[:limit]
    done = json.load(open(OUT)) if os.path.exists(OUT) else {}
    minutes = sum(j.get("audio_duration_ms") or 0 for j, _ in recs) / 60000
    print(f"[muse] {len(recs)} recordings, {minutes:.1f} min, {len(keywords)} keywords, "
          f"~${minutes * 2 * 0.003:.2f}", file=sys.stderr)
    for j, wav in recs:
        rid = j["id"]
        prev = done.get(rid, {})
        plain = prev.get("plain") if "error" not in prev.get("plain", {"error": 1}) else transcribe(key, wav, None)
        biased = prev.get("biased") if "error" not in prev.get("biased", {"error": 1}) else transcribe(key, wav, keywords)
        done[rid] = {"raw": j["raw_transcript"], "final": j["final_transcript"],
                     "duration_ms": j.get("audio_duration_ms"),
                     "plain": plain, "biased": biased}
        json.dump(done, open(OUT, "w"), indent=1)
        print(f"[muse] {rid} plain={plain.get('_latency_s')}s biased={biased.get('_latency_s')}s "
              f"{plain.get('error', '')}", file=sys.stderr)

    rows = []
    agg = {"raw": [], "plain": [], "biased": []}
    for rid, r in done.items():
        if "error" in r["plain"] or "error" in r["biased"]:
            continue
        ref = norm(r["final"])
        for k, text in (("raw", r["raw"]), ("plain", r["plain"]["transcript"]),
                        ("biased", r["biased"]["transcript"])):
            agg[k].append((wer(ref, norm(text)), len(ref)))
        tech = tech_tokens(r["final"])
        if tech:
            rows.append((rid, tech, r["raw"], r["final"], r["plain"]["transcript"],
                         r["biased"]["transcript"]))
    print("\nWER vs Phonon final (token-weighted):")
    for k, v in agg.items():
        if v:
            n = sum(w for _, w in v)
            print(f"  {k:7s} {sum(e * w for e, w in v) / n:.3f}  ({len(v)} recs)")
    print("\nTechnical recordings, side by side:")
    for rid, tech, raw, fin, plain, biased in rows:
        print(f"\n== {rid}  tech={tech[:8]}")
        print(f"  parakeet : {raw[:220]}")
        print(f"  phonon   : {fin[:220]}")
        print(f"  muse     : {plain[:220]}")
        print(f"  muse+kw  : {biased[:220]}")


if __name__ == "__main__":
    main()
