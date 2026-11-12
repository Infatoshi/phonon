#!/usr/bin/env python3
"""A synthetic meeting for the N5 (time to transcript) and N9 (dictations out of "me") checks.

`make <meeting_dir> <minutes> [--no-stamps]`: a meeting folder in the recorded layout
(metadata.json, mic/ and system/ 16 kHz mono 16-bit WAVs in 5 s chunks) spoken by macOS `say`
to files (nothing is played). The mic track is one voice talking to the call plus 20
dictations, each with two words used nowhere else, stamped in metadata.json the way the app
stamps them; the system track is a second voice; turns alternate. `--no-stamps` leaves the
dictation stamps out (the control). Writes <meeting_dir>.truth.json beside it.

`score <meeting_dir>.truth.json <transcript.json>`: dictations whose words are absent from the
"me" side, and the "me" word count against the reference.

Transcribe the folder with the real sidecar (MeetingTranscriptLiveCheck):
    PHONON_MEETING_CHECK_DIR=<meeting_dir> PHONON_MEETING_CHECK_OUT=<empty dir> \\
        swift test --package-path bar --filter MeetingTranscriptLiveCheck
"""
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import wave

RATE = 16000
CHUNK = 5 * RATE
BASE_NS = 1_000_000_000

MEETING = [
    "I think we should move the launch review to Thursday afternoon.",
    "The budget numbers from last quarter look reasonable to me.",
    "Can you send the updated slides before the end of the day?",
    "We still need someone to own the onboarding checklist.",
    "Let us keep the scope small for the first release.",
    "The design team wants another week for the settings page.",
    "I talked to the vendor and they can ship the parts next month.",
    "Our main risk is the testing schedule, not the code.",
    "Please add the open questions to the shared document.",
    "I agree with that plan, as long as we review it on Monday.",
    "The customer asked whether we support offline mode.",
    "We should write down the decision so nobody forgets it.",
]
CALL = [
    "That works for us, Thursday is fine.",
    "I can send the slides in about an hour.",
    "The vendor confirmed the delivery date this morning.",
    "We are still waiting on legal for the contract.",
    "The numbers in the second table do not add up yet.",
    "Could we push the demo to next week instead?",
    "My team will take the onboarding checklist.",
    "I have a question about the pricing page.",
    "Sounds good, let us move on to the next item.",
    "We tested the beta with twelve people last week.",
]
# each dictation has two words that appear nowhere else
DICTATIONS = [
    ("Remind me to buy pineapple and walnuts.", ["pineapple", "walnuts"]),
    ("Email the plumber about the kitchen faucet.", ["plumber", "faucet"]),
    ("Book a table at the sushi restaurant.", ["sushi", "restaurant"]),
    ("Order more printer toner for the studio.", ["toner", "studio"]),
    ("Water the tomato plants on the balcony.", ["tomato", "balcony"]),
    ("Call grandma about the birthday cake.", ["grandma", "cake"]),
    ("Renew the passport before the holiday.", ["passport", "holiday"]),
    ("Pick up the dry cleaning and the umbrella.", ["cleaning", "umbrella"]),
    ("Find a recipe for lemon pancakes.", ["lemon", "pancakes"]),
    ("Schedule the dentist for my daughter.", ["dentist", "daughter"]),
    ("Buy a new charger for the bicycle lamp.", ["bicycle", "lamp"]),
    ("Return the library books about volcanoes.", ["library", "volcanoes"]),
    ("Ask the landlord to fix the radiator.", ["landlord", "radiator"]),
    ("Send flowers to the hospital tomorrow morning.", ["flowers", "hospital"]),
    ("Cancel the gym membership in January.", ["gym", "membership"]),
    ("Look up train times to the mountains.", ["train", "mountains"]),
    ("Feed the goldfish and clean the aquarium.", ["goldfish", "aquarium"]),
    ("Write a thank you card for the neighbors.", ["card", "neighbors"]),
    ("Check the oil level in the motorcycle.", ["oil", "motorcycle"]),
    ("Try the new coffee shop near the museum.", ["coffee", "museum"]),
]


def tts(text, voice, folder, name):
    path = os.path.join(folder, name + ".wav")
    subprocess.run(
        ["say", "-v", voice, "-o", path, "--data-format=LEI16@16000", "--file-format=WAVE", text],
        check=True,
    )
    with wave.open(path, "rb") as w:
        assert w.getframerate() == RATE and w.getsampwidth() == 2 and w.getnchannels() == 1
        return w.readframes(w.getnframes())


def make(out, minutes, stamps=True):
    total = int(minutes * 60 * RATE)
    rng = random.Random(7)
    tmp = tempfile.mkdtemp(prefix="say-")
    try:
        meeting = [tts(t, "Samantha", tmp, f"m{i}") for i, t in enumerate(MEETING)]
        call = [tts(t, "Daniel", tmp, f"c{i}") for i, t in enumerate(CALL)]
        dictated = [tts(t, "Samantha", tmp, f"d{i}") for i, (t, _) in enumerate(DICTATIONS)]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    mic = bytearray(total * 2)
    system = bytearray(total * 2)

    def put(track, at, pcm):
        end = min(len(track), at * 2 + len(pcm))
        track[at * 2:end] = pcm[: end - at * 2]
        return at + len(pcm) // 2

    # dictation slots evenly spread; each is a 0.3 s lead, the phrase, 0.2 s tail
    slots = []
    for i in range(len(DICTATIONS)):
        start = int((i + 0.5) * total / len(DICTATIONS))
        length = int(0.3 * RATE) + len(dictated[i]) // 2 + int(0.2 * RATE)
        slots.append((start, start + length))
    at = int(1.0 * RATE)
    mic_ref, turn = [], 0
    while at < total:
        speaker_mic = turn % 2 == 0
        for _ in range(rng.randint(1, 3)):
            if at >= total:
                break
            if speaker_mic:
                # the user does not talk to the call over a dictation
                clash = next((s for s in slots if at < s[1] + RATE // 2 and s[0] - RATE // 2 < at + 6 * RATE), None)
                if clash:
                    at = clash[1] + RATE // 2
                    continue
                k = rng.randrange(len(meeting))
                at = put(mic, at, meeting[k])
                mic_ref.append(MEETING[k])
            else:
                k = rng.randrange(len(call))
                at = put(system, at, call[k])
            at += int(rng.uniform(0.3, 0.9) * RATE)
        turn += 1
        at += int(rng.uniform(0.2, 0.8) * RATE)
    for i, (start, _) in enumerate(slots):
        put(mic, start + int(0.3 * RATE), dictated[i])

    os.makedirs(out, exist_ok=True)
    meta = {
        "schema_version": 2, "id": os.path.basename(out), "started_at": "2026-11-12T07:00:00Z",
        "started_host_ns": BASE_NS, "system_source": "screencapturekit", "sample_rate": RATE,
        "chunk_seconds": 5, "mic_chunks": [], "system_chunks": [],
        "mic_first_sample_host_ns": BASE_NS, "mic_start_offset_samples": 0,
        "system_first_sample_host_ns": BASE_NS, "system_start_offset_samples": 0,
        "ended_at": "2026-11-12T08:00:00Z", "end_reason": "call_ended",
        "dictations": [
            {"start_host_ns": BASE_NS + s * 1_000_000_000 // RATE, "end_host_ns": BASE_NS + e * 1_000_000_000 // RATE}
            for s, e in slots
        ],
    }
    if not stamps:
        del meta["dictations"]
    for name, track in (("mic", mic), ("system", system)):
        folder = os.path.join(out, name)
        os.makedirs(folder, exist_ok=True)
        for index, first in enumerate(range(0, total, CHUNK)):
            count = min(CHUNK, total - first)
            file = f"{index:04d}.wav"
            with wave.open(os.path.join(folder, file), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(RATE)
                w.writeframes(bytes(track[first * 2:(first + count) * 2]))
            meta[name + "_chunks"].append({
                "index": index, "file": file, "start_sample": first, "sample_count": count,
                "start_host_ns": BASE_NS + first * 1_000_000_000 // RATE,
            })
    with open(os.path.join(out, "metadata.json"), "w") as f:
        json.dump(meta, f)
    truth = {"dictation_keywords": [k for _, k in DICTATIONS], "mic_sentences": mic_ref,
             "audio_seconds": total / RATE}
    with open(os.path.abspath(out).rstrip("/") + ".truth.json", "w") as f:
        json.dump(truth, f)
    print(f"{out}: {total / RATE:.0f} s per track, {len(mic_ref)} mic sentences, "
          f"20 dictations ({'stamped' if stamps else 'not stamped'})")


def score(truth_path, transcript_path):
    truth = json.load(open(truth_path))
    transcript = json.load(open(transcript_path))

    def words(speaker):
        text = " ".join(t["text"] for t in transcript["turns"] if t["speaker"] == speaker)
        return re.findall(r"[a-z']+", text.lower())

    me, call = words("me"), set(words("call"))
    leaked = [k for k in truth["dictation_keywords"] if any(w in set(me) for w in k)]
    on_call = sum(w in call for k in truth["dictation_keywords"] for w in k)
    reference = re.findall(r"[a-z']+", " ".join(truth["mic_sentences"]).lower())
    print(f"dictations removed from me: {20 - len(leaked)} of 20 (leaked: {len(leaked)})")
    print(f"dictation words on the call side: {on_call}")
    print(f"me words: {len(me)}, reference mic words: {len(reference)}")


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "make":
        make(sys.argv[2], float(sys.argv[3]), stamps="--no-stamps" not in sys.argv[4:])
    elif len(sys.argv) == 4 and sys.argv[1] == "score":
        score(sys.argv[2], sys.argv[3])
    else:
        sys.exit(__doc__)
