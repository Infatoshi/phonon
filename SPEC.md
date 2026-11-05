# SPEC

## User vocabulary onboarding (the user world model)

Goal: a new user gets a dictionary and a short profile that fit them on day
one, without reading their files by hand and without any data leaving the
machine. Parakeet and the correction model stay as shipped; the profile and dictionary carry
the personalization. Fine-tuning is a later, opt-in POC (see end).

### Principles

- Local only. The only network call is `gh` read access to the user's own
  repos, and only when they tick it. Email is out of scope for v1.
- Consent per source, shown before anything is read. The list of what was
  read, with counts, stays visible afterwards.
- Deterministic extraction, model-based mining. Extraction pulls only text the
  user authored; models never see the raw trace files.
- The user approves every term. Nothing enters `dictionary.json` unreviewed.
- Held-out evaluation. The miner is developed blind to the user's existing
  dictionary and corpus; those are the test set.

### Flow

0. Consent sheet. Lists detected sources with sizes, a checkbox each,
   a plain statement of where results go
   (`~/Library/Application Support/Phonon/profile/`) and that nothing is
   uploaded. Defaults: agent traces on, local repos on, GitHub off, prior
   dictation apps on.
1. Inventory (seconds, no model). Detect: `~/.claude/projects`,
   `~/.codex/sessions`, `~/.grok/sessions`, Cursor SQLite, Aqua Voice
   `settings.json`, Wispr Flow SQLite, superwhisper and VoiceInk history,
   git repos under `$HOME` to depth 3 (excluding `Library`), and with GitHub
   on, `gh repo list` names, descriptions and READMEs.
2. Extract (deterministic). Per source, one line per user-authored message:
   Claude `type=user` string content, Codex `role=user input_text` with
   injected context dropped (lines over 80 words, headings, tags), Grok
   `type=user`, dictation histories' corrected text, repo docs
   (README, AGENTS, DEVLOG, SPEC) plus file trees to depth 2, no code bodies.
   Output `profile/extract/<source>.txt` with counts shown in the UI.
3. Mine. Three stages; the model only adds and labels, it never drops.
   a. Seed (no model): the user's identity from `id -F`, `git config
      user.name`, `$USER`, the GitHub login, plus repo names. These are
      always candidates; the surname of the author never appeared in the
      first POC because it lived inside paths (`/home/<first><last>`).
   b. Candidates (no model): CamelCase, ALLCAPS, digit tokens, hyphenated
      identifiers, capitalized non-initial words with count >= 2, and
      lowercase tokens absent from a fixed general-English list shipped
      with the app. Never a stoplist derived from the user's own corpus:
      the user's most frequent jargon is by definition in their own top-5k.
   c. ASR oracle (no model): for each candidate, synthesize with macOS
      `say` in two voices, transcribe with the shipped Parakeet, keep the
      term only when the output differs from the term, and record the
      outputs as `spoken_forms`. About 0.7 s per term, all local. The
      model is not asked whether ASR will mangle a term; it cannot know.
   d. Model pass (local correction model, one agent per source): reads
      sampled short lines to add names, multiword units and observed
      manglings from the text; assigns `kind`; writes `evidence`. Output
      `profile/mined/<source>.json`: `{term, count, sources, kind,
      spoken_forms, evidence}`, ranked by count and source breadth.
   Production runs this on the local correction model; the quality bar is
   the Opus baseline in DEVLOG (2026-08-30).
4. Merge and review. Union across sources, score by rank plus multi-source
   bonus, dedup against the dictionary and the rejected list. Review sheet
   grouped by kind; accept, edit, reject; spoken forms editable. Accepted
   terms land in `dictionary.json` with `source: "profile-miner:<source>"`.
   Rejections persist in `profile/state.json` so they are never re-proposed.
5. Profile files, prefix-cached in the polish prompt, bounded to about 1.5k
   tokens together: `profile/user.md` (role, domains, tools, projects,
   people, style) and `profile/vocab.md` (accepted terms by kind with spoken
   forms). The correction model reads them every pass.
6. Refresh button ("Update my vocabulary"). Re-runs extract from per-source
   watermarks in `profile/state.json`, mines only the delta, dedups against
   dictionary plus rejections, shows the diff for review, rewrites the
   profile files. Same code path as onboarding.

### Two vocabulary layers

The 2026-08-30 baseline showed the mined vocabulary and the hand-made
dictionary barely overlap, and both are right. The dictionary is a domain
glossary (FP16, HBM3e, FlashAttention-3, DGX B200, LLaMA 2); the miner finds
the personal layer (machine names, project names, house jargon, collaborators,
the user's own spoken manglings). So:

- Domain packs: shipped, shared, toggleable glossaries per field (GPU and ML
  systems first). No mining; curated once, updated with releases.
- Personal layer: mined per user by the flow above. Plain-English words used
  as names (anvil, gamer, tau, blaze, magma, lane, tape, golden) are the
  highest-value entries: ASR spells them fine, the correction model must
  learn not to "fix" them.

### Evaluation (dev only)

Hold out `dictionary.json` and corpus final transcripts. Gold is the
used-or-starred subset filtered through the ASR oracle: a gold term must be
one Parakeet mangles from clean TTS or one observed mangled in the corpus.
Entries Wispr needed but Parakeet already transcribes (NCCL, FSDP, SM, M3)
are not misses for Phonon. Report recall of gold at top-100/200/400 with
spoken-form variants collapsed onto their term, and a precision proxy
(share of proposals present in dictionary or corpus). A local model ships
in the mining role only when it reaches the Opus baseline within a stated
margin on the same extracts. Numbers and miss analysis live in DEVLOG.

### On-device fine-tuning: Train Now

The refiner learns from the user's own corrections, on the user's Mac, only
when the user presses a button. Decided 2026-10-29.
- Data: dictations whose corpus `metadata.json` holds a `user_edit` record
  with a real edit. The input is the raw ASR transcript in the exact prompt
  Phonon used (system prompt plus the vocabulary line); the target is
  `edited_text` with the inserted text's first-letter case and final period
  put back. Corrections that change only those are not counted: they fit a
  dictation into text already on screen, which the refiner does not see
  (40 of the owner's 44 Wispr corrections did that; a cursor-aware casing
  rule is the cheaper fix for them). Rows never leave the machine.
- UI: History shows "You hand-corrected N transcriptions" and a Train Now
  button once N reaches a minimum set by measurement. Training runs in the
  background with visible progress and Cancel. A dictation preempts it: the
  current step is dropped or finished, the dictation runs at normal latency,
  then training resumes. There is no scheduler.
- Recipe: a LoRA adapter rebuilt from the shipped refiner every time, never
  stacked on an earlier personal adapter (a second pass on a finished
  refiner regressed, DEVLOG 2026-10-29). The training rows are the user's
  corrections plus up to 2.75 times as many of their own uncorrected
  dictations, target the inserted text, at lr 1e-4 for 2 passes. Public
  replay is out: on the owner's data it moved general dictation by +0.013
  to +0.07 WER.
- Gate before use: a held-out slice of the user's corrections (the newest
  20%), a bundled public check set (margin 0.01), and the user's 30 newest
  uncorrected dictations, never trained on: the adapter may rewrite at most
  8% of the shipped refiner's words there, counted as typed. The new adapter
  replaces the current one only if it passes all three. The app reports the
  result in one line and keeps one-click revert to the shipped refiner.
- Serving: a dictation longer than the longest training row uses the shipped
  weights. Merging into the K-quant target is not a serving path: a gentle
  adapter's deltas fall below the quantization step and vanish (measured).
  The engine needs a runtime low-rank add, fused into its per-layer kernels
  (projected +3.3% bytes per pass); until then an active adapter is served
  by the MLX refiner.
- Value is unproven: on 42 held Wispr-era corrections the safe recipe was
  +0.000 [-0.006, +0.006] fair WER, and recipes that moved the corrections
  more broke real_580 and sep. Re-measure on 100+ Phonon-recorded
  corrections before shipping the button.
- Budgets, measured on an 8 GB M1 and the M4 Max (DEVLOG 2026-10-30): wall
  time for 25, 100 and 400 edit rows; peak memory footprint; dictation
  latency while training runs. Where trainer plus dictation stack exceed 85%
  of the GPU's recommended working set, the trainer is evicted at key-down
  (saves its state, exits, restarts after 30 s of quiet); otherwise it is
  paused. On an 8 GB M1 the recognizer still slows about 3x at key-up while
  a trainer is resident, so there Train Now says dictation is slower until
  it finishes.
- Personal adapters never leave the machine and never enter public
  training.
- Parakeet fine-tuning stays out: microphone drift after a one-time tune.
No enrollment scripts. Users do not read generated sentences aloud.

## Input

### Shortcut

Default is the bare Globe (fn) key, because the default user is on a Mac
keyboard. Key-down starts recording at once. A press under 250 ms is a tap;
a second tap within 350 ms of the first release latches recording on; while
latched, the next key-down stops and commits. A lone tap yields a capture
without speech, which the no-speech path discards. Right Option and
Ctrl+Space remain as alternatives (`shortcut_mode` in settings.json).

### Microphone

Phonon never changes the system default input. With nothing ranked in
`microphone_priority`, it follows the system input the user already chose,
except a Bluetooth headset microphone (CoreAudio transport type), which is
skipped for the built-in one: macOS makes a headset the default input when
it connects, and opening its mic drops the headset to call-quality audio.
Ranking a microphone, including a headset, overrides that. The recorder
re-resolves on `AVAudioEngineConfigurationChange` and on system default
input changes, and reads back the input unit's live device before each
capture, because AVAudioEngine can silently move the unit to the new
default when it rebuilds its graph.

## Capsule and Dock (macOS, since 1.0.1)

- The capsule and its panels sit above the Dock: the panel frame's bottom is the screen's visible frame
  (`PanelGeometry.clearOfDock`), so a visible Dock never covers them; with the Dock hidden nothing moves.
- Settings › Dictation › "Show the bar between dictations" (settings.json `show_idle_capsule`, default on). Off
  hides the capsule when idle; it still shows while dictating, while correcting, and while meeting notes record
  (the red light and stop button live in it).
- Settings › System › "Show in Dock" (settings.json `show_in_dock`, default on), also in the ϕ menu. Off runs
  Phonon as a menu-bar-only (accessory) app; the window opens from the ϕ menu.
- Both keys are app-side; phonon-core keeps them through its unknown-key passthrough.

## Competing dictation apps

Two apps holding fn taps means both record and both insert. Phonon detects
the competitor, asks once, quits it with consent, then installs its own tap
last so it sits at the head of the tap chain.

| app | bundle id | default hotkey | hotkey on disk | mic when idle | quit path |
|---|---|---|---|---|---|
| Wispr Flow | `com.electron.wispr-flow` (+ `.accessibility-mac-app` Swift helper) | fn hold, fn+Space hands-free | `~/Library/Application Support/Wispr Flow/config.json` → `prefs.user.shortcuts` key containing `63` | free | `terminate()`; also terminate the helper if it outlives the main app |
| Aqua Voice | `com.electron.aqua-voice` (+ `AquaMacOSBridge` binary) | fn hold, fn+Space lock | `.../Aqua Voice/settings.json` → `hotkeys[].keys == "Fn"` | free | `terminate()`; bridge handles SIGTERM |
| superwhisper | `com.superduper.superwhisper`, `-setapp` | Option+Space (third-party report; fn allowed) | unverified | unverified | `terminate()` (native) |
| VoiceInk | `com.prakashjoshipax.VoiceInk` | user-chosen; fn preset exists | `defaults` `Shortcut_primaryRecording` JSON `keyCode == 63` | free (AudioUnit prepared, started on record) | `terminate()` (no quit override) |
| MacWhisper, Typeless | `com.goodsnooze.MacWhisper`, `now.typeless.desktop` | unverified | unverified | unverified | `terminate()` |

Policy:

1. At launch and on every hotkey change, scan
   `NSWorkspace.shared.runningApplications` against the watch set; subscribe
   to `didLaunchApplicationNotification` for relaunches.
2. Prompt only when a competitor runs AND its stored hotkey equals Phonon's
   (read-only reads of the files above; superwhisper assumed to conflict on
   fn or Option+Space). Text names the app, says it returns at next login,
   offers "Quit it" or "Keep it, change my hotkey".
3. On consent: `terminate()`, poll `isTerminated` at 250 ms, `forceTerminate()`
   at 5 s, then rescan for orphaned helpers (Wispr helper app, Aqua bridge
   via pgrep + SIGTERM).
4. Never quit mid-dictation: if the input device reports
   `kAudioDevicePropertyDeviceIsRunningSomewhere == 1` while Phonon is idle,
   wait and retry.
5. Confirm the mic is free, then install Phonon's fn tap. Re-prompt at most
   once per session on relaunch. Never touch the competitor's login items
   or preferences.

Wispr Flow holds a single-instance lock and ships `openAtLogin: true` by
default, so after a quit it returns only via the login item or the user.
It does not detect other dictation apps.

Untested before shipping: Wispr helper self-exit after SIGKILL, Aqua bridge
orphan exit, superwhisper default hotkey and idle mic state.

## Update check (macOS, since 1.0)

Phonon tells its users when a newer Mac release is out. It never downloads or
installs anything itself (`bar/Sources/UpdateCheck.swift`, `UpdateController.swift`).

- When: 60 s after launch, then every 24 h while it runs, if Settings › System ›
  Check for updates is on (settings.json `check_for_updates`, default on). "Check
  Now" in Settings and Phonon › Check for Updates… work with it off.
- What is sent: one unauthenticated GET to
  `api.github.com/repos/Infatoshi/phonon/releases?per_page=10`, User-Agent
  `Phonon`, ephemeral session (no cookies or cache). No identifier, version or
  usage. GitHub sees the IP address.
- Which release counts: not a draft, not a prerelease, has a `Phonon.dmg` asset,
  tag `vX.Y.Z` (so Windows betas never match), compared numerically and newer
  than the running `CFBundleShortVersionString`. A binary without an Info.plist
  never checks.
- How it tells the user: one alert per version (`update_announced`), held until
  no dictation is recording or processing (up to 10 min, else the next check).
  After that, "Update to Phonon X.Y.Z…" stays in the menu bar menu, the sidebar
  and Settings.
- Update action: a Homebrew install (a `Caskroom/phonon` entry exists) gets
  `brew upgrade --cask phonon` copied; a DMG install gets the release page.
- Release order: attaching `Phonon.dmg` announces the release, so bump the tap's
  cask in the same sitting, or brew users are told before `brew upgrade` can
  see it.
- Reach: 0.2.3 and earlier have no check. The first announcement goes out with
  the release after 1.0.

## Windows

Phonon on Windows is one executable, `phonon-win.exe`, in `crates/phonon-win`.
It is a beta. Continuous integration runs it end to end on `windows-latest`;
no one has run it on real hardware.

### What ships and what is downloaded

The executable carries the correction prompt and `assets/startup.wav`. It
carries no weights and no third-party runtime. First run downloads four
components, in this order:

| Component | Pinned to | Size |
| --- | --- | --- |
| sherpa-onnx tools and ONNX Runtime | release `v1.13.6` | 18 MB |
| llama.cpp CPU build | release `b10726` | 18 MB |
| Parakeet TDT 0.6b v2, int8 ONNX | commit `1ab9323` | 661 MB |
| Gemma 4 E2B instruction tuned, q4_0 GGUF | commit `675cff4` | 3.3 GB |

Every file has its SHA-256 in `crates/phonon-win/src/manifest.rs`. A file that
does not match is deleted, not used. A component is built in a staging
directory and renamed at the end, so an interrupted download never looks
installed. Runtimes come first because they are small: a slow link shows
progress within a minute.

The small runtimes are chosen instead of Rust bindings on purpose. Both
projects publish prebuilt Windows binaries, so continuous integration compiles
no C++ and a user installs no toolchain.

### The two models

Speech recognition is Parakeet TDT 0.6b v2, the same acoustic model macOS runs
on MLX, exported to ONNX and quantised to int8. `sherpa-onnx-offline.exe` reads
one wave file and writes one JSON object to standard output.

Correction is Gemma 4 E2B, Google's own quantisation-aware q4_0 GGUF, resident
in `llama-server.exe`. The prompt is `prompts/polish_v2.txt`, embedded in the
executable. The temperature and the output budget rule, `ceil(spoken * 1.8) +
24` clamped to 48 and 2048 tokens, are the ones in `sidecar/polish_server.py`.
Both platforms therefore correct a transcript the same way.

Correction is mandatory here as it is on macOS. There is no recognition-only
mode.

### Readiness

First run writes the fixture, recognises it, corrects it, and applies both
rules from `crates/phonon-core/src/lib.rs`: the transcript must contain the
fixture words, and the correction must be within one word below and two words
above the transcript's length. Weights that load are not readiness. The same
check is `phonon-win.exe selftest`, and it is the release gate.

### Threads

Two. The main thread owns the message-only window, the notification-area icon,
and the `WH_KEYBOARD_LL` hook, and does nothing else: Windows removes a
low-level hook that blocks past its timeout. The hook reads a virtual key code,
asks the latch, and pushes the decision onto a channel.

The worker thread owns everything slow: the download, the resident correction
server, the microphone, recognition, and insertion. A cpal stream cannot move
between threads on Windows, so a recording begins and ends on that thread.

### The dictation key

Right Control, held to talk. Two taps latch; the next press stops. The rules
are `crates/phonon-hotkey`, ported from the Swift `HoldKeyTapLatch` and tested
against the same cases. Phonon swallows the key, so Right Control stops acting
as a modifier while Phonon runs. `PHONON_WIN_HOTKEY` moves it.

### Capture and insertion

Capture is WASAPI through cpal, following the system default input device, in
whatever format the device offers, mixed to one channel and written as 16-bit.
sherpa-onnx resamples, so no resampler runs here.

Insertion types text of 200 UTF-16 units or fewer with `SendInput` and Unicode
key events, which leaves the clipboard alone. Anything longer, and anything
that fails, goes through the clipboard and Control-V; the previous clipboard
text is restored a quarter of a second later.

### Known gaps

- Unsigned. SmartScreen warns on first launch.
- No microphone priority list, no streaming preview, no screen context, no
  dictionary retrieval, and no corpus. Windows runs the recognise-and-correct
  path only.
- Unpacking uses the `tar` that ships with Windows 10 build 17063 and later.
- Recognition loads the encoder per pass, so the first word costs more than it
  does on macOS.

## Refiner (research)

The next correction model is built in `ml/`: Parakeet TDT v2, then per-utterance retrieval over a
vocabulary indexed from the user's own code, then LFM2.5-1.2B with a LoRA adapter, shipped as the
same int4 weights on every platform. `ml/SPEC.md` has the design and the evaluation roadmap;
`ml/AGENTS.md` has the current state and how to run the pipeline.

A trained refiner runs today as a local model. When
`~/Library/Application Support/Phonon/models/refiner/` holds an MLX model with a
`phonon_refiner.json` marker, the engine passes that directory to the correction
sidecar instead of the pinned Gemma download, and the sidecar switches to the refiner's
own format: the phonon-bench system prompt with a `Vocabulary the speaker may use:` line
built from the dictionary envelope (screen-confirmed, phonetic and exact-replacement
targets, likely, then canonical terms, capped at `min(80, max(30, 3 x words))`), the bare
transcript as the user turn, an output cap of `1.5 x input tokens + 32`, and the word-count
guard (fall back to the transcript when the output is over twice or, past 8 words, under
0.4 of its length). No profile block, no repetition penalty. A refiner trained on a user's
dictations is that user's alone and is never downloaded from or published to a hub.
`phonon doctor` names the model in use.

## Dictionary boosting in speech recognition (macOS, Metal Parakeet)

The engine sends the ASR sidecar `{"cmd":"terms"}` at startup and on every `reload_dictionary`:
each entry's canonical spelling once (the replacement when there is one), minus ordinary English
(every word in `assets/english_words.txt`, the same check the phonetic guard uses), which the
recognizer already spells and a boost could only insert. `phonon dictionary asr-terms` prints the
list. The sidecar builds the term trie (`sidecar/asr_terms.py`: the tokenizer's BPE from the
model's vocabulary, case variants, bonus 4 to start and to continue a term, budget 6) and sends it
to `pa serve` as a dictionary request, which swaps it in place between windows (room for about 11k
terms). The greedy TDT decode follows the trie: a bonus may push it onto a term, a walk commits
only on a complete term at a word boundary and otherwise rolls back, and a committed boosted term
takes the dictionary's spelling. Windows answered on the MLX fallback are not boosted. Meetings are
not transcribed in the app.

Measured (owner dictations, public refiner, 2026-11-02, 733-entry dictionary, 604 terms boosted):
real_580 WER after the refiner -0.0011 [-0.0019, -0.0003], sep -0.0011 [-0.0020, -0.0005];
dictionary-term misses after the refiner 72 -> 54 and 90 -> 63; 1,351 public clips with the same
list: no WER change, one extra term insertion. The in-place request matches a trie loaded at
startup token for token (150 of 150 recent dictations).

## Intended text and writing modes (the public refiner since 1.0)

The refiner's target is the text the speaker intended to type, written down so a teacher, a judge and the student
read the same definition (`intent_v0/code/intent_spec.py`). Universal rules, every mode: fix misrecognized words and
terms only with evidence (other recognizers, the vocabulary line, unambiguous context); remove filler (hesitations,
stutters, repeats, meaningless "like / you know / I mean", tag questions, sentence-opening linking words); keep only
the speaker's final version of a false start or self-correction, dropping the correction words; digits and symbols
the way an engineer types them; never invent (no unspoken word, no completed fragment, no flipped negation) and never
drop meaning (claims, explanations, requests, examples and qualifiers stay in every mode); plain text unless the mode
allows otherwise.

Modes, chosen from the target app's category (`app_context::AppCategory`):
- minimal (docs, other): the universal rules and nothing else; one paragraph.
- casual (chat): the speaker's informal words kept, a restated point kept once, every sentence lowercase except
  names, acronyms and "I", no final period.
- formal (email): complete sentences, grammar slips fixed without rephrasing, slang to its plain form, dictated
  greeting and sign-off on their own lines, paragraphs on topic changes; nothing added.
- prompt (AI prompt, code): a revised, contradicted or restated request kept once in its final version; pure
  hesitation dropped, context and explanation kept; separate steps as a numbered list; backticks around code allowed;
  never answers or extends the request.
The student sees the mode as one system-prompt line ("Writing mode: casual (a chat message).").

In the app: a refiner trained with modes lists its lines in `phonon_refiner.json` as `"writing_modes": {mode: line}`
(optional `"writing_modes_by_app_category"` overrides the category table). The correction sidecar puts the line of
the request's `context.app_category` mode between the fixed instructions and the vocabulary line, exactly as in
training (`polish_server.WRITING_MODE_BY_CATEGORY`: chat casual, email formal, ai_prompt and code prompt, docs and
other minimal; no app named: minimal) and reports it as `diagnostics.writingMode`. A refiner without the key gets
no line, so its prompt is unchanged. Streaming refine needs the mode before key up: the bar sends the target app
known at key down in `stream_start` (`"app"`) and the resolved one (a browser tab's host) in `stream_app`; the
engine sends its category with every `stream_commit`, so background drafts are refined under the same prompt as
the final correction. A mode that changes after a draft only costs speed: the answer is the final prompt's.

Measured by a blind pairwise judge (Claude Opus 5.5, both orders, a verdict only when both orders agree) that
reads the spec, the mode and the evidence of what was said (both raw recognizers, and the creator's captions on the
three-speaker set), ranking meaning over correct words over cleanup over register. Objective side checks: leftover
filler per 100 words, WER against captions.

## Streaming refine (settings.json `stream_refine`; default on with 16 GB of memory or more)

settings.json `"stream_refine": true` or `false` decides; without the key it is on for Macs with
16 GB of memory or more (`hw.memsize`) and off below, since it keeps the GPU busy about 28% of
dictation time on an M4 Max and is unmeasured on 8 GB machines. `PHONON_STREAM_REFINE=0` or `=1`
in the engine's environment overrides both for that run. The engine passes its decision to both
sidecars (`PHONON_STREAM_REFINE` in their environment) and to the bar (`"stream_refine"` in its
ready event). It needs the live preview stream (`streaming`, also on by default): without a
stream there is nothing to refine early.

Goal: most of the transcription and correction work happens while the speaker is still talking,
so the wait after key release no longer grows with the length of the dictation. The text that is
inserted does not depend on streaming: the refiner's output is the greedy output of the final
prompt, verified token by token.

ASR (`sidecar/stream_asr.py` with its `SCHEME`, driven by `asr_server.py` when `stream_start`
carries `"segments": true`). Every 0.4 s stream chunk, once the stream has heard speech, a
speculative round transcribes the whole recording so far, so the transcript so far reaches the
present with all of its left context. Past 60 s of pending audio a round also commits its words up
to 2 s before its end, and later rounds keep 20 s of left context before that point, so no window
exceeds 80.4 s (Metal Parakeet's capacity is 82 s). At key release only `[t_cut - 8 s, end]` is
transcribed, where `t_cut` is the end of the last round minus 2 s: words before `t_cut` come from
the round, which heard them with at least 2 s of right context. The causal energy VAD
(`sidecar/stream_vad.py`) only starts the rounds: in long stretches without a pause its noise
floor rises to speech level and it stops hearing speech, so it never decides what is transcribed.
Measured against whole-file ASR of the same clips on pubeval_v0 (1,351 public clips of 3-60 s
with human verbatim transcripts; windowed minus whole-file WER, paired bootstrap 95% CI;
pre-registered bound: CI upper at most +0.004): this scheme +0.0009 [-0.0006, +0.0023], every
length band's CI including 0; the same rounds with a 1 s release window +0.0047
[+0.0029, +0.0064]; commits every 30 s with 20 s of left context +0.0044 [+0.0015, +0.0079]
(+0.010 on 30-60 s clips); 0.2.0's pause-cut segments (a 0.5 s pause closes a segment,
transcribed with 4 s of left and 2 s of right context, 1 s release window) +0.0119
[+0.0093, +0.0147], and +0.0230 while the VAD also decided which segments to transcribe (empty
transcripts on clips without pauses). Transcribing the whole recording at release instead (the
final transcript then is the whole-file ASR) was slower at key up (below). parakeet-mlx's own
chunked streaming is not used for this: its partials differed from batch ASR of the same audio
by 0.46 WER.
At key up the bar sends the last partial chunk, so the stream holds the whole recording (the
recorder keeps only the last 120 s of samples), and then `stream_finish`: the release window is
transcribed while the bar writes the WAV. The bar's `transcribe` with the WAV follows as before
(corpus entry, speech gate); the engine sends the ASR text once both are in, or an empty result
as soon as the gate finds no speech (the release text is then dropped, so a no-speech clip never
inserts text). If the sidecar fails to finish the stream, the WAV is transcribed the usual way. A
`transcribe` for a segment stream without a prior `stream_finish` (an older bar) appends whatever
audio the WAV holds beyond the stream and finishes it then.

Correction (`sidecar/stream_decode.py`, `sidecar/stream_draft.py`, `polish_server.py`). The
engine forwards each transcript so far as `stream_commit` with the same dictionary envelope as
the final polish (and the screen text, once the bar sends `screen_context`). The sidecar keeps a
rollback-able model state, refines the transcript so far, and keeps the output as a draft. The
state lives in one of two backends behind the same decode loop (`RollbackDecoder`): with the
native refiner engine (`phonon_engine.json`), its per-dictation session (`pe serve` ops `open`,
`rollback`, `forward` with argmax ids per row; `PeEngine`), into which the sidecar sends the
token ids it encodes with the refiner's tokenizer (the GGUF's vocabulary and merges are the
same); with MLX, a second cache over the same weights (`stream_refiner.Engine`: plain KV for full
and sliding attention, full input history for the short convolutions). A one-shot request to the
native engine (warmup, prewarm, a run without commits) ends its session; the next commit opens a
new one from the cached system-prompt prefix. A session forward that hits the engine's 2048-token
context cap falls back to the one-shot run, which continues past it. Draft blocks after a
rejection follow a per-backend ladder: on the native engine 7, 15, 31, ... tokens (its megakernel
verifies up to 8 rows for the price of 2), on MLX 3, 31, 63, ... A newer commit replaces one that
has not started; a run aborts a running commit between forward passes, and background passes are
capped at 127 tokens so that wait is short. At release, when the final prompt equals the prompt of the last completed background
refine (the speaker stopped inside the last 0.4 s chunk), that draft is the answer and no pass
runs; otherwise the run prefills only the prompt after the cached prefix, fused with the draft
in one forward pass, verifies the draft in large blocks, and decodes what the draft does not
cover with prompt-lookup drafts. Off this path (no commits, or a Gemma polisher), `run` is
unchanged.

Measured (M4 Max, MLX refiner, 40 held-out dictations replayed at 1x through the two sidecar
processes): post-release p50/p90 for dictations of 5 s and longer 51/110 ms against 528/1178 ms
for today's path on the same rows; over 20 s 47/97 against 866/1224; under 5 s 64/77 against
194/248. GPU busy during dictation 28% of wall time. End-to-end fair WER on 564 rows +0.0001
[-0.0054, +0.0053] against today's path; term hit +0.0034. On the native engine (same 40
dictations, 2 runs each, the 7-15-31 ladder): 5 s and longer 56/100 and 53/101 ms against
504/1568 and 476/1677 ms for its one-shot path; under 5 s 68/95 and 61/73 against 213/241 and
230/257; GPU busy 27.8%; final text identical to the one-shot refine of the same transcript on
39 of 40 in every run. Those numbers are for 0.2.0's pause-cut ASR scheme.

With the whole-recording rounds above (the bundle's `phonon engine` driven as the bar drives
it, 40 held-out dictations at 1x, owner refiner on the native engine, 2 runs each), key-up to
text p50/p90: on Metal Parakeet 31.4/130.2 and 32.0/133.9 ms against 221.2/681.0 and
208.8/658.1 with streaming refine off; on MLX Parakeet 96.6/339.1 and 97.6/356.5 ms (0.2.0's
scheme on MLX, same rows: 58.8/135.1 and 60.6/135.7; off 246.3/777.7 and 253.0/773.1). The
release reused the last background draft without a forward pass on 25 of 40 dictations. The
alternatives measured the same way (Metal, 1 run each): a whole-file release up to 82 s after
pause-cut rounds 75.7/465.7 ms (draft reused 10/40), after whole-recording rounds 65.9/372.3
(17/40).

Engine (`phonon-core`): `stream_start` opens a segment stream when streaming refine is on and the
correction stage is ready; `segment` events become `stream_commit`s; `stream_finish` (or, from an
older bar, `transcribe`) for that id finishes the stream. The serve loop blocks on one channel for
commands and sidecar events instead of polling every 15 ms (through the core, 20 one-shot
dictations, 2 runs: key-up-to-text p50 212/157 ms against 238/226 with the poll). The final
polish reuses the dictionary envelope of the stream's last commit when the transcript and screen
text are unchanged; building one costs 1.8 ms at 19 words and 4.6 ms at 228 (it was 5.0 and
42 ms before the transcript's phonetic window keys were computed once per request instead of
once per dictionary entry; same output on 1128 held-out transcripts).

Known limits: the dictionary envelope is rebuilt from each transcript so far, so a vocabulary
line that changes late (a new phonetic match in the last words) makes the release prefill the
whole prompt again (still lossless, slower). Batched verification and one-token decoding can
differ in the last bit of a logit, so a near-tie token can flip; the streaming output is the
greedy output of the batched forward, not bit-identical to the one-token path in every case
(39 of 40 replayed dictations identical). The ASR side shares the GPU with the background
refine: a round runs every 0.4 s over up to 80 s of audio, and a key release that lands during
a round of a long dictation waits for it (Metal: dictations of 45 s and longer p50 130-134 ms
against 28-37 ms by length band below 45 s; MLX rounds cost more, p50 295-298 ms over 45 s).

## Live text (settings.json `live_text`, off by default; on in 1.0 only)

Off by default since 2026-11-05 (owner: the text mid-dictation is a lot to look at and makes them
second-guess themselves); the default feedback is the capsule's energy. Since then the card has one
look (`live_text_style` still round-trips but is ignored, no picker): every word fades from 0.2 and
rises 3 pt in 120 ms when it first shows, as the ASR tail or as a draft; rewrites morph and glow as
below; an ember dot after the last word glows and swells with the mic level and slides 120 ms to
the new end; the card wakes on the first voice, before any word. The last word's ".", "?" or "!"
shows after 0.35 s of quiet or at key up; at the paste the card shows the pasted text at once,
holds 250 ms and fades out over 150 ms. The text below describes the 1.0 card where it differs.
Every paste (live text or not) types one trailing space (`Typer.spaced`), none after a space or
newline, so back-to-back dictations do not run together.

While the key is held, a non-activating panel just above the capsule shows the corrected text
so far; each background refine replaces it, so later words can rewrite earlier ones. It needs
streaming refine. The engine reads `live_text` at every `stream_start` (re-parsed only when
settings.json changed), so a flip applies from the next dictation. For that stream it adds
`draft_id` and `draft_seq` (the commit's index) to each `stream_commit` context; the sidecar
then adds `draftId`, `draftSeq`, `draftText` (cleaned and guarded like a run) and
`draftComplete` (false when the draft stopped at the output budget) to the reply. Without the
key the reply is unchanged. The engine sends `{"type":"result","kind":"draft","id","text",
"source","complete"}` for the open stream only, after the same `finish_correction` as the
polish result (dictionary replacements, safety fallback, approved styles); the bar applies the
same `safeFinalText` check as for the paste. At the polish result the panel shows the pasted
text, then hides with the capsule; it hides on an empty or failed pass and at the next key down.
It never becomes key and ignores the mouse; what is inserted is the polish result as before.

The card (bar/Sources/LiveText.swift) is 420 pt wide, 13.5 pt text at line height 1.45. It
is top-anchored in a panel that reserves room for five lines above the capsule: it starts one
line tall and grows down as the text wraps (height eases over 140 ms, cubic-bezier(.2,.8,.3,1)),
then scrolls to keep the newest line in view. The capsule is the usual PillView. After the
corrected text comes the uncorrected tail: the words of the latest ASR partial beyond the
draft's `source` (the transcript the engine committed for it), counted in words; a draft
covers exactly its source, so no engine field was added. At key up the caret goes and a
pending tail resolves into the polish result, which the card shows after the paste, never
before it. `live_text_style` (settings.json, "fade" default or "blur"; picker under the
toggle): fade starts new corrected words at 0.35 opacity, full in 140 ms, and dims the tail to
0.5; blur blurs the tail 1.6 pt and sharpens new words in 160 ms. A draft that changes words
already shown (word diff against the shown corrected text, `WordDiff`) morphs each changed
run: the old words dissolve (to 0 opacity, 2 pt blur, up 3 pt) while the new ones rise in from
3 pt below, over 200 ms, and the slot's width eases so neighbors slide; the new words glow
white with a #ffd2b0 halo and settle to the ink in 0.42 s. Pure appends only fade or sharpen.
Every effect is a function of time from its start, so new text is drawn on the first frame and
nothing waits on an animation; the view's timeline runs at display rate only while something
moves (10 Hz for the caret otherwise).

Measured (bundle `phonon engine` driven as the bar drives it, 10 owner dictations of 3.6-46 s
at 1x, native refiner, Metal Parakeet, off/on alternating, 2 runs each): drafts every 0.40 s
p50, 0.80 s p90; a draft is shown p50 91-117 ms, p90 206-414 ms after the end of the audio it
covers (plus up to one 0.4 s chunk); none hit the output budget. Key-up to text p50 30.9/29.3
ms off/on in the clean run (p90 68.9/71.6); the other run had ASR-side outliers in both arms
(off 1 of 10, on 4 of 10, up to 2 s) not reproduced. Final text identical on 10 of 10 in all
four arm-runs.
