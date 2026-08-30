# SPEC

## User vocabulary onboarding (the user world model)

Goal: a new user gets a dictionary and a short profile that fit them on day
one, without reading their files by hand and without any data leaving the
machine. Parakeet and Gemma stay as shipped; the profile and dictionary carry
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
      first POC because it lived inside paths (`/home/elliotarledge`).
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
   forms). Gemma reads them every pass.
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

### Fine-tuning POC (later, anvil RTX PRO 6000)

Only if the profile plus dictionary plateaus. Two candidates, both from
data the app already keeps:
- Gemma polish on pairs (raw ASR transcript, what the user finally kept).
  Needs post-insert edit capture first; the corpus stores raw and final
  but not the user's later edits.
- Parakeet on corpus audio with NeMo, evaluated on held-out corpus WER.
  Risk: microphone drift after a one-time tune. Prefer the LLM side.
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
