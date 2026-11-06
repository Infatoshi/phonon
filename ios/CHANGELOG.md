# Changelog

## 0.1.0 (build 1)

First release of Phonon for iPhone: the app and a dictation keyboard running Phonon's speech
recognition and cleanup models on the phone.

- App: setup in five screens (microphone, model download, keyboard and Full Access), in-app
  dictation with copy and recent dictations, settings (writing style, cleanup on or off, auto
  insert, how long the mic stays ready, vocabulary, delete models, diagnostics and a self-test,
  licenses).
- Keyboard: mic, stop, cancel, insert last, globe, space, repeating delete, return. The first tap
  opens Phonon to start the mic; later taps within the ready window record without leaving the
  current app.
- Pipeline: 16 kHz capture, Parakeet TDT 0.6B v2 on Core ML (CPU and Neural Engine), then the
  Phonon refiner 1.7B (Q4_K_S) on llama.cpp with the Mac app's prompt, token budget, word-count
  guard and term restore. The raw transcript is inserted if cleanup fails.
- Models: 22 files, 1.45 GB, pinned Hugging Face revisions, sha256 per file, resumable download.
- Resilience: calls and route changes keep what was said; audio is journaled to disk, so a
  dictation cut off by a crash or a kill is finished on the next launch and offered under Insert
  last.
- Privacy: the model download is the only network request, so audio and text stay on the phone.
- Tests: 11 PhononCore unit tests, simulator fixture tests through both models, and a simulator
  keyboard flow test in Messages.
