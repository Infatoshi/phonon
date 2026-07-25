# Phonon

Phonon is a local-first voice writing app that feels intelligent without asking users to configure intelligence.

## Product promise

Press one shortcut and speak naturally. The app understands mid-sentence corrections, removes false starts, preserves intent, and inserts polished text into the active app. A bundled Gemma 4 model performs refinement with speculative decoding. Users never choose a provider, prompt, mode, or cleanup level.

## Interaction model

The primary experience is a transient recorder pill with four states:

1. Ready
2. Listening
3. Refining
4. Done

The main window is secondary. It contains only Home, History, Dictionary, and Settings.

- Home shows readiness, the shortcut, context/privacy status, and recent transcriptions.
- History shows compact searchable rows. Selection reveals the final text and minimal metadata.
- Dictionary has two concepts under one surface: soft Terms for spelling context and exact Replacements.
- Settings contains only shortcut, active-screen context, local history, microphone, and launch behavior.

## Intelligent defaults

- Treat phrases such as “no, actually” and “I didn't mean to word it that way” as live revision intent.
- Capture selected text first, then OCR text from only the active window at recording start.
- Show a subtle `Using context from <App>` indicator while recording.
- Keep screenshots local. Only extracted text enters the local refinement pipeline.
- Fall back to context-free dictation when screen access is unavailable.

## Explicit non-goals for v0

- AI provider catalogs, API keys, model pickers, or prompt editors
- User-created modes, per-app profiles, or trigger builders
- Command chat, shell execution, and file transcription
- Analytics dashboards, milestones, teams, billing, or referrals
- Audio playback, bulk export, and performance inspection in History

## Proof-of-concept acceptance

- Native GPUI window with aligned navigation and interactive page switching
- Home, History, Dictionary, and Settings layouts render without a webview
- Visual hierarchy communicates readiness and privacy with little explanatory text
- `cargo nextest run`, `cargo fmt --check`, and Clippy pass

