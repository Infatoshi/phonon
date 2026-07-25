# Devlog

## 2026-07-20

- Started a Mac-native GPUI proof of concept.
- Reduced the product to Home, History, Dictionary, and Settings.
- Made screen context an intelligent active-window default rather than a configurable mode system.
- Kept Gemma refinement behind the single user-facing state `Refining`.
- Used a compact recorder card and progressively disclosed history/dictionary detail.
- Audited VoiceInk 2.0 and retained only its transient recorder, graceful context capture, compact history, and term-versus-replacement dictionary ideas.
- Added a uniquely identified macOS app bundle after visual verification caught a collision with an older installed Phonon build.
- Verified Home, History, Dictionary, Settings, and the context toggle in the running GPUI window.
