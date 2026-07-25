# GPUI prototype (superseded, 2026-07-20)

The visual prototype this app was designed against. It lived at `~/dev/phonon`,
was never committed, and is archived here so the design record survives the
folder.

It was one 664-line GPUI view (`main.rs.original`, renamed so it is not built)
with a single dependency and hardcoded sample data: no audio, no ASR, no model,
no sidecar. It existed to answer one question, quoted from its own acceptance
criteria: whether a "native GPUI window with aligned navigation and interactive
page switching" could carry the product surface without a webview. The four
screenshots are that window.

The answer was yes, and the shipping app then implemented the same surface in
SwiftUI under `bar/`, so the design carried over and the code did not. Nothing
here is reachable from the built app.

`SPEC.md` is the product spec the shipping app implements: the four recorder
states, the Home / History / Dictionary / Settings surface, the Term versus
Replacement dictionary split, and the explicit non-goals (no provider catalogs,
API keys, model pickers, or prompt editors). Read it as the current intent.

`DEVLOG.md` records what the VoiceInk 2.0 audit contributed: the transient
recorder, graceful context capture when permission is absent, compact history,
and the term-versus-replacement split.

`AGENTS.md.original` is kept only as a record. It claims that folder is the
source of truth and requires GPUI, both of which describe the prototype and are
false of this repository.
