import numpy as np

from sidecar.stream_asr import SCHEME, StreamTranscriber, select
from sidecar.stream_vad import FRAME, StreamVAD

RATE = 16000
CHUNK = 6400  # 0.4 s, the app's stream chunk


def speech(seconds):
    # a tone with a 4 Hz syllable envelope: like speech, it dips between syllables, so the VAD's
    # noise floor (the quietest tenth of the frames) stays at the room level
    t = np.arange(int(seconds * RATE)) / RATE
    envelope = 0.15 + 0.85 * np.abs(np.sin(2 * np.pi * 2 * t))
    return (0.3 * envelope * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def silence(seconds):
    # room noise: the VAD threshold is relative to the quietest tenth of the frames
    rng = np.random.default_rng(int(seconds * 1000))
    return (0.001 * rng.standard_normal(int(seconds * RATE))).astype(np.float32)


def dictation(*parts):
    """Speech and pause lengths alternating, after a second of room noise (key down)."""
    out = [silence(1.0)]
    for i, seconds in enumerate(parts):
        out.append(speech(seconds) if i % 2 == 0 else silence(seconds))
    return np.concatenate(out)


class FakeASR:
    """One word per 0.25 s of loud audio, named by its absolute time, so a word is the same
    whichever window hears it; calls are recorded."""

    def __init__(self, audio):
        self.audio = audio
        self.calls = []

    def __call__(self, samples, offset_s):
        self.calls.append((offset_s, len(samples) / RATE))
        words = []
        step = int(0.25 * RATE)
        start = round(offset_s * RATE)
        first = -(-start // step) * step  # word grid aligned to absolute time
        for pos in range(first, start + len(samples) - step + 1, step):
            if np.abs(self.audio[pos : pos + step]).mean() > 0.05:
                words.append((f"w{pos // step}", pos / RATE))
        return words


def stream(audio, **kw):
    asr = FakeASR(audio)
    st = StreamTranscriber(asr, **kw)
    outs = []
    full = len(audio) // CHUNK * CHUNK  # the last partial chunk only reaches finish()
    for off in range(0, full, CHUNK):
        outs += st.feed(audio[off : off + CHUNK])
    return st, asr, outs, st.finish(audio[full:])


def whole(audio):
    return " ".join(w for w, _ in FakeASR(audio)(audio, 0.0))


def test_stream_vad_matches_any_chunking():
    audio = dictation(2, 0.8, 1.5, 0.6, 1)
    a, b = StreamVAD(0.5), StreamVAD(0.5)
    cuts_a = a.feed(audio)
    cuts_b = []
    for off in range(0, len(audio), 1000):
        cuts_b += b.feed(audio[off : off + 1000])
    assert cuts_a == cuts_b and len(cuts_a) == 2
    assert a.mask == b.mask


def test_final_transcript_equals_whole_recording_words():
    audio = dictation(2, 0.8, 3, 0.7, 2.2)
    st, _asr, outs, final = stream(audio)
    assert final == whole(audio)
    assert len(st.pieces) == 2
    # every transcript so far is a prefix of the final one in this noise-free case
    assert outs and all(final.startswith(o) for o in outs)


def test_release_transcribes_a_short_window_only():
    audio = dictation(3, 0.8, 6.1)
    _st, asr, _outs, final = stream(audio, release_before_s=1.0, release_overlap_s=0.5)
    assert final == whole(audio)
    _offset, seconds = asr.calls[-1]
    assert seconds < 2.0  # about release_before + overlap + the unchunked tail, not 6 s


def test_without_rounds_release_covers_the_open_segment():
    audio = dictation(2, 0.8, 2.4)
    _st, asr, _outs, final = stream(audio, round_every=0)
    assert final == whole(audio)
    assert asr.calls[-1][1] > 2.4


def test_folding_queued_chunks_keeps_the_transcript():
    audio = dictation(2, 0.8, 3, 0.7, 2.2)
    asr = FakeASR(audio)
    st = StreamTranscriber(asr)
    full = len(audio) // CHUNK * CHUNK
    for i, off in enumerate(range(0, full, CHUNK)):
        st.feed(
            audio[off : off + CHUNK], work=i % 3 == 2
        )  # behind: work on every third chunk
    assert st.finish(audio[full:]) == whole(audio)
    assert len(asr.calls) < full // CHUNK


def test_soft_cuts_keep_rounds_short_without_pauses():
    audio = dictation(14)  # no pause for 14 s
    st, asr, _outs, final = stream(audio, soft_max_s=4.0, soft_overlap_s=1.0)
    assert final == whole(audio)
    assert st.soft_cuts >= 2
    # every round heard at most before_s + soft_max_s + one chunk of audio
    assert max(seconds for _, seconds in asr.calls[:-1]) <= 4.0 + 4.0 + 0.45


def test_rounds_keep_running_where_the_vad_hears_no_more_speech():
    # 40 s without a pause: the VAD's noise floor rises to speech level and it stops hearing
    # speech, but rounds continue to the end, so the release window stays short
    audio = dictation(40)
    st, asr, _outs, final = stream(audio, soft_max_s=10.0)
    assert final == whole(audio)
    assert not st.vad.has_speech(st.frames() - 250, st.frames())
    assert st.last_round[1] >= st.frames() - 25
    # the release: release_before_s + release_overlap_s + the unchunked tail
    assert asr.calls[-1][1] < 1.0 + 0.5 + 0.45


def test_the_app_scheme_rounds_over_the_recording_so_far_in_windows_under_82_s():
    audio = dictation(70)
    st, asr, outs, final = stream(audio, **SCHEME)
    assert final == whole(audio) and outs
    rounds, release = asr.calls[:-1], asr.calls[-1]
    # until 60 s of pending audio every round transcribes the whole recording so far
    assert all(offset == 0.0 for offset, seconds in rounds if offset + seconds <= 60.0)
    assert min(offset for offset, seconds in rounds if offset + seconds > 61.0) > 30.0
    assert st.soft_cuts == 1
    # before_s + soft_max_s + one chunk, inside Metal Parakeet's 82 s
    assert max(seconds for _, seconds in rounds) <= 20.0 + 60.0 + 0.45
    assert release[1] <= 8.0 + 2.0 + 0.45


def test_a_pause_before_key_up_releases_the_last_round_without_transcribing():
    audio = dictation(3.0, 0.8)  # the speaker stops 0.8 s before letting go
    asr = FakeASR(audio)
    st = StreamTranscriber(asr, **SCHEME)
    full = len(audio) // CHUNK * CHUNK
    for off in range(0, full, CHUNK):
        st.feed(audio[off : off + CHUNK])
    calls_before_release = len(asr.calls)
    assert st.finish(audio[full:]) == whole(audio) and st.reused
    assert len(asr.calls) == calls_before_release  # finish() ran no ASR


def test_speech_up_to_key_up_still_gets_the_release_window():
    audio = dictation(3.0)
    st, asr, _, final = stream(audio, **SCHEME)
    assert not st.reused and final == whole(audio)
    offset, seconds = asr.calls[-1]
    # the release reached the end of the audio
    assert offset + seconds == len(audio) / RATE


def test_a_soft_word_after_the_last_round_blocks_the_reuse():
    # 4.8 s of whole chunks (the last round ends there), then a word ~24 dB under the speech in
    # the part only finish() sees: the VAD hears it even though it is under the fixed level
    audio = np.concatenate([dictation(3.0, 0.8), 0.06 * speech(0.2), silence(0.1)])
    assert len(audio) % CHUNK == int(0.3 * RATE)
    st, _, _, _ = stream(audio, **SCHEME)
    assert not st.reused


def test_rounds_pause_while_only_silence_follows_the_last_one():
    audio = dictation(2.0, 2.0)
    st, _, _, final = stream(audio, **SCHEME)
    assert st.skipped_rounds >= 3 and st.reused and final == whole(audio)
    off = StreamTranscriber(FakeASR(audio), **dict(SCHEME, quiet_reuse_s=0.0))
    assert off.quiet_f < 0 and not off.quiet_skip


def test_select_keeps_words_starting_in_range():
    words = [("a", 0.10), ("b", 0.50), ("c", 1.00)]
    assert select(words, 0, 25) == "a"  # frames: [0 s, 0.5 s)
    assert select(words, 25, 100) == "b c"
    assert FRAME == 320
