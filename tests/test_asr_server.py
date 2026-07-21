import base64
import struct

import pytest

from sidecar.asr_server import decode_pcm16


def test_decode_pcm16_normalizes_little_endian_samples():
    encoded = base64.b64encode(struct.pack("<hhh", -32768, 0, 32767)).decode()

    decoded = decode_pcm16(encoded)

    assert decoded.tolist() == pytest.approx([-1.0, 0.0, 32767 / 32768])


def test_decode_pcm16_rejects_invalid_base64():
    with pytest.raises(ValueError):
        decode_pcm16("not base64")
