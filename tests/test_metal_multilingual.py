"""Opt-in hardware check: compare the Metal decoder with MLX on local models.

Set PHONON_TEST_PA to a compiled pa binary and PHONON_TEST_PA_DIR to its shader
and mel-filter directory. Set PHONON_TEST_V2 and PHONON_TEST_V3 to model
snapshot directories. PHONON_TEST_AUDIO can name an additional 16 kHz mono WAV.
No model or audio is downloaded by this test.
"""

import os
from pathlib import Path

import pytest


@pytest.mark.parametrize("model_var", ["PHONON_TEST_V2", "PHONON_TEST_V3"])
def test_metal_tokens_match_mlx(model_var):
    pa, shader_dir, snapshot = (os.getenv(key) for key in
                                ("PHONON_TEST_PA", "PHONON_TEST_PA_DIR", model_var))
    if not all((pa, shader_dir, snapshot)):
        pytest.skip("set the local Metal helper, shader directory, and model snapshot")
    import mlx.core as mx
    from parakeet_mlx import from_pretrained

    from sidecar.asr_server import read_wav, transcribe_samples
    from sidecar.pa_client import LightModel, MetalParakeet

    model = from_pretrained(snapshot)
    mx.eval(model.parameters())
    metal = MetalParakeet(LightModel(snapshot), weights=str(Path(snapshot) / "model.safetensors"),
                          pa=pa, pa_dir=shader_dir, timeout_s=90)
    try:
        files = [Path(__file__).resolve().parents[1] / "assets/startup.wav"]
        if os.getenv("PHONON_TEST_AUDIO"):
            files.append(Path(os.environ["PHONON_TEST_AUDIO"]))
        for path in files:
            samples = read_wav(str(path), model.preprocessor_config.sample_rate)
            reference = transcribe_samples(model, samples)
            result = metal.transcribe_samples(samples)
            assert not metal.dead, "Metal helper fell back to MLX"
            result_tokens = [t.id for sentence in result.sentences for t in sentence.tokens]
            reference_tokens = [t.id for sentence in reference.sentences for t in sentence.tokens]
            assert result_tokens == reference_tokens
        assert "dictionary boosting" in metal.set_terms(["Python", "Slack"])
        metal.transcribe_samples(samples)
        assert not metal.dead
    finally:
        metal.close()
