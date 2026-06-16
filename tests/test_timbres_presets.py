# tests/test_timbres_presets.py
import torch
import numpy as np
import os
import pytest
from loom.synth import SubtractiveSynth
from loom.core import SAMPLE_RATE, DEVICE
from tests.conftest import REFERENCE_DIR, save_test_wav

N_SAMPLES = SAMPLE_RATE * 4


class PresetTestBase:
    """Base class for preset acoustic tests with audio output.

    Subclass and set:
        PRESET_NAME = "01_sub_bass"
        PARAMS = {...}  # full synth param dict

    Then add acoustic assertion methods.
    Rendered audio is saved to tests/output/ for manual inspection.
    """

    PRESET_NAME = None
    PARAMS = None

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.synth = SubtractiveSynth(
            sample_rate=SAMPLE_RATE, n_samples=N_SAMPLES
        ).to(DEVICE)
        self._audio = None

    @property
    def audio(self):
        if self._audio is None:
            params = {k: v.to(DEVICE) for k, v in self.PARAMS.items()}
            with torch.no_grad():
                self._audio = self.synth(params)
        return self._audio

    @property
    def audio_np(self):
        return self.audio[0].cpu().numpy()

    def test_save_audio(self):
        """Save rendered audio for manual inspection."""
        path = save_test_wav(self.audio[0], self.PRESET_NAME)
        assert os.path.exists(path)

    def _reference_path(self):
        return os.path.join(REFERENCE_DIR, f"serum_{self.PRESET_NAME}.wav")

    @pytest.mark.reference
    def test_reference_comparison(self):
        ref_path = self._reference_path()
        if not os.path.exists(ref_path):
            pytest.skip(f"Reference not found: {ref_path}")
        from scipy.io import wavfile
        from tests.timbre_helpers import mel_spectrogram_distance
        sr, ref_audio = wavfile.read(ref_path)
        ref_audio = ref_audio.astype(np.float32) / 32768.0
        if sr != SAMPLE_RATE:
            pytest.skip(f"Sample rate mismatch: {sr} != {SAMPLE_RATE}")
        distance = mel_spectrogram_distance(self.audio_np, ref_audio, SAMPLE_RATE)
        assert distance < 0.2, (
            f"Mel-spec distance to Serum reference: {distance:.4f} (threshold: 0.2)"
        )
