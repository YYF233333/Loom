# tests/test_timbres_presets.py
import torch
import numpy as np
import os
import pytest
from loom.synth import SubtractiveSynth
from loom.core import SAMPLE_RATE, DEVICE
from tests.conftest import GOLDEN_DIR, REFERENCE_DIR

N_SAMPLES = SAMPLE_RATE * 4


class PresetTestBase:
    """Base class for preset golden + acoustic tests.

    Subclass and set:
        PRESET_NAME = "01_sub_bass"
        PARAMS = {...}  # full synth param dict

    Then add acoustic assertion methods.
    """

    PRESET_NAME = None
    PARAMS = None

    @pytest.fixture(autouse=True)
    def _setup(self, update_golden):
        self.synth = SubtractiveSynth(
            sample_rate=SAMPLE_RATE, n_samples=N_SAMPLES
        ).to(DEVICE)
        self.update_golden = update_golden
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

    def _golden_path(self):
        return os.path.join(GOLDEN_DIR, f"{self.PRESET_NAME}.pt")

    def test_golden_snapshot(self):
        golden_path = self._golden_path()
        if self.update_golden or not os.path.exists(golden_path):
            torch.save(self.audio.cpu(), golden_path)
            pytest.skip(f"Golden updated: {golden_path}")
        golden = torch.load(golden_path, weights_only=True).to(DEVICE)
        assert torch.allclose(self.audio, golden, atol=1e-5), (
            f"Golden mismatch for {self.PRESET_NAME}. "
            f"Max diff: {(self.audio - golden).abs().max().item():.6f}. "
            f"Run with --update-golden to regenerate."
        )

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


# --- Preset tests are added here as they are onboarded ---
# Example (uncomment and fill in when sub_bass preset is validated):
#
# class TestSubBass(PresetTestBase):
#     PRESET_NAME = "01_sub_bass"
#     PARAMS = { ... }  # full param dict
#
#     def test_fundamental_below_80hz(self):
#         from tests.timbre_helpers import fundamental_freq
#         f0 = fundamental_freq(self.audio_np, SAMPLE_RATE)
#         assert f0 < 80.0
#
#     def test_spectral_centroid_below_200hz(self):
#         from tests.timbre_helpers import spectral_centroid
#         sc = spectral_centroid(self.audio_np, SAMPLE_RATE)
#         assert sc < 200.0
