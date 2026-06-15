import torch
import pytest
from loom.oscillators import AdditiveOscillator
from loom.core import SAMPLE_RATE, N_SAMPLES, DEVICE


class TestAdditiveOscillator:
    def setup_method(self):
        self.osc = AdditiveOscillator(sample_rate=SAMPLE_RATE, n_samples=N_SAMPLES).to(DEVICE)

    def test_output_shape(self):
        batch = 4
        pitch = torch.full((batch,), 0.5, device=DEVICE)
        waveform = torch.zeros(batch, 4, device=DEVICE)
        waveform[:, 0] = 1.0
        audio = self.osc(pitch, waveform)
        assert audio.shape == (batch, N_SAMPLES)

    def test_sine_frequency(self):
        """Pure sine at A4 (440Hz) should have dominant FFT peak at 440Hz."""
        batch = 1
        midi_note = 69
        pitch = torch.tensor([(midi_note - 24) / (96 - 24)], device=DEVICE)
        waveform = torch.zeros(batch, 4, device=DEVICE)
        waveform[:, 0] = 1.0
        audio = self.osc(pitch, waveform)

        fft = torch.fft.rfft(audio[0])
        magnitudes = torch.abs(fft)
        freqs = torch.fft.rfftfreq(N_SAMPLES, 1.0 / SAMPLE_RATE, device=DEVICE)
        peak_freq = freqs[torch.argmax(magnitudes[1:]) + 1]
        assert abs(peak_freq.item() - 440.0) < 2.0

    def test_amplitude_range(self):
        """Output should be roughly in [-1, 1]."""
        pitch = torch.tensor([0.5], device=DEVICE)
        waveform = torch.zeros(1, 4, device=DEVICE)
        waveform[:, 0] = 1.0
        audio = self.osc(pitch, waveform)
        assert audio.abs().max().item() <= 1.01

    def test_saw_has_harmonics(self):
        """Saw wave should have energy beyond the fundamental."""
        pitch = torch.tensor([0.3], device=DEVICE)
        waveform = torch.zeros(1, 4, device=DEVICE)
        waveform[:, 1] = 1.0
        audio = self.osc(pitch, waveform)

        fft = torch.fft.rfft(audio[0])
        magnitudes = torch.abs(fft)
        fundamental_idx = torch.argmax(magnitudes[1:]) + 1
        harmonic_energy = magnitudes[fundamental_idx * 2:].sum()
        assert harmonic_energy.item() > 0.1

    def test_detune(self):
        """Detuning should shift the peak frequency."""
        midi_note = 69
        pitch = torch.tensor([(midi_note - 24) / (96 - 24)], device=DEVICE)
        waveform = torch.zeros(1, 4, device=DEVICE)
        waveform[:, 0] = 1.0
        detune = torch.tensor([0.7], device=DEVICE)

        audio_detuned = self.osc(pitch, waveform, detune)
        fft = torch.fft.rfft(audio_detuned[0])
        magnitudes = torch.abs(fft)
        freqs = torch.fft.rfftfreq(N_SAMPLES, 1.0 / SAMPLE_RATE, device=DEVICE)
        peak_freq = freqs[torch.argmax(magnitudes[1:]) + 1]
        assert peak_freq.item() > 440.0
