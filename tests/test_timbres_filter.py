import torch
import numpy as np
import math
import pytest
from loom.svfilter import SVFilter
from loom.core import SAMPLE_RATE, N_SAMPLES, DEVICE
from tests.timbre_helpers import spectral_centroid


def _cutoff_norm(hz):
    """Convert Hz to normalized [0,1] for SVFilter."""
    log_min = math.log(20.0)
    log_max = math.log(20000.0)
    return (math.log(hz) - log_min) / (log_max - log_min)


def _q_norm(q_val):
    """Convert Q value to normalized [0,1] for SVFilter."""
    log_min = math.log(0.5)
    log_max = math.log(20.0)
    return (math.log(q_val) - log_min) / (log_max - log_min)


def _freq_response_smooth(filter_fn, sr, n_samples=88200, seed=42, smooth_window=51):
    """Compute smoothed frequency response for chunk-based filters."""
    np.random.seed(seed)
    noise = np.random.randn(n_samples).astype(np.float32)

    noise_t = torch.from_numpy(noise).unsqueeze(0)
    with torch.no_grad():
        filtered_t = filter_fn(noise_t)
    filtered = filtered_t.squeeze(0).numpy()

    fft_in = np.fft.rfft(noise)
    fft_out = np.fft.rfft(filtered)

    mag_in = np.abs(fft_in)
    mag_out = np.abs(fft_out)

    # Avoid division by near-zero bins — use magnitude ratio
    safe = mag_in > (mag_in.max() * 1e-4)
    ratio = np.ones_like(mag_in)
    ratio[safe] = mag_out[safe] / mag_in[safe]
    ratio[~safe] = 0.0

    mag_db = 20 * np.log10(ratio + 1e-10)

    # Smooth to remove noise
    kernel = np.ones(smooth_window) / smooth_window
    mag_db = np.convolve(mag_db, kernel, mode="same")

    freqs = np.fft.rfftfreq(n_samples, 1.0 / sr)
    return freqs, mag_db


def _find_3db_point(freqs, mag_db, direction="lowpass"):
    """Find -3dB cutoff frequency from a frequency response curve."""
    n = len(mag_db)
    if direction == "lowpass":
        # Use low-frequency bins (100-400 Hz range) for passband estimate
        lo = max(1, np.searchsorted(freqs, 100))
        hi = max(lo + 1, np.searchsorted(freqs, 400))
        passband = np.mean(mag_db[lo:hi])
    else:
        # Use high-frequency bins (15-20 kHz range) for passband estimate
        lo = np.searchsorted(freqs, 15000)
        hi = min(n, np.searchsorted(freqs, 20000))
        passband = np.mean(mag_db[lo:hi])
    target = passband - 3.0
    if direction == "lowpass":
        for i in range(len(mag_db) - 1):
            if mag_db[i] >= target and mag_db[i + 1] < target:
                frac = (target - mag_db[i]) / (mag_db[i + 1] - mag_db[i] + 1e-10)
                return freqs[i] + frac * (freqs[i + 1] - freqs[i])
    else:
        for i in range(len(mag_db) - 1, 0, -1):
            if mag_db[i] >= target and mag_db[i - 1] < target:
                frac = (target - mag_db[i]) / (mag_db[i - 1] - mag_db[i] + 1e-10)
                return freqs[i] + frac * (freqs[i - 1] - freqs[i])
    return 0.0


class TestSVFilterDenorm:
    def setup_method(self):
        self.filt = SVFilter(sample_rate=SAMPLE_RATE).to(DEVICE)

    @pytest.mark.parametrize("norm,expected", [
        (0.0, 20.0), (1.0, 20000.0), (0.5, 632.5),
    ])
    def test_cutoff_denorm(self, norm, expected):
        actual = self.filt._denorm_cutoff(torch.tensor([norm])).item()
        assert abs(actual - expected) / expected < 0.01

    @pytest.mark.parametrize("norm,expected", [
        (0.0, 0.5), (1.0, 20.0),
    ])
    def test_q_denorm(self, norm, expected):
        actual = self.filt._denorm_q(torch.tensor([norm])).item()
        assert abs(actual - expected) / expected < 0.01


class TestSVFilterFreqResponse:
    def setup_method(self):
        self.filt = SVFilter(sample_rate=SAMPLE_RATE).to(DEVICE)

    def _make_filter_fn(self, cutoff_hz, q_val, filter_type_vec):
        cutoff_n = _cutoff_norm(cutoff_hz)
        q_n = _q_norm(q_val)
        filt = self.filt

        def fn(x):
            x = x.to(DEVICE)
            n = x.shape[1]
            cutoff = torch.full((1, n), cutoff_n, device=DEVICE)
            q = torch.tensor([q_n], device=DEVICE)
            ft = torch.tensor([filter_type_vec], device=DEVICE, dtype=torch.float32)
            return filt(x, cutoff, q, ft).cpu()
        return fn

    def test_lp_cutoff_1000hz(self):
        fn = self._make_filter_fn(1000.0, 0.707, [1, 0, 0])
        freqs, mag = _freq_response_smooth(fn, SAMPLE_RATE)
        cutoff = _find_3db_point(freqs, mag, "lowpass")
        assert abs(cutoff - 1000.0) < 100.0, f"LP -3dB at {cutoff:.0f}Hz, expected 1000Hz"

    def test_lp_rolloff_slope(self):
        fn = self._make_filter_fn(1000.0, 0.707, [1, 0, 0])
        freqs, mag = _freq_response_smooth(fn, SAMPLE_RATE)
        idx_2k = np.searchsorted(freqs, 2000)
        idx_8k = np.searchsorted(freqs, 8000)
        if idx_2k < len(mag) and idx_8k < len(mag):
            octaves = np.log2(8000 / 2000)
            slope = (mag[idx_8k] - mag[idx_2k]) / octaves
            assert -16 < slope < -10, f"LP slope: {slope:.1f} dB/oct, expected ~-12"

    def test_hp_cutoff_1000hz(self):
        fn = self._make_filter_fn(1000.0, 0.707, [0, 1, 0])
        freqs, mag = _freq_response_smooth(fn, SAMPLE_RATE)
        cutoff = _find_3db_point(freqs, mag, "highpass")
        assert abs(cutoff - 1000.0) < 100.0, f"HP -3dB at {cutoff:.0f}Hz, expected 1000Hz"

    def test_bp_center_freq(self):
        fn = self._make_filter_fn(1000.0, 2.0, [0, 0, 1])
        freqs, mag = _freq_response_smooth(fn, SAMPLE_RATE)
        peak_idx = np.argmax(mag[1:]) + 1
        peak_freq = freqs[peak_idx]
        assert abs(peak_freq - 1000.0) < 100.0

    def test_q_resonance_peak(self):
        fn_low_q = self._make_filter_fn(1000.0, 0.707, [1, 0, 0])
        fn_high_q = self._make_filter_fn(1000.0, 10.0, [1, 0, 0])
        _, mag_low = _freq_response_smooth(fn_low_q, SAMPLE_RATE)
        _, mag_high = _freq_response_smooth(fn_high_q, SAMPLE_RATE)
        assert mag_high.max() > mag_low.max() + 3.0

    def test_bp_bandwidth_narrows_with_q(self):
        fn_low_q = self._make_filter_fn(1000.0, 1.0, [0, 0, 1])
        fn_high_q = self._make_filter_fn(1000.0, 8.0, [0, 0, 1])
        freqs, mag_low = _freq_response_smooth(fn_low_q, SAMPLE_RATE)
        _, mag_high = _freq_response_smooth(fn_high_q, SAMPLE_RATE)

        def _bandwidth(m):
            peak = m.max()
            above = freqs[m > peak - 3.0]
            return above[-1] - above[0] if len(above) > 1 else 0

        bw_low = _bandwidth(mag_low)
        bw_high = _bandwidth(mag_high)
        assert bw_high < bw_low

    def test_time_varying_cutoff_sweep(self):
        torch.manual_seed(42)
        noise = torch.randn(1, N_SAMPLES, device=DEVICE)
        cutoff = torch.linspace(0.1, 0.9, N_SAMPLES, device=DEVICE).unsqueeze(0)
        q = torch.tensor([0.3], device=DEVICE)
        ft = torch.tensor([[1.0, 0.0, 0.0]], device=DEVICE)
        with torch.no_grad():
            filtered = self.filt(noise, cutoff, q, ft)[0].cpu().numpy()
        half = N_SAMPLES // 2
        sc_first = spectral_centroid(filtered[:half], SAMPLE_RATE)
        sc_second = spectral_centroid(filtered[half:], SAMPLE_RATE)
        assert sc_second > sc_first

    def test_extreme_params_stable(self):
        signal = torch.randn(1, N_SAMPLES, device=DEVICE)
        cutoff = torch.full((1, N_SAMPLES), 0.01, device=DEVICE)
        q = torch.tensor([0.99], device=DEVICE)
        ft = torch.tensor([[1.0, 0.0, 0.0]], device=DEVICE)
        with torch.no_grad():
            out = self.filt(signal, cutoff, q, ft)
        assert not torch.isnan(out).any()
        assert out.abs().max().item() < 100.0


class TestFilterEnvelopeInteraction:
    def setup_method(self):
        from loom.synth import SubtractiveSynth
        self.synth = SubtractiveSynth(
            sample_rate=SAMPLE_RATE, n_samples=N_SAMPLES
        ).to(DEVICE)

    def _base_params(self):
        return {
            "osc_pitch": torch.tensor([0.4], device=DEVICE),
            "osc_waveform": torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=DEVICE),
            "osc_detune": torch.full((1,), 0.5, device=DEVICE),
            "osc_type": torch.tensor([[1.0, 0.0, 0.0]], device=DEVICE),
            "wt_position": torch.full((1,), 0.5, device=DEVICE),
            "fm_carrier_ratio": torch.full((1,), 0.0, device=DEVICE),
            "fm_mod_ratio": torch.full((1,), 0.0, device=DEVICE),
            "fm_mod_index": torch.full((1,), 0.0, device=DEVICE),
            "lfo_rate": torch.full((1,), 0.5, device=DEVICE),
            "lfo_depth": torch.full((1,), 0.0, device=DEVICE),
            "lfo_waveform": torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=DEVICE),
            "lfo_target": torch.zeros(1, 4, device=DEVICE),
            "lfo_phase": torch.full((1,), 0.0, device=DEVICE),
            "amp_attack": torch.full((1,), 0.1, device=DEVICE),
            "amp_decay": torch.full((1,), 0.3, device=DEVICE),
            "amp_sustain": torch.full((1,), 0.9, device=DEVICE),
            "amp_release": torch.full((1,), 0.3, device=DEVICE),
            "filter_cutoff": torch.full((1,), 0.4, device=DEVICE),
            "filter_q": torch.full((1,), 0.3, device=DEVICE),
            "filter_type": torch.tensor([[1.0, 0.0, 0.0]], device=DEVICE),
            "filt_env_attack": torch.full((1,), 0.2, device=DEVICE),
            "filt_env_decay": torch.full((1,), 0.4, device=DEVICE),
            "filt_env_sustain": torch.full((1,), 0.3, device=DEVICE),
            "filt_env_release": torch.full((1,), 0.3, device=DEVICE),
            "filt_env_amount": torch.full((1,), 0.5, device=DEVICE),
            "dist_amount": torch.full((1,), 0.0, device=DEVICE),
            "dist_mix": torch.full((1,), 0.0, device=DEVICE),
            "master_gain": torch.full((1,), 0.85, device=DEVICE),
            "comp_threshold": torch.full((1,), 0.5, device=DEVICE),
            "comp_ratio": torch.full((1,), 0.3, device=DEVICE),
            "comp_attack": torch.full((1,), 0.5, device=DEVICE),
            "comp_release": torch.full((1,), 0.5, device=DEVICE),
            "comp_makeup": torch.full((1,), 0.0, device=DEVICE),
            "comp_mix": torch.full((1,), 0.0, device=DEVICE),
            "chorus_rate": torch.full((1,), 0.5, device=DEVICE),
            "chorus_depth": torch.full((1,), 0.5, device=DEVICE),
            "chorus_mix": torch.full((1,), 0.0, device=DEVICE),
            "delay_time": torch.full((1,), 0.5, device=DEVICE),
            "delay_feedback": torch.full((1,), 0.3, device=DEVICE),
            "delay_mix": torch.full((1,), 0.0, device=DEVICE),
            "reverb_room_size": torch.full((1,), 0.5, device=DEVICE),
            "reverb_decay": torch.full((1,), 0.5, device=DEVICE),
            "reverb_damping": torch.full((1,), 0.3, device=DEVICE),
            "reverb_mix": torch.full((1,), 0.0, device=DEVICE),
            "eq_low_gain": torch.full((1,), 0.5, device=DEVICE),
            "eq_mid_gain": torch.full((1,), 0.5, device=DEVICE),
            "eq_high_gain": torch.full((1,), 0.5, device=DEVICE),
        }

    def test_positive_env_opens_filter(self):
        params = self._base_params()
        params["filt_env_amount"] = torch.tensor([0.8], device=DEVICE)
        with torch.no_grad():
            audio = self.synth(params)[0].cpu().numpy()
        attack_sc = spectral_centroid(audio[:N_SAMPLES // 4], SAMPLE_RATE)
        sustain_sc = spectral_centroid(audio[N_SAMPLES // 3:2 * N_SAMPLES // 3], SAMPLE_RATE)
        assert attack_sc > sustain_sc, (
            f"Positive env: attack SC {attack_sc:.0f} should > sustain SC {sustain_sc:.0f}"
        )

    def test_negative_env_closes_filter(self):
        params = self._base_params()
        params["filt_env_amount"] = torch.tensor([0.2], device=DEVICE)
        with torch.no_grad():
            audio = self.synth(params)[0].cpu().numpy()
        attack_sc = spectral_centroid(audio[:N_SAMPLES // 4], SAMPLE_RATE)
        sustain_sc = spectral_centroid(audio[N_SAMPLES // 3:2 * N_SAMPLES // 3], SAMPLE_RATE)
        assert attack_sc < sustain_sc, (
            f"Negative env: attack SC {attack_sc:.0f} should < sustain SC {sustain_sc:.0f}"
        )
