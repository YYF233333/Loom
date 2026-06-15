import numpy as np
from scipy.signal import spectrogram as _spectrogram


def _to_numpy(audio):
    if hasattr(audio, "detach"):
        return audio.detach().cpu().numpy().astype(np.float64)
    return np.asarray(audio, dtype=np.float64)


def fundamental_freq(audio, sr):
    audio = _to_numpy(audio).flatten()
    window = np.hanning(len(audio))
    fft = np.fft.rfft(audio * window)
    magnitudes = np.abs(fft)
    freqs = np.fft.rfftfreq(len(audio), 1.0 / sr)
    peak_idx = np.argmax(magnitudes[1:]) + 1
    return float(freqs[peak_idx])


def harmonic_amplitudes(audio, sr, f0, n):
    audio = _to_numpy(audio).flatten()
    window = np.hanning(len(audio))
    fft = np.fft.rfft(audio * window)
    magnitudes = np.abs(fft)
    freqs = np.fft.rfftfreq(len(audio), 1.0 / sr)
    freq_resolution = freqs[1] - freqs[0]
    amps_db = np.zeros(n)
    for k in range(n):
        target = f0 * (k + 1)
        idx = int(round(target / freq_resolution))
        if idx < len(magnitudes):
            window = magnitudes[max(0, idx - 2):idx + 3]
            peak = window.max() if len(window) > 0 else 1e-10
        else:
            peak = 1e-10
        amps_db[k] = 20 * np.log10(max(peak, 1e-10))
    return amps_db


def spectral_centroid(audio, sr):
    audio = _to_numpy(audio).flatten()
    fft = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(len(audio), 1.0 / sr)
    total = fft.sum()
    if total < 1e-10:
        return 0.0
    return float(np.sum(freqs * fft) / total)


def spectral_rolloff(audio, sr, pct=0.85):
    audio = _to_numpy(audio).flatten()
    fft = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(len(audio), 1.0 / sr)
    cumulative = np.cumsum(fft)
    threshold = pct * cumulative[-1]
    idx = np.searchsorted(cumulative, threshold)
    return float(freqs[min(idx, len(freqs) - 1)])


def rms_envelope(audio, hop=512):
    audio = _to_numpy(audio).flatten()
    n_frames = len(audio) // hop
    env = np.zeros(n_frames)
    for i in range(n_frames):
        frame = audio[i * hop:(i + 1) * hop]
        env[i] = np.sqrt(np.mean(frame ** 2))
    return env


def thd(audio, sr, f0):
    audio = _to_numpy(audio).flatten()
    fft = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(len(audio), 1.0 / sr)
    freq_res = freqs[1] - freqs[0]

    def _peak_at(freq):
        idx = int(round(freq / freq_res))
        if idx >= len(fft):
            return 0.0
        window = fft[max(0, idx - 2):idx + 3]
        return float(window.max()) if len(window) > 0 else 0.0

    fundamental_power = _peak_at(f0) ** 2
    harmonic_power = 0.0
    for k in range(2, 20):
        harmonic_power += _peak_at(f0 * k) ** 2
    if fundamental_power < 1e-20:
        return 0.0
    return float(np.sqrt(harmonic_power / fundamental_power))


def envelope_shape(audio, sr):
    audio = _to_numpy(audio).flatten()
    # Use amplitude envelope (smoothed abs) instead of RMS to get true peak
    hop = 256
    n_frames = len(audio) // hop
    env = np.zeros(n_frames)
    for i in range(n_frames):
        frame = np.abs(audio[i * hop:(i + 1) * hop])
        env[i] = frame.max()
    hop_sec = hop / sr

    peak_idx = np.argmax(env)
    peak = float(env[peak_idx])
    attack_ms = float(peak_idx * hop_sec * 1000)

    if peak_idx + 1 < len(env):
        sustain_region = env[peak_idx + int(len(env) * 0.1):int(len(env) * 0.6)]
        sustain_level = float(np.median(sustain_region)) if len(sustain_region) > 0 else 0.0
    else:
        sustain_level = 0.0

    threshold = peak * 0.05
    release_start = len(env) - 1
    for i in range(len(env) - 1, peak_idx, -1):
        if env[i] > threshold:
            release_start = i
            break
    tail_end = len(env) - 1
    for i in range(release_start, len(env)):
        if env[i] < threshold:
            tail_end = i
            break
    release_ms = float((tail_end - release_start) * hop_sec * 1000)

    return attack_ms, peak, sustain_level, release_ms


def _hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def mel_spectrogram_distance(a, b, sr, n_mels=128, n_fft=2048, hop=512):
    a = _to_numpy(a).flatten()
    b = _to_numpy(b).flatten()
    min_len = min(len(a), len(b))
    a, b = a[:min_len], b[:min_len]

    def _mel_spec(x):
        f, t, Sxx = _spectrogram(x, fs=sr, nperseg=n_fft, noverlap=n_fft - hop)
        mel_freqs = np.linspace(0, _hz_to_mel(sr / 2), n_mels + 2)
        hz_freqs = _mel_to_hz(mel_freqs)
        filterbank = np.zeros((n_mels, len(f)))
        for i in range(n_mels):
            lo, center, hi = hz_freqs[i], hz_freqs[i + 1], hz_freqs[i + 2]
            for j, freq in enumerate(f):
                if lo <= freq <= center:
                    filterbank[i, j] = (freq - lo) / (center - lo + 1e-10)
                elif center < freq <= hi:
                    filterbank[i, j] = (hi - freq) / (hi - center + 1e-10)
        mel = filterbank @ Sxx
        return np.log(mel + 1e-10)

    return float(np.mean(np.abs(_mel_spec(a) - _mel_spec(b))))


def freq_response(filter_fn, sr, n_samples=88200, seed=42):
    np.random.seed(seed)
    noise = np.random.randn(n_samples).astype(np.float32)

    import torch
    noise_t = torch.from_numpy(noise).unsqueeze(0)
    with torch.no_grad():
        filtered_t = filter_fn(noise_t)
    filtered = filtered_t.squeeze(0).numpy()

    fft_in = np.fft.rfft(noise)
    fft_out = np.fft.rfft(filtered)
    H = fft_out / (fft_in + 1e-10)
    freqs = np.fft.rfftfreq(n_samples, 1.0 / sr)
    mag_db = 20 * np.log10(np.abs(H) + 1e-10)
    return freqs, mag_db
