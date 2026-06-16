import torch
import numpy as np
import pytest
import os
from scipy.io import wavfile
from loom.core import SAMPLE_RATE, DEVICE

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
REFERENCE_DIR = os.path.join(FIXTURES_DIR, "reference")
TEST_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def pytest_configure(config):
    config.addinivalue_line("markers", "reference: tests requiring Serum reference audio files")
    os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)


def pytest_report_header():
    return (
        f"loom device: {DEVICE} (CUDA {torch.version.cuda})"
        if DEVICE.type == "cuda"
        else f"loom device: {DEVICE}"
    )


@pytest.fixture
def reference_dir():
    return REFERENCE_DIR


def save_test_wav(audio, name, sample_rate=SAMPLE_RATE):
    """Save test-generated audio to tests/output/ for manual inspection."""
    if hasattr(audio, "detach"):
        audio = audio.detach().cpu().float().numpy()
    audio = np.asarray(audio, dtype=np.float64).flatten()
    peak = np.abs(audio).max()
    if peak > 1e-8:
        audio = audio / peak * 0.9
    audio_16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
    path = os.path.join(TEST_OUTPUT_DIR, f"{name}.wav")
    wavfile.write(path, sample_rate, audio_16)
    return path
