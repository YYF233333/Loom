import torch
import pytest
import os
from loom.core import DEVICE

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
GOLDEN_DIR = os.path.join(FIXTURES_DIR, "golden")
REFERENCE_DIR = os.path.join(FIXTURES_DIR, "reference")


def pytest_addoption(parser):
    parser.addoption(
        "--update-golden", action="store_true", default=False,
        help="Regenerate golden audio snapshots",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "reference: tests requiring Serum reference audio files")


def pytest_report_header():
    return (
        f"loom device: {DEVICE} (CUDA {torch.version.cuda})"
        if DEVICE.type == "cuda"
        else f"loom device: {DEVICE}"
    )


@pytest.fixture
def update_golden(request):
    return request.config.getoption("--update-golden")


@pytest.fixture
def golden_dir():
    return GOLDEN_DIR


@pytest.fixture
def reference_dir():
    return REFERENCE_DIR
