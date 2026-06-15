import torch
from loom.core import DEVICE


def pytest_report_header():
    return f"loom device: {DEVICE} (CUDA {torch.version.cuda})" if DEVICE.type == "cuda" else f"loom device: {DEVICE}"
