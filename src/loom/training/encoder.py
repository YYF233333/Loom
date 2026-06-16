import torch
import torch.nn as nn

from loom.training.dataset import N_CONTINUOUS, N_PARAMS, N_ROUTING, CATEGORICAL_KEYS


class ParamEncoder(nn.Module):
    """CNN encoder: mel spectrogram -> synth parameter vector."""

    def __init__(self, n_mels: int = 128):
        super().__init__()
        self.n_continuous = N_CONTINUOUS
        self.n_routing = N_ROUTING
        self.categorical_groups = CATEGORICAL_KEYS

        self.backbone = nn.Sequential(
            nn.Conv1d(n_mels, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, 3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, N_PARAMS),
        )

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        x = self.backbone(mel).squeeze(-1)
        x = self.head(x)

        continuous = torch.sigmoid(x[:, :self.n_continuous])

        cats = []
        idx = self.n_continuous
        for _, n in self.categorical_groups:
            cats.append(torch.softmax(x[:, idx:idx + n], dim=-1))
            idx += n

        routing = x[:, idx:idx + self.n_routing]

        return torch.cat([continuous] + cats + [routing], dim=1)
