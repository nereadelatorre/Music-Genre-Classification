import torch
import torch.nn as nn
import torch.nn.functional as F


class Baseline(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim=128,
                 num_layers=2,
                 dropout=0.4,
                 num_classes=16):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(hidden_dim * 2, hidden_dim * 4, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.encoder = nn.GRU(
            hidden_dim * 4,
            hidden_dim * 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        # Processem audio features
        x = x.permute(0, 2, 1)  # [batch, features, time]
        x = self.feature_extractor(x)
        x = x.permute(0, 2, 1)  # [batch, time, features]

        # GRU
        gru_out, hidden = self.encoder(x)

        features = torch.mean(gru_out, dim=1)

        # CLASSIFIER
        logits = self.classifier(features)

        return logits