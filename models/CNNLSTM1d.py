import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim=128,
                 num_layers=2,
                 dropout=0.4,
                 num_classes=16):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=5, padding=2), # hidden dim es el numero filtros
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(), #(16, 128, 1292)
            nn.MaxPool1d(2), #time ⌊1292/2⌋ = 646 -> (16, 128, 646)
            nn.Conv1d(hidden_dim, hidden_dim*2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim*2),
            nn.ReLU(), # (16, 256, 646)
            nn.MaxPool1d(2), #time ⌊646/2⌋ = 323 -> (16, 256, 323)
            nn.Conv1d(hidden_dim*2, hidden_dim*4, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim*4),
            nn.ReLU(),
            #nn.MaxPool1d(2),
            nn.Dropout(dropout), #(16, 512, 323)
        )

        # LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_dim * 4,
            hidden_size=128,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        ) #(16, 323, 128*2=256)

        # Classification
        self.classifier = nn.Sequential(
            nn.Linear(128 * 2 * 2, 128),  # 128 hidden, *2 per bidirectional, *2 per avg+max
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)  # [batch, features, time]
        x = self.feature_extractor(x)
        x = x.permute(0, 2, 1)  # [batch, time, features]

        lstm_out, (h_n, c_n) = self.lstm(x)
        avg_pool = torch.mean(lstm_out, dim=1)
        max_pool, _ = torch.max(lstm_out, dim=1)
        final = torch.cat([avg_pool, max_pool], dim=1)

        logits = self.classifier(final)
        return logits
