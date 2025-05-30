import torch
import torch.nn as nn

class Conv2D_GRU(nn.Module):
    def __init__(self,
                 input_dim,
                 time_dim,
                 hidden_dim=128,
                 num_layers=2,
                 dropout=0.4,
                 num_classes=16):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, hidden_dim, kernel_size=(5, 5), padding=(2, 2)),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(hidden_dim, hidden_dim*2, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(hidden_dim*2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(hidden_dim*2, hidden_dim*4, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(hidden_dim*4),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )

        # Dimensions despres de les capes conv (maxpool=2 dos cops)
        self.time_after_conv = time_dim // 4
        self.freq_after_conv = input_dim // 4

        gru_input_size = hidden_dim * 4 * self.freq_after_conv  # Quan s'aplana la freq.

        self.encoder = nn.GRU(
            input_size=gru_input_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout = dropout if num_layers > 1 else 0
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        # x: [batch, time, freq]
        x = x.unsqueeze(1)  # [batch, 1, time, freq]

        x = self.feature_extractor(x)  # [batch, channels, time//4, freq//4]
        x = x.permute(0, 2, 1, 3)      # [batch, time_steps, channels, freq]
        x = x.flatten(2)               # [batch, time_steps, features]

        gru_out, _ = self.encoder(x)   # [batch, time, hidden_dim*2]

        features = torch.mean(gru_out, dim=1)  # [batch, hidden_dim*2]

        logits = self.classifier(features)     # [batch, num_classes]
        return logits
