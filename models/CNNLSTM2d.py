import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN2D_LSTM(nn.Module):
    def __init__(self,
                 in_channels=1,
                 base_channels=64,
                 lstm_hidden=128,
                 num_layers=2,
                 dropout=0.4,
                 num_classes=16,
                 n_mels=128):
        super().__init__()

        # Bloc convolucional 2D
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=(5, 5), padding=(2, 2)),  # (B, 64, 1292, 128)
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),  # (B, 64, 1292/2, 128/2)=(B, 64, 645, 64)

            nn.Conv2d(base_channels, base_channels * 2, kernel_size=(3, 3), padding=(1, 1)),  # (B, 128, 645, 64)
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),  # (B, 128, 323, 32)

            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=(3, 3), padding=(1, 1)),  # (B, 256, 323, 32)
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),  # (B, 256, 323, 32)
        )

        # Calcular dimensió de frequencia despres de dos pooling (MaxPool2d((2,2)))
        freq_out = n_mels // 4  # 2**2 = 4
        # Dimensió d'entrada per LSTM: canals * freq_out
        lstm_input_size = base_channels * 4 * freq_out  # [B, T'=323, lstm_input_size= 256*32=8192]

        # LSTM
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )  # [B, T, 128*2]

        # Classificació
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2 * 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x: [batch, time, n_mels] -> [batch, 1, time, n_mels]
        x = x.unsqueeze(1)

        # Extracció de característiques 2D
        feat = self.feature_extractor(x)  # [B, C=256, T'=323, F'=32]

        # Permutar i aplastar freq.
        B, C, T, F = feat.shape
        feat = feat.permute(0, 2, 1, 3).contiguous()  # [B, T'=323, C=256, F'=32]
        feat = feat.view(B, T, C * F)  # [B, T', lstm_input_size= 256*32=8192]

        # LSTM
        lstm_out, _ = self.lstm(feat)

        # Pooling temporal (avg + max)
        avg_pool = torch.mean(lstm_out, dim=1)  # [B, hidden*2]
        max_pool, _ = torch.max(lstm_out, dim=1)  # [B, hidden*2]
        cat = torch.cat([avg_pool, max_pool], dim=1)  # [B, hidden*2*2]

        # Classificació
        logits = self.classifier(cat)
        return logits
