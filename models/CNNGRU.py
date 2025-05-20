import torch
import torch.nn as nn
import torch.nn.functional as F


# use_bidir = True (GRU BIDIRECCIONAL)
class Baseline(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim=128,
                 num_layers=2,
                 dropout=0.4,
                 num_classes=16):
        super().__init__()

        # Ara fas servir hidden_dim, num_layers i dropout del config
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.MaxPool1d(2),  # es com (kernel_size=2, stride=2),
            nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(hidden_dim * 2, hidden_dim * 4, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # El GRU ara rep num_layers i bidireccional
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

    def forward(self, x):  # prev_tokens, teacher_forcing_ratio=0.5):
        """
        Args:
            x: Input audio features (batch, time, feature)
            prev_tokens: Target tokens for teacher forcing (batch, sequence_len)
            teacher_forcing_ratio: Probability of using teacher forcing

        Args:
            x: Tensor [batch_size, time, features]
        Returns:
            logits: Tensor [batch_size, num_classes]
        """
        # batch_size = x.size(0)
        # max_len = prev_tokens.size(1)

        # Process audio features
        x = x.permute(0, 2, 1)  # [batch, features, time]
        x = self.feature_extractor(x)
        x = x.permute(0, 2, 1)  # [batch, time, features]

        # GRU
        gru_out, hidden = self.encoder(x)

        # Opción 1: usar última capa oculta (si no es bidireccional, solo hidden[-1])
        # if self.use_bidir:
        # hidden tiene forma [2, batch, hidden_size] → concat ambos sentidos
        # final_hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)  # [batch, 2*hidden]
        # else:

        # features = hidden[-1]  # [batch, hidden], coge la uLtima representación del GRU (resumen final del audio)

        # features = torch.cat((hidden[0], hidden[1]), dim=1)  # [batch, 2*hidden_dim]

        features = torch.mean(gru_out, dim=1)  # [batch, hidden_dim * 2]

        logits = self.classifier(features)

        return logits


"""
Eliminamos la capa de embedding y los argumentos `prev_tokens` y `teacher_forcing_ratio` 
porque no se trata de una tarea secuencial autoregresiva, sino de clasificación directa. 
En este contexto, no se necesita representar tokens previos ni aplicar técnicas como *teacher forcing*. 
El modelo recibe un espectrograma completo y genera una única predicción, 
por lo que simplificar el `forward` lo hace más claro, eficiente y adecuado para la tarea.
"""