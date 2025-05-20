import torch
import torch.nn as nn
import torch.nn.functional as F

# use_bidir = True (GRU BIDIRECCIONAL)

class LSTM(nn.Module):
    def __init__(self,
                 input_dim, #(16, 1292, 128) → (16, 128, 1292)
                 hidden_dim=128,
                 num_layers=2,
                 dropout=0.4,
                 num_classes=16):
        super().__init__()

        # Ara fas servir hidden_dim, num_layers i dropout del config
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(), #→ (16, 128, 1292)
            nn.MaxPool1d(2), #time ↓ 1292 → ⌊1292/2⌋ = 646 → (16, 128, 646)
            nn.Conv1d(hidden_dim, hidden_dim*2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim*2),
            nn.ReLU(), # → (16, 256, 646)
            nn.MaxPool1d(2), #time ↓ 646 → ⌊646/2⌋ = 323 → (16, 256, 323)
            nn.Conv1d(hidden_dim*2, hidden_dim*4, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim*4),
            nn.ReLU(),
            #nn.MaxPool1d(2),
            nn.Dropout(dropout), #(16, 512, 323)
        )

        # LSTM recurrent block
        self.lstm = nn.LSTM(
            input_size=hidden_dim * 4,
            hidden_size=128,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        ) #→ (16, 323, 128)

        # Dense classification head
        self.classifier = nn.Sequential(
            nn.Linear(128 * 2 * 2, 128),  # 128 hidden, *2 for bidirectional, *2 for avg+max
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)  # [batch, features, time]
        x = self.feature_extractor(x)
        x = x.permute(0, 2, 1)  # [batch, time, features]

        lstm_out, (h_n, c_n) = self.lstm(x)  # [batch, seq_len, hidden_dim*2]
        avg_pool = torch.mean(lstm_out, dim=1)
        max_pool, _ = torch.max(lstm_out, dim=1)
        final = torch.cat([avg_pool, max_pool], dim=1)

        logits = self.classifier(final)
        return logits

"""
POSSIBLES MILLORES-> 
- Probar kernel_size=7 en la primera capa para un contexto temporal más amplio.
- Añadir otro bloque Conv1d + BN + ReLU
- Descomentar un tercer MaxPool1d(2) tras el tercer Conv.
- Añadir dropout también en las capas recurrentes (self.lstm = nn.LSTM(..., dropout=dropout))
"""
"""
Eliminamos la capa de embedding y los argumentos `prev_tokens` y `teacher_forcing_ratio` 
porque no se trata de una tarea secuencial autoregresiva, sino de clasificación directa. 
En este contexto, no se necesita representar tokens previos ni aplicar técnicas como *teacher forcing*. 
El modelo recibe un espectrograma completo y genera una única predicción, 
por lo que simplificar el `forward` lo hace más claro, eficiente y adecuado para la tarea.
"""
"""
Input                         (16, 1292, 128)          # (batch, time, mel_bins)
↓ permute  
(16, 128, 1292)               # (batch, channels, time)
↓ Conv1d(128→128)→ReLU→Pool2  
(16, 128, 646)
↓ Conv1d(128→256)→ReLU→Pool2  
(16, 256, 323)
↓ Conv1d(256→512)→ReLU→Dropout  
(16, 512, 323)
↓ permute  
(16, 323, 512)                # (batch, seq_len, input_size)
↓ LSTM(unidir, 512→128)  
lstm_out: (16, 323, 128)
↓ take last time step  
last_hidden: (16, 128)
↓ Classifier (128→64→16)  
(16, 16)                       # logits final (num_classes)

"""