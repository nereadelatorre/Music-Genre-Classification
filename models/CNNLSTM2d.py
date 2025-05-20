import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN2D_LSTM(nn.Module):
    def __init__(self,
                 in_channels=1,  # espectrogram ([batch, 1, time, n_mels])
                 base_channels=64,
                 lstm_hidden=128,
                 num_layers=2,
                 dropout=0.4,
                 num_classes=16,
                 n_mels=128):  # número de bandas Mel de entrada
        super().__init__()

        # Bloque convolucional 2D
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=(5, 5), padding=(2, 2)),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(base_channels, base_channels * 2, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )

        # Calcular dimensión de frecuencia tras dos pooling (MaxPool2d((2,2)) dos veces)
        freq_out = n_mels // 4  # 2**2 = 4
        # Dimensión de entrada para la LSTM: canales * freq_out
        lstm_input_size = base_channels * 4 * freq_out

        # LSTM sobre la dimensión tempral
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Cabeza de clasificación
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2 * 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x: [batch, time, n_mels] → [batch, 1, time, n_mels]
        x = x.unsqueeze(1)

        # 1) extracción de características 2D
        feat = self.feature_extractor(x)  # [B, C, T', F']

        # 2) preparar para LSTM: permutar y aplastar freq
        B, C, T, F = feat.shape
        feat = feat.permute(0, 2, 1, 3).contiguous()  # [B, T', C, F']
        feat = feat.view(B, T, C * F)  # [B, T', lstm_input_size]

        # 3) LSTM
        lstm_out, _ = self.lstm(feat)  # [B, T', hidden*2]

        # 4) pooling temporal (avg + max)
        avg_pool = torch.mean(lstm_out, dim=1)  # [B, hidden*2]
        max_pool, _ = torch.max(lstm_out, dim=1)  # [B, hidden*2]
        cat = torch.cat([avg_pool, max_pool], dim=1)  # [B, hidden*2*2]

        # 5) clasificación
        logits = self.classifier(cat)
        return logits


"""
class LSTM2D(nn.Module):
    def __init__(self,
                 input_dim=128,    # Dimensión de características mel
                 time_dim=1292,    # Dimensión temporal del espectrograma
                 hidden_dim=64,    # Reducido de 128 a 64 para ahorrar memoria
                 num_layers=1,     # Reducido de 2 a 1 para ahorrar memoria
                 dropout=0.4,
                 num_classes=12):  # 12 géneros según la salida proporcionada
        super().__init__()

        # Reducir dimensiones en la extracción de características
        self.feature_extractor = nn.Sequential(
            # Primera capa convolucional 2D - kernels más pequeños y stride=2 para reducir dimensiones rápidamente
            nn.Conv2d(1, hidden_dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),

            # Segunda capa convolucional 2D
            nn.Conv2d(hidden_dim, hidden_dim*2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(hidden_dim*2),
            nn.ReLU(),

            # Tercera capa convolucional 2D con menos filtros
            nn.Conv2d(hidden_dim*2, hidden_dim*2, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(hidden_dim*2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Calculamos las dimensiones después de las convoluciones
        # Primera capa: stride=2 reduce a la mitad
        # Segunda capa: stride=2 reduce a la mitad nuevamente
        self.time_after_conv = time_dim // 4
        self.freq_after_conv = input_dim // 4

        # Calcular features después del extractor para determinar el tamaño de input de la LSTM
        feature_size = hidden_dim * 2 * self.freq_after_conv

        # Reducimos la cantidad de features si es necesario para la LSTM
        if feature_size > 2048:
            self.use_projection = True
            self.projection = nn.Linear(feature_size, 512)
            lstm_input_size = 512
        else:
            self.use_projection = False
            lstm_input_size = feature_size

        # LSTM recurrent block con menos unidades
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_dim,  # Reducido a hidden_dim en lugar de 128
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

        # Dense classification head con menos unidades
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2 * 2, hidden_dim),  # hidden_dim, *2 para bidireccional, *2 para avg+max pooling
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        # x shape: [batch, time, freq] o [16, 1292, 128] según la salida
        batch_size = x.size(0)

        # Añadir dimensión de canal y reorganizar para Conv2D [batch, channel, time, freq]
        x = x.unsqueeze(1)  # [batch, 1, time, freq]

        # Pasar por extractor de características convolucional 2D
        x = self.feature_extractor(x)  # [batch, hidden_dim*2, time_after_conv, freq_after_conv]

        # Reorganizar para LSTM - queremos mantener la dimensión temporal como secuencia
        # Primero transponer para que time_after_conv sea la segunda dimensión
        x = x.permute(0, 2, 1, 3)  # [batch, time_after_conv, hidden_dim*2, freq_after_conv]
        # Luego aplanar todas las features para la LSTM
        x = x.reshape(batch_size, self.time_after_conv, -1)  # [batch, time_after_conv, hidden_dim*2*freq_after_conv]

        # Proyectar a menor dimensión si es necesario
        if self.use_projection:
            x = self.projection(x)

        # Pasar por LSTM
        lstm_out, _ = self.lstm(x)  # [batch, time_after_conv, hidden_dim*2]

        # Aplicar pooling global a la salida de LSTM
        avg_pool = torch.mean(lstm_out, dim=1)  # [batch, hidden_dim*2]
        max_pool, _ = torch.max(lstm_out, dim=1)  # [batch, hidden_dim*2]

        # Concatenar para formar el vector final de características
        final = torch.cat([avg_pool, max_pool], dim=1)  # [batch, hidden_dim*2*2]

        # Clasificador final
        logits = self.classifier(final)  # [batch, num_classes]

        return logits
"""
"""
Flujo del modelo 2D optimizado para memoria:

Input                         (16, 1292, 128)          # (batch, time, freq) - forma original del dataloader
↓ unsqueeze
(16, 1, 1292, 128)            # (batch, channels, time, freq) - añadimos canal para Conv2D
↓ Conv2d(1→64) stride=2 + ReLU 
(16, 64, 646, 64)             # time/2, freq/2 usando stride en lugar de maxpool
↓ Conv2d(64→128) stride=2 + ReLU
(16, 128, 323, 32)            # time/4, freq/4 usando stride en lugar de maxpool
↓ Conv2d(128→128) + ReLU + Dropout  
(16, 128, 323, 32)            # Mantenemos el mismo número de filtros (en lugar de duplicar)
↓ Reshape para LSTM (permute y reshape)
(16, 323, 128*32)             # (batch, time_after_conv, hidden_dim*2*freq_after_conv)
↓ Proyección lineal (opcional si el tensor es muy grande)
(16, 323, 512)                # Si es necesario, reduce dimensionalidad
↓ LSTM bidireccional (1 capa en lugar de 2)
lstm_out: (16, 323, 64*2)     # (batch, time_after_conv, hidden_dim*2) (bidireccional)
↓ Global pooling (avg + max)
(16, 64*2*2)                  # (batch, hidden_dim*2*2)
↓ Classifier (más pequeño)
(16, 12)                      # logits final (num_classes=12 para los géneros musicales)

Cambios para optimizar memoria:
1. Reducción de hidden_dim de 128 a 64
2. Uso de stride=2 en convoluciones en lugar de MaxPool2d
3. Menos capas en LSTM (1 en lugar de 2)
4. Capa de proyección opcional para reducir dimensionalidad
5. Tercer bloque convolucional mantiene el mismo número de filtros
"""