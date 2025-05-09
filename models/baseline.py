import torch
import torch.nn as nn
import torch.nn.functional as F

class Baseline(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, embedding_dim=32, num_classes=8, sampled_features_from_processor= 128):
        super().__init__()
        
        # Simple feature extraction
        # Procesa secuencias (como espectrogramas) para extraer características
        self.feature_extractor = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        
        # Token embedding for previous predictions
        self.embedding = nn.Embedding(num_classes, embedding_dim)
        
        # Encoder RNN
        # Modela la secuencia en el tiempo (como una RNN)
        ## It is 80 because is the sample form the Image processor, change it when you change other things
        """
        self.encoder = nn.GRU(sampled_features_from_processor, hidden_dim, batch_first=True)
        """
        self.encoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

        # Output layer
        # Predice la clase (en este caso, el género musical)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, prev_tokens=None, teacher_forcing_ratio=0.5):
        """
        Args:
            x: Input audio features (batch, time, feature)
            prev_tokens: Target tokens for teacher forcing (batch, sequence_len)
            teacher_forcing_ratio: Probability of using teacher forcing
        """
        batch_size = x.size(0)
        #max_len = prev_tokens.size(1)

        # In FMA dataset, x should be (batch, time, features),
            # en el print de les shapes teniem --> shape del dataloader:  torch.Size([32, 938, 128]) torch.Size([32])
        # Conv1d expects (batch, channels, time), so we need to permute
        # Permute x from [batch, time, features] to [batch, features, time]
        x = x.permute(0, 2, 1)
        
        # Process audio features
        x = F.relu(self.feature_extractor(x))

        # Permute back to [batch, time, hidden_dim] for GRU
        x = x.permute(0, 2, 1)
        
        # Encode the audio sequence
        features, features_hidden = self.encoder(x)#.relu()
        
        logits = self.classifier(features)
        
        return logits
