import wandb
import torch
import torch.nn 
import torchvision
import torchvision.transforms as transforms
from models.models import *
from models.baseline import *

import os
import pandas as pd
import numpy as np
from feature_extraction import load_audio, normalize_audio, trim_silence
import librosa
from pathlib import Path


def get_data(data_path="./fma_small", metadata_path="./fma_metadata", slice=1, train=True):
    """
        Load FMA_small dataset

        Args:
            data_path: Path to FMA_small audio files
            metadata_path: Path to FMA metadata
            slice: Subset every nth sample
            train: Whether to load training or test data

        Returns:
            A custom FMA dataset
        """

    #Lista de IDs realmente disponibles en fma_small
    mp3_paths = list(Path(data_path).rglob("*.mp3"))
    subset_ids = [int(p.stem) for p in mp3_paths]

    #Carga metadatos y reduce SOLO a subset_ids
    tracks = pd.read_csv(os.path.join(metadata_path, 'tracks.csv'),
                         index_col=0, header=[0, 1])
    genre_labels = tracks.loc[subset_ids, ('track','genre_top')]

    #Mapea géneros a índices, con orden estable
    top = sorted(genre_labels.unique())
    genre_dict = {g:i for i,g in enumerate(top)}

    #Usamos solo subset_ids para el split
    track_ids = list(genre_labels.index)
    np.random.seed(42)
    np.random.shuffle(track_ids)
    split_idx = int(0.8 * len(track_ids))
    selected_ids = (track_ids[:split_idx] if train else track_ids[split_idx:])
    selected_ids = selected_ids[::slice]

    print("Selected genre labels: ", genre_labels)
    print("Selected genre dict: ", genre_dict)
    print("Selected track length: ", len(selected_ids))
    print("Selected genre length: ", len(selected_ids))

    dataset = FMADataset(
        data_path=data_path,
        track_ids=selected_ids,
        genre_dict=genre_dict,
        genre_labels=genre_labels,
        transform=None
    )
    print("Example of features and label: ", dataset.__getitem__(0))

    return dataset


def make_loader(dataset, batch_size):

    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=2,
        collate_fn=collate_fn
    )
    return loader


def collate_fn(batch):
    """
    Custom collate function to handle variable-length 2D feature tensors.
    Each item in batch is (features, label) where
      features: Tensor[T, n_mels]
      label: int
    """
    # Separa características y etiquetas
    feats = [item[0] for item in batch]  # lista de Tensor[T_i, n_mels]
    labels = [item[1] for item in batch]

    # Encuentra la longitud máxima T_max
    T_max = max(f.shape[0] for f in feats)
    n_mels = feats[0].shape[1]

    # Prepara lista de tensores padded
    padded_feats = []
    for f in feats:
        T_i = f.shape[0]
        if T_i < T_max:
            # Crea un tensor de ceros (T_max, n_mels) y copia f en las primeras T_i filas
            pad = torch.zeros((T_max, n_mels), dtype=f.dtype)
            pad[:T_i, :] = f
            padded_feats.append(pad)
        else:
            padded_feats.append(f)

    # Apila en batch: (batch, T_max, n_mels)
    feat_tensor = torch.stack(padded_feats, dim=0)
    label_tensor = torch.tensor(labels, dtype=torch.long)

    return feat_tensor, label_tensor



class FMADataset(torch.utils.data.Dataset):

    def __init__(self, data_path, track_ids, genre_dict, genre_labels, transform=None):
        self.data_path = data_path
        self.track_ids = track_ids
        self.genre_dict = genre_dict
        self.genre_labels = genre_labels
        self.transform = transform

    def __len__(self):
        return len(self.track_ids)

    def __getitem__(self, idx):
        track_id = self.track_ids[idx]

        #construim el path sencer al track_id
        #aqui estem detctant el nom de la carpeta q son 3 numeros sempre (pq FMA utilita carpetes basades en el track_id)
        track_id_str = str(track_id).zfill(6)
        folder = track_id_str[:3]
        file_path = os.path.join(self.data_path, folder, track_id_str + '.mp3')

        try:
            #carreguem audio amb librosa
            audio, sr = librosa.load(file_path, sr=16000, mono=True)
            audio = torch.from_numpy(audio).float()

            audio = normalize_audio(audio)
            audio = trim_silence(audio, sr)

            #extract features (mel spectrogram)
            if len(audio) > 0:  # Ensure we have audio data
                features = librosa.feature.melspectrogram(
                    y=audio.numpy(),
                    sr=sr,
                    n_mels=128
                )
                features = librosa.power_to_db(features, ref=np.max)
                features = torch.from_numpy(features).float()

                #shape (time, features)
                features = features.transpose(0, 1)
            else:
                features = torch.zeros((128, 128))
                features = features.transpose(0, 1)

            genre = self.genre_labels[track_id]
            label = self.genre_dict[genre]

            return features, label

        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            features = torch.zeros((128, 128))
            features = features.transpose(0, 1)
            label = 0

            return features, label


def make(config, device="cpu"):
    """
    Create model, dataloaders, loss function, and optimizer
    """
    # Make the data
    train, test = get_data(
        data_path=config.data_path,
        metadata_path=config.metadata_path,
        train=True
    ), get_data(
        data_path=config.data_path,
        metadata_path=config.metadata_path,
        train=False
    )

    train_loader = make_loader(train, batch_size=config.batch_size)
    test_loader = make_loader(test, batch_size=config.batch_size)

    # Make the model
    model = Baseline(
        input_dim=128,  # mel spectrogram features
        hidden_dim=128,
        embedding_dim=32,
        num_classes=len(train.genre_dict),  # num of genres
        sampled_features_from_processor=128  # mel bins
    ).to(device)

    # Make the loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate)

    return model, train_loader, test_loader, criterion, optimizer