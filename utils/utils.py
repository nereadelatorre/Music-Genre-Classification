import wandb
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from models.CNNGRU import *
from models.CNNLSTM import *
from models.CNNLSTM2d import *
from dataAugmentation import *
import h5py
import random

import os
import pandas as pd
import numpy as np
from feature_extraction import load_audio, normalize_audio, trim_silence
import librosa
from pathlib import Path


def split_filtered_dataset(dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    Divide the dataset into train, validation and test sets while maintaining class distribution.

    Args:
        dataset: Original dataset
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for testing
        seed: Seed for reproducibility

    Returns:
        train_dataset, val_dataset, test_dataset
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-10, "Proportions must sum to 1"

    np.random.seed(seed)

    # Get IDs and labels
    track_ids = dataset.track_ids
    labels = [dataset.labels[tid] for tid in track_ids]

    # IDs for each set by genre
    train_ids = []
    val_ids = []
    test_ids = []

    # Get unique labels after filtering
    unique_labels = set(labels)

    for label in unique_labels:
        # Get IDs for this genre
        genre_ids = [id for i, id in enumerate(track_ids) if labels[i] == label]
        np.random.shuffle(genre_ids)

        # Calculate split indices
        train_end = int(len(genre_ids) * train_ratio)
        val_end = train_end + int(len(genre_ids) * val_ratio)

        # Split by genre
        train_ids.extend(genre_ids[:train_end])
        val_ids.extend(genre_ids[train_end:val_end])
        test_ids.extend(genre_ids[val_end:])

    # Verify no overlap
    assert len(set(train_ids).intersection(set(val_ids))) == 0
    assert len(set(train_ids).intersection(set(test_ids))) == 0
    assert len(set(val_ids).intersection(set(test_ids))) == 0

    # Create subdatasets
    train_dataset = FilteredMelSpectrogramSubset(dataset, train_ids)
    val_dataset = FilteredMelSpectrogramSubset(dataset, val_ids)
    test_dataset = FilteredMelSpectrogramSubset(dataset, test_ids)

    print(f"Dataset split completed:")
    print(f"  - Train: {len(train_dataset)} samples ({len(train_dataset) / len(dataset):.1%})")
    print(f"  - Validation: {len(val_dataset)} samples ({len(val_dataset) / len(dataset):.1%})")
    print(f"  - Test: {len(test_dataset)} samples ({len(test_dataset) / len(dataset):.1%})")

    print("GENRES: ", dataset.genres)

    return train_dataset, val_dataset, test_dataset


def make_loader(dataset, batch_size):
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
        collate_fn=pad_collate  # Custom collate function to handle variable-length audio
    )
    return loader


class FilteredMelSpectrogramDataset(torch.utils.data.Dataset):
    def __init__(self, h5_path, DICC_GENRE_TO_NAME, excluded_genres=None, transform=None):
        """
        Dataset for loading mel spectrograms from HDF5 file with genre filtering.

        Args:
            h5_path (str): Path to HDF5 file
            excluded_genres (list): List of genre names to exclude
            transform (callable, optional): Optional transform
        """
        self.h5_path = h5_path
        self.transform = transform
        self.excluded_genres = ["Easy Listening", "Blues", "Soul-RnB", "Country"]  # excluded_genres or []

        # Load IDs and labels (metadata only)
        with h5py.File(h5_path, 'r') as f:
            # First, get all track IDs and their genre labels
            all_track_ids = list(f['features'].keys())
            all_labels = {tid: f['labels'][tid][()] for tid in all_track_ids}
            self.genre_to_name = DICC_GENRE_TO_NAME

            # Create a set of indices for excluded genres
            excluded_indices = set()
            for genre_idx, genre_name in self.genre_to_name.items():
                if genre_name in self.excluded_genres:
                    excluded_indices.add(genre_idx)

            # Filter out tracks with excluded genres
            self.track_ids = [tid for tid in all_track_ids
                              if all_labels[tid] not in excluded_indices]
            self.labels = {tid: all_labels[tid] for tid in self.track_ids}

            # Create a new mapping for the filtered genres
            self.used_genre_indices = sorted(set(self.labels.values()))
            self.index_mapping = {old_idx: new_idx for new_idx, old_idx
                                  in enumerate(self.used_genre_indices)}

            # Get the filtered genre names
            self.genres = [self.genre_to_name[idx] for idx in self.used_genre_indices]

    def __len__(self):
        return len(self.track_ids)

    def __getitem__(self, idx):
        track_id = self.track_ids[idx]
        with h5py.File(self.h5_path, 'r') as f:
            mel = f['features'][track_id][()]
            original_label = f['labels'][track_id][()]

        # Remap the label to the new contiguous indices
        remapped_label = self.index_mapping[original_label]

        mel_tensor = torch.from_numpy(mel).T.float()  # [time, 128]
        if self.transform:
            mel_tensor = self.transform(mel_tensor)

        label_tensor = torch.tensor(remapped_label, dtype=torch.long)
        return mel_tensor, label_tensor

    def get_genres(self):
        return self.genres

    def get_label_distribution(self):
        """Get the class distribution in this dataset"""
        counts = {}
        for genre_label in self.labels.values():
            # Map to the new index
            new_index = self.index_mapping[genre_label]
            genre = self.genres[new_index]
            counts[genre] = counts.get(genre, 0) + 1
        return counts


class FilteredMelSpectrogramSubset(torch.utils.data.Dataset):
    """Dataset that uses a subset of IDs from a FilteredMelSpectrogramDataset"""

    def __init__(self, parent_dataset, subset_ids):
        self.h5_path = parent_dataset.h5_path
        self.track_ids = subset_ids
        self.genres = parent_dataset.genres
        self.transform = parent_dataset.transform
        self.index_mapping = parent_dataset.index_mapping
        self.genre_to_name = parent_dataset.genre_to_name

        self.labels = {
            tid: parent_dataset.labels[tid]
            for tid in subset_ids
        }

    def __len__(self):
        return len(self.track_ids)

    def __getitem__(self, idx):
        tid = self.track_ids[idx]
        with h5py.File(self.h5_path, 'r') as f:
            mel = f['features'][tid][()]
            original_label = f['labels'][tid][()]

        # Remap the label to the new contiguous indices
        remapped_label = self.index_mapping[original_label]

        mel_spec_tensor = torch.from_numpy(mel).T.float()  # [time, 128]
        if self.transform:
            mel_spec_tensor = self.transform(mel_spec_tensor)

        label_tensor = torch.tensor(remapped_label, dtype=torch.long)
        return mel_spec_tensor, label_tensor

    def get_label_distribution(self):
        """Get the class distribution in this subset"""
        counts = {}
        for genre_label in self.labels.values():
            # Map to the new index
            new_index = self.index_mapping[genre_label]
            genre = self.genres[new_index]
            counts[genre] = counts.get(genre, 0) + 1
        return counts


def pad_collate(batch):
    specs, labels = zip(*batch)
    # calcular max_len
    max_len = max(s.size(-1) for s in specs)
    padded = []
    for s in specs:
        pad = max_len - s.size(-1)
        if pad > 0:
            s = torch.nn.functional.pad(s, (0, pad))
        padded.append(s)
    return torch.stack(padded), torch.tensor(labels)


def make(config, excluded_genres=None, device="cuda:0"):
    """
    Create model, dataloaders, loss function, and optimizer with genre filtering

    Args:
        config: Configuration object
        excluded_genres: List of genre names to exclude (e.g., ["Blues", "Country"])
        device: Device to use for training
    """
    # Create transformations for augmentation
    transforms = get_audio_transforms(augment=True)

    # Create the full dataset without augmentation but with filter of genres
    excluded_genres = ["Easy Listening", "Blues", "Soul-RnB", "Country"]  # excluded_genres or []

    mp3_paths = list(Path("/home/datasets/FreeMusicArchive/fma_large").rglob("*.mp3"))
    available_ids = [int(p.stem) for p in mp3_paths]

    metadata = pd.read_csv(
        "/home/datasets/FreeMusicArchive/fma_metadata/tracks.csv",
        index_col=0,
        header=[0, 1],
        low_memory=False
    )
    genre_series = metadata[('track', 'genre_top')].loc[available_ids].dropna()
    genres = sorted(genre_series.unique())
    dicc = {i: name for i, name in enumerate(genres)}
    full_dataset = FilteredMelSpectrogramDataset(config.h5_path, DICC_GENRE_TO_NAME=dicc,
                                                 excluded_genres=excluded_genres, transform=None)

    # Split into train, val and test
    train, val, test = split_filtered_dataset(full_dataset)

    # Add transformations only to the training set
    train.transform = transforms

    train_loader = make_loader(train, batch_size=config.batch_size)
    val_loader = make_loader(val, batch_size=config.batch_size)
    test_loader = make_loader(test, batch_size=config.batch_size)

    print("DataLoaders DONE\n")

    print(f"Dataset loaded with {len(full_dataset)} samples after filter of genres")
    print(f"Remaining genres: {full_dataset.genres}")

    # Check genre distribution in each set
    print("\nGenre distribution in train:")
    for genre, count in train_loader.dataset.get_label_distribution().items():
        print(f"  - {genre}: {count} samples")

    print("\nGenre distribution in validation:")
    for genre, count in val_loader.dataset.get_label_distribution().items():
        print(f"  - {genre}: {count} samples")

    print("\nGenre distribution in test:")
    for genre, count in test_loader.dataset.get_label_distribution().items():
        print(f"  - {genre}: {count} samples")

    # Make the model
    """model = Baseline(
        input_dim=128,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
        num_classes=len(full_dataset.genres)  # Updated number after filtering
    ).to(device)
    model = LSTM(
        input_dim=128,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
        num_classes=len(full_dataset.genres)  # Updated number after filtering
    ).to(device)

    model = LSTM2D(
        input_dim=128,  
        time_dim=1292,     # Dimensión temporal del espectrograma
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
        num_classes=len(full_dataset.genres)  # Updated number after filtering).to(device)
    ).to(device)
    """
    model = CNN2D_LSTM(in_channels=1,
                       base_channels=64,
                       lstm_hidden=128,
                       num_layers=config.num_layers,
                       num_classes=len(full_dataset.genres),
                       n_mels=128).to(device)

    # Make the loss and optimizer
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=1e-4
    )

    return model, train_loader, val_loader, test_loader, criterion, optimizer

