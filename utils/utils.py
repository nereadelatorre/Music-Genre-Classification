import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from models.CNNGRU1d import *
from models.CNNLSTM1d import *
from models.CNNLSTM2d import *
from models.CNNGRU2d import *
from models.CNN import *
from dataAugmentation import *
import h5py
import numpy as np


def split_filtered_dataset(dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    Divideix el conjunt de dades en conjunts d'entrenament, validació i prova mantenint la distribució de les classes.

    Args:
        dataset: Conjunt de dades original
        train_ratio: Proporció per a l'entrenament
        val_ratio: Proporció per a la validació
        test_ratio: Proporció per a la prova
        seed: Llavor per a la reproductibilitat

    Returns:
        train_dataset, val_dataset, test_dataset
    """

    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-10, "Les proporcions han de sumar 1"

    np.random.seed(seed)

    # IDs i labels
    track_ids = dataset.track_ids
    labels = [dataset.labels[tid] for tid in track_ids]

    # IDs per cada genre
    train_ids = []
    val_ids = []
    test_ids = []

    # Labels unics despres de filtrar
    unique_labels = set(labels)

    for label in unique_labels:
        genre_ids = [id for i, id in enumerate(track_ids) if labels[i] == label]
        np.random.shuffle(genre_ids)

        # Split indices
        train_end = int(len(genre_ids) * train_ratio)
        val_end = train_end + int(len(genre_ids) * val_ratio)

        # Split per genre
        train_ids.extend(genre_ids[:train_end])
        val_ids.extend(genre_ids[train_end:val_end])
        test_ids.extend(genre_ids[val_end:])

    # Verifiquem no overlap
    assert len(set(train_ids).intersection(set(val_ids))) == 0
    assert len(set(train_ids).intersection(set(test_ids))) == 0
    assert len(set(val_ids).intersection(set(test_ids))) == 0

    # Subdatasets
    train_dataset = FilteredMelSpectrogramSubset(dataset, train_ids)
    val_dataset = FilteredMelSpectrogramSubset(dataset, val_ids)
    test_dataset = FilteredMelSpectrogramSubset(dataset, test_ids)

    print(f"Dataset split completat:")
    print(f"  - Train: {len(train_dataset)} samples ({len(train_dataset) / len(dataset):.1%})")
    print(f"  - Validation: {len(val_dataset)} samples ({len(val_dataset) / len(dataset):.1%})")
    print(f"  - Test: {len(test_dataset)} samples ({len(test_dataset) / len(dataset):.1%})")

    return train_dataset, val_dataset, test_dataset


def make_loader(dataset, batch_size):
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
        collate_fn=pad_collate
    )
    return loader


class FilteredMelSpectrogramDataset(torch.utils.data.Dataset):
    def __init__(self, h5_path, DICC_GENRE_TO_NAME, excluded_genres=None, transform=None):
        """
        Conjunt de dades per carregar espectrogrames de mel des d'un fitxer HDF5 amb filtratge per gènere.

        Args:
            h5_path (str): Ruta al fitxer HDF5
            DICC_GENRE_TO_NAME: diccionari de noms dels generes
            excluded_genres (list): Llista de noms de gèneres a excloure
            transform (callable, optional): Transformació opcional
        """

        self.h5_path = h5_path
        self.transform = transform
        self.excluded_genres = ["Easy Listening", "Blues", "Soul-RnB", "Country"] or []

        # IDs i labels (metadata nomes)
        with h5py.File(h5_path, 'r') as f:
            # Obtenim tots els track IDs i el seus genre labels
            all_track_ids = list(f['features'].keys())
            all_labels = {tid: f['labels'][tid][()] for tid in all_track_ids}
            self.genre_to_name = DICC_GENRE_TO_NAME

            # Set d'indices per excluded genres
            excluded_indices = set()
            for genre_idx, genre_name in self.genre_to_name.items():
                if genre_name in self.excluded_genres:
                    excluded_indices.add(genre_idx)

            # Filtrar tracks amb excluded genres
            self.track_ids = [tid for tid in all_track_ids
                              if all_labels[tid] not in excluded_indices]
            self.labels = {tid: all_labels[tid] for tid in self.track_ids}

            self.used_genre_indices = sorted(set(self.labels.values()))
            self.index_mapping = {old_idx: new_idx for new_idx, old_idx
                                  in enumerate(self.used_genre_indices)}

            self.genres = [self.genre_to_name[idx] for idx in self.used_genre_indices]

    def __len__(self):
        return len(self.track_ids)

    def __getitem__(self, idx):
        track_id = self.track_ids[idx]
        with h5py.File(self.h5_path, 'r') as f:
            mel = f['features'][track_id][()]
            original_label = f['labels'][track_id][()]

        remapped_label = self.index_mapping[original_label]

        mel_tensor = torch.from_numpy(mel).T.float()  # [time, 128]
        if self.transform:
            mel_tensor = self.transform(mel_tensor)

        label_tensor = torch.tensor(remapped_label, dtype=torch.long)
        return mel_tensor, label_tensor

    def get_genres(self):
        return self.genres

    def get_label_distribution(self):
        counts = {}
        for genre_label in self.labels.values():
            new_index = self.index_mapping[genre_label]
            genre = self.genres[new_index]
            counts[genre] = counts.get(genre, 0) + 1
        return counts


class FilteredMFCCDataset(torch.utils.data.Dataset):
    def __init__(self, h5_path, DICC_GENRE_TO_NAME, excluded_genres=None, transform=None):
        self.h5_path = h5_path
        self.transform = transform
        self.excluded_genres = excluded_genres or []

        with h5py.File(h5_path, 'r') as f:
            all_track_ids = list(f['features'].keys())
            all_labels = {tid: f['labels'][tid][()] for tid in all_track_ids}
            self.genre_to_name = DICC_GENRE_TO_NAME

            excluded_indices = set()
            for genre_idx, genre_name in self.genre_to_name.items():
                if genre_name in self.excluded_genres:
                    excluded_indices.add(genre_idx)

            self.track_ids = [tid for tid in all_track_ids if all_labels[tid] not in excluded_indices]
            self.labels = {tid: all_labels[tid] for tid in self.track_ids}

            self.used_genre_indices = sorted(set(self.labels.values()))
            self.index_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(self.used_genre_indices)}
            self.genres = [self.genre_to_name[idx] for idx in self.used_genre_indices]

    def __len__(self):
        return len(self.track_ids)

    def __getitem__(self, idx):
        tid = self.track_ids[idx]
        with h5py.File(self.h5_path, 'r') as f:
            mfcc = f['features'][tid][()]  # Assumim que les features són MFCCs
            label = f['labels'][tid][()]

        remapped_label = self.index_mapping[label]

        mfcc_tensor = torch.from_numpy(mfcc).T.float()
        if self.transform:
            mfcc_tensor = self.transform(mfcc_tensor)

        return mfcc_tensor, torch.tensor(remapped_label, dtype=torch.long)

    def get_label_distribution(self):
        counts = {}
        for genre_label in self.labels.values():
            new_index = self.index_mapping[genre_label]
            genre = self.genres[new_index]
            counts[genre] = counts.get(genre, 0) + 1
        return counts


class FilteredMelSpectrogramSubset(torch.utils.data.Dataset):
    """Dataset que utilitza un subset de IDs del FilteredMelSpectrogramDataset"""

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

        remapped_label = self.index_mapping[original_label]

        mel_spec_tensor = torch.from_numpy(mel).T.float()  # [time, 128]
        if self.transform:
            mel_spec_tensor = self.transform(mel_spec_tensor)

        label_tensor = torch.tensor(remapped_label, dtype=torch.long)
        return mel_spec_tensor, label_tensor

    def get_label_distribution(self):
        counts = {}
        for genre_label in self.labels.values():
            new_index = self.index_mapping[genre_label]
            genre = self.genres[new_index]
            counts[genre] = counts.get(genre, 0) + 1
        return counts


def pad_collate(batch):
    """Funcio per ajustar longitud variable dels audios"""
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


def make(config, dicc_genre_to_name, excluded_genres=None, device="cuda:0"):
    """
    Crea el model, els carregadors de dades, la funció de pèrdua i l'optimitzador amb filtratge per gènere.

    Args:
        config: Objecte de configuració
        dicc_genre_to_name: diccionari amb els noms dels generes
        excluded_genres: Llista de noms de gèneres a excloure (p. ex., ["Blues", "Country"])
        device: Dispositiu a utilitzar per a l'entrenament
    """

    # Crear les transformations per augmentation
    transforms = get_audio_transforms(augment=True)

    if config.get("feature_type", "mel") == "mfcc":
        full_dataset = FilteredMFCCDataset(config.h5_path, DICC_GENRE_TO_NAME=dicc_genre_to_name,
                                           excluded_genres=excluded_genres, transform=None)
        input_dim = 20
    else:
        full_dataset = FilteredMelSpectrogramDataset(config.h5_path, DICC_GENRE_TO_NAME=dicc_genre_to_name,
                                                     excluded_genres=excluded_genres, transform=None)
        input_dim = 128

    # Split en train, val i test
    train, val, test = split_filtered_dataset(full_dataset)

    # Afegim les transformations NOMES al training set
    train.transform = transforms

    train_loader = make_loader(train, batch_size=config.batch_size)
    val_loader = make_loader(val, batch_size=config.batch_size)
    test_loader = make_loader(test, batch_size=config.batch_size)

    print("DataLoaders done\n")

    print(f"Dataset carregat amb {len(full_dataset)} samples despres de filtrar els generes")

    # Check genre distribution in each set
    print("\nGenre distribution en train:")
    for genre, count in train_loader.dataset.get_label_distribution().items():
        print(f"  - {genre}: {count} samples")

    print("\nGenre distribution en validation:")
    for genre, count in val_loader.dataset.get_label_distribution().items():
        print(f"  - {genre}: {count} samples")

    print("\nGenre distribution en test:")
    for genre, count in test_loader.dataset.get_label_distribution().items():
        print(f"  - {genre}: {count} samples")

    # Crear el model (descomentar el model a utilitzar)

    """model = Baseline(
        input_dim=input_dim, 
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
        num_classes=len(full_dataset.genres)
    ).to(device)"""

    """
    model = LSTM(
        input_dim=input_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
        num_classes=len(full_dataset.genres)
    ).to(device)
    """
    """
    model = LSTM2D(
        input_dim=input_dim,
        time_dim=1292,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
        num_classes=len(full_dataset.genres)
    ).to(device)
    """
    """
    model = CNN2D_LSTM(in_channels=1, 
                       base_channels=64, 
                       lstm_hidden=128, # 20
                       num_layers=config.num_layers, 
                       num_classes=len(full_dataset.genres),
                       n_mels=128).to(device)
    """

    model = Conv2D_GRU(
        input_dim=input_dim,
        time_dim=1292,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
        num_classes=len(full_dataset.genres))

    """
    model = CNN_2d(
                input_channels=1, 
                num_classes=len(full_dataset.genres)
            ).to(device)
    """

    # Loss i optimizer
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=1e-4
    )

    return model, train_loader, val_loader, test_loader, criterion, optimizer

