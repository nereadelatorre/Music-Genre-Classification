import os
import random
import wandb
import numpy as np
import torch
import pandas as pd
from pathlib import Path

from train import *
from test import *
from utils.utils import *
from feature_extraction import extract_and_save_features, extract_and_save_mfcc_features
from models import *

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Reproducibilidad
torch.backends.cudnn.deterministic = True
random.seed(hash("setting random seeds") % 2**32 - 1)
np.random.seed(hash("improves reproducibility") % 2**32 - 1)
torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)


def model_pipeline(cfg: dict, dicc_genre_to_name):
    with wandb.init(project="pytorch-demo", config=cfg):
        config = wandb.config

        excluded_genres = config.get("excluded_genres", [])
        model, train_loader, val_loader, test_loader, criterion, optimizer = make(
            config,
            dicc_genre_to_name=dicc_genre_to_name,
            excluded_genres=excluded_genres,
            device=device
        )

        model = model.to(device)

        train(model, train_loader, val_loader, criterion, optimizer, config, device=device)
        print("Train done\n")

        test(model, test_loader, dicc_genre_to_name, excluded_genres, device=device)
        print("Test done\n")

        return model


if __name__ == "__main__":
    wandb.login()

    print("GPU disponible:", torch.cuda.is_available())
    print("Nombre GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No hay GPU")
    torch.cuda.empty_cache()

    audio_dir = "/home/datasets/FreeMusicArchive/fma_large"
    metadata_csv = "/home/datasets/FreeMusicArchive/fma_metadata/tracks.csv"

    feature_type = "mfcc"  # Canvia a "mel" per mel spectrograms, i a "mfcc" per MFCC

    if feature_type == "mfcc":
        H5_PATH = "./fma_mfcc_features.h5"
        if not os.path.exists(H5_PATH):
            extract_and_save_mfcc_features(audio_dir, metadata_csv, H5_PATH)
    else:
        H5_PATH = "./fma_mel_features.h5"
        if not os.path.exists(H5_PATH):
            extract_and_save_features(audio_dir, metadata_csv, H5_PATH)

    # Carreguem metadata i diccionari dels generes:
    metadata = pd.read_csv(metadata_csv, index_col=0, header=[0, 1], low_memory=False)
    mp3_paths = list(Path(audio_dir).rglob("*.mp3"))
    available_ids = [int(p.stem) for p in mp3_paths]
    genre_series = metadata[('track', 'genre_top')].loc[available_ids].dropna()
    genres = sorted(genre_series.unique())
    DICC_GENRE_TO_NAME = {i: name for i, name in enumerate(genres)}

    # Configuració
    default_cfg = dict(
        epochs=40,
        batch_size=16,
        learning_rate=0.001,
        hidden_dim=64,
        num_layers=2,
        dropout=0.3,
        step_lr_step_size=10,
        step_lr_gamma=0.7,
        reduce_lr_factor=0.7,
        dataset="CNN(2D)-LSTM",
        feature_type=feature_type,
        h5_path=H5_PATH,
        excluded_genres=["Easy Listening", "Blues", "Soul-RnB", "Country"]
    )

    model = model_pipeline(default_cfg, DICC_GENRE_TO_NAME)
