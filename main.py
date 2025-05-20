import os
import random
import wandb

import numpy as np
import torch
import torchaudio
from train import *
from test import *
from utils.utils import *
from feature_extraction import *
from tqdm.auto import tqdm
from models import *

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Reproducibilitat
torch.backends.cudnn.deterministic = True
random.seed(hash("setting random seeds") % 2 ** 32 - 1)
np.random.seed(hash("improves reproducibility") % 2 ** 32 - 1)
torch.manual_seed(hash("by removing stochasticity") % 2 ** 32 - 1)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2 ** 32 - 1)


def model_pipeline(cfg: dict):
    with wandb.init(project="pytorch-demo", config=cfg):
        config = wandb.config

        # Crear model i dataloaders
        excluded_genres = config.get("excluded_genres", [])
        model, train_loader, val_loader, test_loader, criterion, optimizer = make(config,
                                                                                  excluded_genres=excluded_genres,
                                                                                  device=device)

        model = model.to(device)

        # exemple del dataloader:
        for x, y in train_loader:
            print("shape del dataloader:", x.shape, y.shape)
            break

        # Train
        train(model, train_loader, val_loader, criterion, optimizer, config, device=device)
        print("TRAIN DONE\n")

        # Test
        test(model, test_loader, config, device=device)
        print("TEST DONE\n")

        return model


if __name__ == "__main__":
    wandb.login()

    print("GPU disponible:", torch.cuda.is_available())
    print("Nombre GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No hay GPU")

    H5_PATH = "./fma_mel_features.h5"

    # Extreure i guardar features (noms un cop)
    if not os.path.exists(H5_PATH):
        extract_and_save_features("/home/datasets/FreeMusicArchive/fma_large",
                                  "/home/datasets/FreeMusicArchive/fma_metadata/tracks.csv", H5_PATH)

    metadata = pd.read_csv(
        "/home/datasets/FreeMusicArchive/fma_metadata/tracks.csv",
        index_col=0,
        header=[0, 1],
        low_memory=False
    )

    mp3_paths = list(Path("/home/datasets/FreeMusicArchive/fma_large").rglob("*.mp3"))
    available_ids = [int(p.stem) for p in mp3_paths]
    genre_series = metadata[('track', 'genre_top')].loc[available_ids].dropna()
    genres = sorted(genre_series.unique())
    DICC_GENRE_TO_NAME = {i: name for i, name in enumerate(genres)}

    # Hiperparametres i rutes:

    default_cfg = dict(
        epochs=30,
        batch_size=16,
        learning_rate=1e-3,
        hidden_dim=128,
        num_layers=2,
        dropout=0.4,
        step_lr_step_size=5,
        step_lr_gamma=0.7,
        reduce_lr_factor=0.7,
        dataset="FMA_large",
        architecture="CNN-LSTM",
        h5_path=H5_PATH,
        dicc=DICC_GENRE_TO_NAME,
        excluded_genres=["Easy Listening", "Blues", "Soul-RnB", "Country"]
        # ["Genre_0", "Genre_3", "Genre_4", "Genre_14"]
    )

    model = model_pipeline(default_cfg)
