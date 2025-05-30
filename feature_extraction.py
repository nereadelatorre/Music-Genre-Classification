import torch
import os
from pathlib import Path
import h5py
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm


# Configuració per extracció de característiques
SR = 22050       # Tasa de mostreig
N_MELS = 128     # Numero de bandas mel
HOP_LENGTH = 512 # Hop length per STFT
N_FFT = 2048     # Longitud de la finestra FFT


def extract_and_save_features(audio_dir, metadata_csv, output_h5_path, duration=30.0):
    """
    Extreu les característiques mel de tots els fitxers d'àudio i les desa en un fitxer HDF5.

    Args:
        audio_dir (str): Directori amb els fitxers d'àudio (estructura FMA: subcarpetes basades en l'ID)
        metadata_csv (str): Ruta al CSV de metadades amb MultiIndex (tracks.csv)
        output_h5_path (str): Ruta on desar el fitxer HDF5
        duration (float): Durada en segons per retallar/omplir tots els àudios
    """
    metadata = pd.read_csv(
        metadata_csv,
        index_col=0,
        header=[0,1],
        low_memory=False
    )

    mp3_paths = list(Path(audio_dir).rglob("*.mp3"))
    available_ids = [int(p.stem) for p in mp3_paths]
    genre_series = metadata[('track','genre_top')].loc[available_ids].dropna()

    unique_genres = sorted(genre_series.unique())
    genre_dict = {g: i for i, g in enumerate(unique_genres)}

    os.makedirs(os.path.dirname(output_h5_path), exist_ok=True)

    with h5py.File(output_h5_path, 'w') as h5f:
        features_grp = h5f.create_group('features')
        labels_grp   = h5f.create_group('labels')

        for track_id, genre in tqdm(genre_series.items(), desc="Extrayendo features"):  # track_id es int
            tid_str = str(track_id).zfill(6)
            folder = tid_str[:3]
            audio_path = os.path.join(audio_dir, folder, f"{tid_str}.mp3")
            try:
                # Carregar audio i retallar/pad
                y, _ = librosa.load(audio_path, sr=SR, duration=duration)
                target_len = int(duration * SR)
                if len(y) < target_len:
                    y = np.pad(y, (0, target_len - len(y)), 'constant')
                else:
                    y = y[:target_len]

                # Mel spectrogram -> dB -> normalitzacio [0,1]
                mel = librosa.feature.melspectrogram(
                    y=y, sr=SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH
                )
                mel_db = librosa.power_to_db(mel, ref=np.max)
                eps = 1e-6
                mel_norm = normalize_audio(mel_db)

                features_grp.create_dataset(tid_str, data=mel_norm, compression='lzf')
                labels_grp.create_dataset(tid_str, data=genre_dict[genre])

            except Exception as e:
                print(f"Error processant {tid_str}: {e}")

    print(f"Features i labels guardades en {output_h5_path}")


def normalize_audio(mel_db):
    eps = 1e-6
    return (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + eps)

def trim_silence(waveform, sr):
    trimmed, _ = librosa.effects.trim(waveform.numpy(), top_db=20)
    return torch.tensor(trimmed)


def extract_and_save_mfcc_features(audio_dir, metadata_csv, output_h5_path, duration=30.0):
    """
    Extreu les característiques MFCC de tots els fitxers d'àudio i les desa en un fitxer HDF5.

    Args:
        audio_dir (str): Directori amb els fitxers d'àudio (estructura FMA: subcarpetes basades en l'ID)
        metadata_csv (str): Ruta al CSV de metadades amb MultiIndex (tracks.csv)
        output_h5_path (str): Ruta on desar el fitxer HDF5
        duration (float): Durada en segons per retallar/omplir tots els àudios
    """
    metadata = pd.read_csv(
        metadata_csv,
        index_col=0,
        header=[0,1],
        low_memory=False
    )

    mp3_paths = list(Path(audio_dir).rglob("*.mp3"))
    available_ids = [int(p.stem) for p in mp3_paths]
    genre_series = metadata[('track','genre_top')].loc[available_ids].dropna()

    unique_genres = sorted(genre_series.unique())
    genre_dict = {g: i for i, g in enumerate(unique_genres)}

    os.makedirs(os.path.dirname(output_h5_path), exist_ok=True)

    with h5py.File(output_h5_path, 'w') as h5f:
        features_grp = h5f.create_group('features')
        labels_grp   = h5f.create_group('labels')

        for track_id, genre in tqdm(genre_series.items(), desc="Extrayendo MFCCs"):
            tid_str = str(track_id).zfill(6)
            folder = tid_str[:3]
            audio_path = os.path.join(audio_dir, folder, f"{tid_str}.mp3")
            try:
                y, _ = librosa.load(audio_path, sr=SR)

                target_len = int(duration * SR)
                if len(y) < target_len:
                    y = np.pad(y, (0, target_len - len(y)), 'constant')
                else:
                    y = y[:target_len]


                mfcc = librosa.feature.mfcc(
                    y=y,
                    sr=SR,
                    n_mfcc=20,
                    n_fft=N_FFT,
                    hop_length=HOP_LENGTH
                )
                mfcc_norm = normalize_audio(mfcc)

                features_grp.create_dataset(tid_str, data=mfcc_norm, compression='lzf')
                labels_grp.create_dataset(tid_str, data=genre_dict[genre])

            except Exception as e:
                print(f"Error processant {tid_str}: {e}")

    print(f"MFCCs i labels guardades en {output_h5_path}")
