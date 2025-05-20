import torch
import os
from pathlib import Path
import h5py
import librosa
import numpy as np
import pandas as pd
import torchaudio
from tqdm import tqdm
from transformers import Speech2TextProcessor

# Configuración para extracción de características
SR = 22050       # Tasa de muestreo
N_MELS = 128     # Número de bandas mel
HOP_LENGTH = 512 # Hop length para STFT
N_FFT = 2048     # Longitud de la ventana FFT


def extract_and_save_features(audio_dir, metadata_csv, output_h5_path, duration=30.0):
    """
    Extrae las características mel de todos los archivos de audio y las guarda en un archivo HDF5.

    Args:
        audio_dir (str): Directorio con los archivos de audio (estructura FMA: subcarpetas en base ID)
        metadata_csv (str): Ruta al CSV de metadatos con MultiIndex (tracks.csv)
        output_h5_path (str): Ruta donde guardar el archivo HDF5
        duration (float): Duración en segundos para recortar/rellenar todos los audios
    """
    # 1) Cargar metadatos con MultiIndex y configurar índice
    metadata = pd.read_csv(
        metadata_csv,
        index_col=0,
        header=[0,1],
        low_memory=False
    )
    # Extraer la serie de géneros y filtrar las pistas disponibles en audio_dir
    # Asumimos ficheros .mp3 con nombres zfilled a 6 dígitos y subcarpetas por los primeros 3 dígitos
    mp3_paths = list(Path(audio_dir).rglob("*.mp3"))
    available_ids = [int(p.stem) for p in mp3_paths]
    genre_series = metadata[('track','genre_top')].loc[available_ids].dropna()

    # Mapea géneros a índices, con orden estable
    unique_genres = sorted(genre_series.unique())
    genre_dict = {g: i for i, g in enumerate(unique_genres)}

    # 2) Crear carpeta destino si no existe
    os.makedirs(os.path.dirname(output_h5_path), exist_ok=True)

    # 3) Crear archivo HDF5 y grupos
    with h5py.File(output_h5_path, 'w') as h5f:
        features_grp = h5f.create_group('features')
        labels_grp   = h5f.create_group('labels')

        # Iterar solo sobre las pistas con género válido
        for track_id, genre in tqdm(genre_series.items(), desc="Extrayendo features"):  # track_id es int
            tid_str = str(track_id).zfill(6)
            # Construir ruta al mp3 en FMA_SMALL
            folder = tid_str[:3]
            audio_path = os.path.join(audio_dir, folder, f"{tid_str}.mp3")
            try:
                # Cargar audio y recortar/pad
                y, _ = librosa.load(audio_path, sr=SR, duration=duration)
                target_len = int(duration * SR)
                if len(y) < target_len:
                    y = np.pad(y, (0, target_len - len(y)), 'constant')
                else:
                    y = y[:target_len]

                # Mel spectrogram -> dB -> normalización [0,1]
                mel = librosa.feature.melspectrogram(
                    y=y, sr=SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH
                )
                mel_db = librosa.power_to_db(mel, ref=np.max)
                eps = 1e-6
                mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + eps)

                # Guardar features y label (int)
                features_grp.create_dataset(tid_str, data=mel_norm, compression='lzf')
                labels_grp.create_dataset(tid_str, data=genre_dict[genre])

            except Exception as e:
                print(f"Error procesando {tid_str}: {e}")

    print(f"> Features y labels guardadas en {output_h5_path}")

#  Normalize waveform
#  Normaliza el volumen del audio
def normalize_audio(waveform):
    return waveform / waveform.abs().max()

#  Trim silence using librosa
#  Elimina los silencios al principio/final
def trim_silence(waveform, sr):
    trimmed, _ = librosa.effects.trim(waveform.numpy(), top_db=20)
    return torch.tensor(trimmed)



