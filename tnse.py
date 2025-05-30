import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import seaborn as sns
import colorcet as cc
from collections import Counter
from utils.utils import FilteredMelSpectrogramDataset
from models.CNNGRU1d import *

# Configuració
plt.style.use('seaborn')
sns.set(style="whitegrid", font_scale=1.2)
H5_PATH = "./fma_mel_features.h5"

GENRE_MAPPING = {
    0: "Blues", 1: "Classical", 2: "Country", 3: "Easy Listening", 4: "Electronic", 5: "Experimental", 6: "Folk",
    7: "Hip-Hop", 8: "Instrumental", 9: "International", 10: "Jazz",
    11: "Old-Time / Historic", 12: "Pop", 13: "Rock", 14: "Soul-RnB", 15: "Spoken",
}

EXCLUDED_GENRES = ["Easy Listening", "Blues", "Soul-RnB", "Country"]


def setup_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilitzem dispositivo: {device}")
    return device


def load_model(device, input_dim=128, hidden_dim=64, num_layers=2, dropout=0.4, num_classes=12):
    model = Baseline(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        num_classes=num_classes
    )
    model.load_state_dict(torch.load("best_model.pt", map_location=device))
    model.to(device)
    model.eval()
    return model


def debug_labels_and_dataset(dataset):
    """Funció per debuggear les etiquetes i verificar la distribució"""
    print("=== DEBUG D'ETIQUETES ===")

    print(f"Gèneres en el dataset: {dataset.genres}")
    print(f"Número de gèneres: {len(dataset.genres)}")

    all_labels = []
    for i in range(len(dataset)):
        _, label = dataset[i]
        all_labels.append(label.item() if torch.is_tensor(label) else label)

    label_counts = Counter(all_labels)
    print(f"\nDistribució d'etiquetas (indexos): {dict(sorted(label_counts.items()))}")

    # Mapear a nombres de géneros
    genre_counts = {}
    for label_idx, count in label_counts.items():
        if label_idx < len(dataset.genres):
            genre_name = dataset.genres[label_idx]
            genre_counts[genre_name] = count
        else:
            print(f"Índex {label_idx} fora de rang per gèneres")

    print(f"\nDistribució por gèneres:")
    for genre, count in sorted(genre_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(all_labels)) * 100
        print(f"  {genre}: {count} mostres ({percentage:.1f}%)")

    return all_labels, genre_counts


def extract_embeddings_with_debug(model, dataloader, device):
    """Extrau embeddings i manté tracking de les etiquetes"""
    embeddings_list = []
    labels_list = []
    batch_labels_debug = []

    with torch.no_grad():
        for i, (mel_specs, labels) in enumerate(dataloader):
            print(f"Processant batch {i + 1}/{len(dataloader)}", end='\r')

            batch_labels_debug.extend(labels.cpu().numpy().tolist())

            mel_specs = mel_specs.to(device)
            x = mel_specs.permute(0, 2, 1)
            x = model.feature_extractor(x)
            x = x.permute(0, 2, 1)
            gru_out, _ = model.encoder(x)
            embedding = torch.mean(gru_out, dim=1)

            embeddings_list.append(embedding.cpu())
            labels_list.append(labels)

    print(f"\nDebug: Etiquetes extretes en els primers 10 exemples: {batch_labels_debug[:10]}")
    return torch.cat(embeddings_list, dim=0).numpy(), torch.cat(labels_list, dim=0).numpy()


def create_correct_label_mapping(dataset_genres):
    """Mapping entre indexos del dataset i noms"""
    return {i: genre for i, genre in enumerate(dataset_genres)}


def plot_tsne_with_stats(X_2d, label_indices, genre_names, title="t-SNE de embeddings musicals"):
    """Plot t-SNE amb estadístiques de distribució"""

    labels = np.array([genre_names[idx] for idx in label_indices])
    unique_genres = sorted(set(labels))

    total_samples = len(labels)
    genre_stats = {}
    for genre in unique_genres:
        count = np.sum(labels == genre)
        percentage = (count / total_samples) * 100
        genre_stats[genre] = {'count': count, 'percentage': percentage}

    sorted_genres = sorted(genre_stats.items(), key=lambda x: x[1]['percentage'], reverse=True)

    for genre, stats in sorted_genres:
        print(f"{genre}: {stats['count']} mostres ({stats['percentage']:.1f}%)")

    # Identifiquem classes minoritàries: (<5%)
    minority_classes = [genre for genre, stats in genre_stats.items() if stats['percentage'] < 5]
    if minority_classes:
        print(f"\nClasses minoritaries (<5%): {minority_classes}")
        print("Si apareixen molt separades, confirma el problema d'etiquetes.")

    plt.figure(figsize=(16, 12))

    palette = {genre: cc.glasbey[i] for i, genre in enumerate(unique_genres)}

    scatter = sns.scatterplot(
        x=X_2d[:, 0],
        y=X_2d[:, 1],
        hue=labels,
        palette=palette,
        alpha=0.85,
        edgecolor='black',
        linewidth=0.3,
        s=60,
        legend='full'
    )

    handles, legend_labels = plt.gca().get_legend_handles_labels()
    new_labels = []
    for label in legend_labels:
        if label in genre_stats:
            percentage = genre_stats[label]['percentage']
            new_labels.append(f"{label} ({percentage:.1f}%)")
        else:
            new_labels.append(label)

    plt.legend(handles, new_labels, title='Gèneres (% del dataset)',
               title_fontsize='13', fontsize='10',
               bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.title(title, fontsize=16, pad=20)
    plt.xlabel("t-SNE 1", fontsize=14)
    plt.ylabel("t-SNE 2", fontsize=14)

    for genre in unique_genres:
        genre_mask = labels == genre
        if np.sum(genre_mask) > 0:
            x_mean = X_2d[genre_mask, 0].mean()
            y_mean = X_2d[genre_mask, 1].mean()
            plt.annotate(
                f"{genre}\n({genre_stats[genre]['count']})",
                xy=(x_mean, y_mean),
                xytext=(0, 5),
                textcoords='offset points',
                ha='center',
                va='bottom',
                fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='black', lw=0.5, alpha=0.8)
            )

    plt.tight_layout()
    plt.show()


def main():
    device = setup_device()

    print("Creant dataset...")
    dataset = FilteredMelSpectrogramDataset(
        H5_PATH,
        GENRE_MAPPING,
        excluded_genres=EXCLUDED_GENRES,
        transform=None
    )

    # DEBUG: Analizar etiquetas antes de continuar
    all_labels, genre_counts = debug_labels_and_dataset(dataset)

    print(f"\nDataset creat amb {len(dataset)} mostres.")

    model = load_model(device)
    print("Model carregat correctament.")

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=8
    )

    print("\nExtraccio d'embeddings...")
    embeddings, labels = extract_embeddings_with_debug(model, dataloader, device)

    # Crear mapeo correcto
    correct_genre_mapping = create_correct_label_mapping(dataset.genres)
    print(f"\nMapping correcte de gèneres: {correct_genre_mapping}")

    print("\nExecutant t-SNE...")
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        n_iter=1000,
        learning_rate='auto',
        random_state=42,
        verbose=1
    )
    X_2d = tsne.fit_transform(embeddings)

    print("\nGenerant visualització amb estadiístiques...")
    plot_tsne_with_stats(X_2d, labels, correct_genre_mapping)


if __name__ == "__main__":
    main()