import wandb
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd


def test(model, test_loader, config, device="cuda:0", save=True):
    model.eval()
    all_preds, all_labels = [], []

    # 1) Inferencia
    with torch.no_grad():
        for specs, labels in test_loader:
            specs, labels = specs.to(device), labels.to(device)
            outputs = model(specs)
            if outputs.dim() > 2:
                outputs = outputs.mean(dim=1)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 2) Accuracy
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    print(f"Test accuracy: {accuracy:.2%}")
    wandb.log({"test_accuracy": accuracy})
    if save:
        wandb.run.summary["test_accuracy"] = accuracy

    # F1_Score
    f1 = f1_score(all_labels, all_preds, average='weighted')
    print(f"F1-Score (weighted): {f1:.4f}")
    wandb.log({"test_f1_score": f1})
    if save:
        wandb.run.summary["test_f1_score"] = f1

    # 3) Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
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

    genres = [genre for genre in dicc.values() if genre not in excluded_genres]

    print(genres)

    # Create both normalized and non-normalized confusion matrices
    plt.figure(figsize=(18, 8))

    # Raw counts confusion matrix
    plt.subplot(1, 2, 1)
    if genres:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=genres, yticklabels=genres)
    else:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (Raw Counts)')

    # Normalized confusion matrix (row normalization - shows recall)
    plt.subplot(1, 2, 2)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # Replace NaN with 0 (happens when a class has 0 samples)
    cm_normalized = np.nan_to_num(cm_normalized)

    if genres:
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=genres, yticklabels=genres, vmin=0, vmax=1)
    else:
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', vmin=0, vmax=1)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Normalized Confusion Matrix (Row Normalization)')

    # Save locally and to wandb
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    if save:
        wandb.log({"confusion_matrix": wandb.Image('confusion_matrix.png')})

    # 4) Classification Report
    if genres:
        report = classification_report(
            all_labels, all_preds,
            target_names=genres,
            output_dict=True
        )
    else:
        report = classification_report(
            all_labels, all_preds,
            output_dict=True
        )
    """
    # Log metrics to wandb
    if save:
        for label, metrics in report.items():
            if isinstance(metrics, dict):
                for metric_name, value in metrics.items():
                    wandb.run.summary[f"{label}_{metric_name}"] = value

    """

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds,
                                target_names=genres if genres else None))

    # 5) Per-class accuracy
    per_class_acc = {}
    for i in range(len(np.unique(all_labels))):
        mask = np.array(all_labels) == i
        if mask.sum() > 0:  # avoid division by zero
            class_acc = np.mean(np.array(all_preds)[mask] == i)
            class_name = genres[i] if genres else f"Class {i}"
            per_class_acc[class_name] = class_acc
            print(f"Accuracy for {class_name}: {class_acc:.2%}")
    """
    if save:
        wandb.log({"per_class_accuracy": per_class_acc})
    """
    # 6) Export ONNX (opcional)
    if save:
        try:
            sample_input, _ = next(iter(test_loader))
            sample_input = sample_input.to(device)
            torch.onnx.export(
                model,
                sample_input,
                "model.onnx",
                export_params=True,
                opset_version=10,
                do_constant_folding=True,
                input_names=['input_spectrograms'],
                output_names=['output'],
                dynamic_axes={
                    'input_spectrograms': {0: 'batch_size', 1: 'time_steps'},
                    'output': {0: 'batch_size'}
                }
            )
            wandb.save("model.onnx")
            print("ONNX exportado correctamente")
        except Exception as e:
            print(f"Error exportando ONNX: {e}")

    return accuracy, cm, report