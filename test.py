import wandb
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def test(model, test_loader, device="cpu", save=True):
    # Set model to evaluation mode
    model.eval()

    all_predictions = []
    all_labels = []

    # Run the model on test examples
    with torch.no_grad():
        correct, total = 0, 0
        for spectrograms, labels in test_loader:
            spectrograms, labels = spectrograms.to(device), labels.to(device)

            # Same as in train_batch, prepare dummy prev_tokens for the model
            batch_size = spectrograms.size(0)
            dummy_prev_tokens = torch.zeros((batch_size, 1), dtype=torch.long, device=device)

            outputs = model(spectrograms, dummy_prev_tokens)

            # Handle different output dimensions
            if outputs.dim() > 2:
                outputs = outputs.mean(dim=1)  # Average across time dimension

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Store predictions and labels for detailed metrics
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        accuracy = correct / total
        print(f"Accuracy of the model on the {total} " +
              f"test audio samples: {accuracy:.2%}")

        # Log test accuracy to wandb
        wandb.log({"test_accuracy": accuracy})

        # Get the dataset's genre dictionary to map indices to genre names
        genre_dict = {v: k for k, v in test_loader.dataset.genre_dict.items()} if hasattr(test_loader.dataset,
                                                                                          'genre_dict') else None


    if save:
        # Save the model in the exchangeable ONNX format
        try:
            # Keep a single batch for ONNX export
            for spec_batch, _ in test_loader:
                sample_input = spec_batch.to(device)
                sample_prev_tokens = torch.zeros((sample_input.size(0), 1), dtype=torch.long, device=device)
                break

            # Export the model
            torch.onnx.export(
                model,  # model being run
                (sample_input, sample_prev_tokens),  # model input (as a tuple for multiple inputs)
                "model.onnx",  # where to save the model
                export_params=True,  # store the trained parameter weights inside the model file
                opset_version=10,  # the ONNX version to export the model to
                do_constant_folding=True,  # whether to execute constant folding for optimization
                input_names=['input_spectrograms', 'input_prev_tokens'],  # the model's input names
                output_names=['output'],  # the model's output names
                dynamic_axes={
                    'input_spectrograms': {0: 'batch_size', 1: 'time_steps'},
                    'input_prev_tokens': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            wandb.save("model.onnx")
            print("Model exported to ONNX format successfully")
        except Exception as e:
            print(f"Error exporting model to ONNX: {e}")
