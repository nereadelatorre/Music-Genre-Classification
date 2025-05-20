from tqdm.auto import tqdm
import wandb
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import f1_score


def train(model, train_loader, val_loader, criterion, optimizer, config, device="cuda:0"):
    """
    Main training function for the music genre classifier with GRU bidirectional architecture

    Args:
        model: The GRU bidirectional model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimization algorithm
        config: Configuration with hyperparameters
        device: Device to run training on ("cuda" or "cpu")
    """
    # Tell wandb to watch what the model gets up to: gradients, weights, and more!
    wandb.watch(model, criterion, log="all", log_freq=10)

    # Run training and track with wandb
    total_batches = len(train_loader) * config.epochs
    example_ct = 0  # number of examples seen
    batch_ct = 0

    scheduler_step = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.step_lr_step_size,
                                                     gamma=config.step_lr_gamma)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=config.reduce_lr_factor, threshold=1e-4,
                                  threshold_mode='abs', verbose=True)  # bajará la LR cuando la val_loss deje de mejorar
    best_val = float('inf')
    early_stop_patience = 6
    no_improve = 0

    for epoch in tqdm(range(config.epochs)):
        # Set model to training mode
        model.train()

        # Track epoch loss
        epoch_loss = 0.0

        for _, (spectrograms, labels) in enumerate(train_loader):
            loss = train_batch(spectrograms, labels, model, optimizer, criterion, device)
            example_ct += len(spectrograms)
            batch_ct += 1
            epoch_loss += loss.item()

            # Report metrics every 25th batch
            if ((batch_ct + 1) % 25) == 0:
                train_log(loss, example_ct, epoch)

        # Log epoch average loss
        avg_epoch_loss = epoch_loss / len(train_loader)
        wandb.log({"epoch": epoch, "epoch_avg_loss": avg_epoch_loss}, step=example_ct)
        print(f"Epoch {epoch} average loss: {avg_epoch_loss:.4f}")

        # Run validation after each epoch
        val_loss, val_acc, val_f1 = validate(model, val_loader, criterion, device)
        wandb.log({"val_loss": val_loss, "val_accuracy": val_acc, "val_f1": val_f1}, step=example_ct)
        print(f"Validation loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}, F1 Score: {val_f1:.4f}")

        scheduler.step(val_loss)
        scheduler_step.step()
        current_lr = optimizer.param_groups[0]['lr']
        wandb.log({"learning_rate": current_lr}, step=example_ct)

        # Early stopping logic
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), "best_model.pt")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= early_stop_patience:
                print(f"Early stopping at epoch {epoch}")
                break


def train_batch(spectrograms, labels, model, optimizer, criterion, device="cuda:0"):
    """
    Trains a single batch of data

    Args:
        spectrograms: Batch of input spectrograms
        labels: Corresponding genre labels
        model: The neural network model
        optimizer: Optimization algorithm
        criterion: Loss function
        device: Device to run training on

    Returns:
        loss: The loss value for this batch
    """
    # Move data to the appropriate device
    spectrograms, labels = spectrograms.to(device), labels.to(device)
    # print("Input NaN?", torch.isnan(spectrograms).any())
    # print("Input Inf?", torch.isinf(spectrograms).any())

    # Forward pass
    outputs = model(spectrograms)

    if outputs.dim() > 2:
        # For bidirectional GRU classification, we could:
        # 1. Take the last hidden state from each direction (already combined in many implementations)
        # 2. Or take the mean across time for a global representation
        outputs = outputs.mean(dim=1)  # Average across time dimension

    loss = criterion(outputs, labels)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()

    # Step with optimizer
    optimizer.step()

    return loss


def train_log(loss, example_ct, epoch):
    """
    Logs training metrics to wandb

    Args:
        loss: Current batch loss
        example_ct: Number of examples seen so far
        epoch: Current epoch number
    """
    wandb.log({"epoch": epoch, "batch_loss": loss.item()}, step=example_ct)
    print(f"Loss after {str(example_ct).zfill(5)} examples: {loss.item():.3f}")


def validate(model, loader, criterion, device="cuda:0"):
    """
    Validates the model on a validation dataset

    Returns:
        avg_loss: Average loss on validation set
        accuracy: Classification accuracy on validation set
        f1: F1 score on validation set (macro average)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for spectrograms, labels in loader:
            spectrograms, labels = spectrograms.to(device), labels.to(device)

            outputs = model(spectrograms)
            if outputs.dim() > 2:
                outputs = outputs.mean(dim=1)

            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            all_preds.append(predicted.cpu())
            all_labels.append(labels.cpu())

    avg_loss = total_loss / len(loader)
    accuracy = correct / total

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    f1 = f1_score(all_labels, all_preds, average='weighted')  # or 'weighted'

    return avg_loss, accuracy, f1


"""Eliminamos los "dummy_prev_tokens" y simplificado el forward pass del modelo porque estabas usando un GRU bidireccional, 
que no requiere tokens previos artificiales como entrada. Estos tokens son típicos en modelos generativos o Transformers, 
pero innecesarios para un GRU que procesa directamente la secuencia completa de espectrogramas. Al eliminar este código, 
el flujo de datos es más limpio y refleja mejor la arquitectura real del modelo, 
evitando parámetros innecesarios que podrían causar confusión o errores de implementación."""