from tqdm.auto import tqdm
import wandb
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import f1_score


def train(model, train_loader, val_loader, criterion, optimizer, config, device="cuda:0"):
    """
    Funció principal d'entrenament per al classificador de gèneres.

    Args:
        model: El model GRU bidireccional
        train_loader: DataLoader per a les dades d'entrenament
        val_loader: DataLoader per a les dades de validació
        criterion: Funció de pèrdua
        optimizer: Algorisme d'optimització
        config: Configuració amb els hiperparàmetres
        device: Dispositiu on executar l'entrenament ("cuda" o "cpu")
    """

    # Tell wandb to watch what the model gets up to: gradients, weights, and more!
    wandb.watch(model, criterion, log="all", log_freq=10)

    total_batches = len(train_loader) * config.epochs
    example_ct = 0  # Numero d'exemples vistos
    batch_ct = 0

    scheduler_step = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.step_lr_step_size,
                                                     gamma=config.step_lr_gamma)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=config.reduce_lr_factor, threshold=1e-4,
                                  threshold_mode='abs', verbose=True)
    best_val = float('inf')
    early_stop_patience = 6
    no_improve = 0

    for epoch in tqdm(range(config.epochs)):
        # Fixem model al training mode
        model.train()

        # Track epoch loss
        epoch_loss = 0.0

        for _, (spectrograms, labels) in enumerate(train_loader):
            loss = train_batch(spectrograms, labels, model, optimizer, criterion, device)
            example_ct += len(spectrograms)
            batch_ct += 1
            epoch_loss += loss.item()

            # Reportem metrics cada 25th batch
            if ((batch_ct + 1) % 25) == 0:
                train_log(loss, example_ct, epoch)

        torch.cuda.empty_cache()

        # Log epoch average loss
        avg_epoch_loss = epoch_loss / len(train_loader)
        wandb.log({"epoch": epoch, "epoch_avg_loss": avg_epoch_loss}, step=example_ct)
        print(f"Epoch {epoch} average loss: {avg_epoch_loss:.4f}")

        # Validation despres de cada epoch
        val_loss, val_acc, val_f1_weighted, val_f1_macro = validate(model, val_loader, criterion, device)
        wandb.log({"val_loss": val_loss, "val_accuracy": val_acc, "val_f1_weighted": val_f1_weighted,
                   "val_f1_macro": val_f1_macro}, step=example_ct)
        print(
            f"Validation loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}, F1 Score Weighted: {val_f1_weighted:.4f}, F1 Score Macro: {val_f1_macro:.4f}")

        scheduler.step(val_loss)
        scheduler_step.step()
        current_lr = optimizer.param_groups[0]['lr']
        wandb.log({"learning_rate": current_lr}, step=example_ct)

        # Early stopping logic
        if val_loss < best_val:
            best_val = val_loss
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= early_stop_patience:
                print(f"Early stopping at epoch {epoch}")
                break


def train_batch(spectrograms, labels, model, optimizer, criterion, device="cuda:0"):
    """
    Entrena un únic lot de dades.

    Args:
        spectrograms: Lot d'espectrogrames d'entrada
        labels: Etiquetes de gènere corresponents
        model: El model de xarxa neuronal
        optimizer: Algorisme d'optimització
        criterion: Funció de pèrdua
        device: Dispositiu on executar l'entrenament

    Returns:
        loss: El valor de la pèrdua per aquest lot
    """

    spectrograms, labels = spectrograms.to(device), labels.to(device)

    # Forward pass
    outputs = model(spectrograms)

    if outputs.dim() > 2:
        outputs = outputs.mean(dim=1)  # Mitjana al llarg de la dimensio temporal

    loss = criterion(outputs, labels)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()

    # Step amb optimizer
    optimizer.step()

    return loss


def train_log(loss, example_ct, epoch):
    """
    Registra les mètriques d'entrenament a wandb.

    Args:
        loss: Pèrdua del lot actual
        example_ct: Nombre d'exemples vistos fins ara
        epoch: Número de l'època actual
    """

    wandb.log({"epoch": epoch, "batch_loss": loss.item()}, step=example_ct)
    print(f"Loss after {str(example_ct).zfill(5)} examples: {loss.item():.3f}")


def validate(model, loader, criterion, device="cuda:0"):
    """
    Valida el model en un conjunt de dades de validació.

    Returns:
        avg_loss: Pèrdua mitjana en el conjunt de validació
        accuracy: Precisió de classificació en el conjunt de validació
        f1_w: Puntuació F1 en el conjunt de validació (mitjana weighted)
        f1_m: " " (mitjana macro)
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
    f1_w = f1_score(all_labels, all_preds, average='weighted')
    f1_m = f1_score(all_labels, all_preds, average='macro')

    return avg_loss, accuracy, f1_w, f1_m

