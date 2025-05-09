from tqdm.auto import tqdm
import wandb
import torch


def train(model, loader, criterion, optimizer, config, device="cpu"):
    # Tell wandb to watch what the model gets up to: gradients, weights, and more!
    wandb.watch(model, criterion, log="all", log_freq=10)

    # Run training and track with wandb
    total_batches = len(loader) * config.epochs
    example_ct = 0  # number of examples seen
    batch_ct = 0
    for epoch in tqdm(range(config.epochs)):
        # Set model to training mode
        model.train()

        # Track epoch loss
        epoch_loss = 0.0

        for _, (spectrograms, labels) in enumerate(loader):
            loss = train_batch(spectrograms, labels, model, optimizer, criterion, device)
            example_ct += len(spectrograms)
            batch_ct += 1
            epoch_loss += loss.item()

            if ((batch_ct + 1) % 25) == 0:
                train_log(loss, example_ct, epoch)

        # Log epoch average loss
        avg_epoch_loss = epoch_loss / len(loader)
        wandb.log({"epoch": epoch, "epoch_avg_loss": avg_epoch_loss}, step=example_ct)
        print(f"Epoch {epoch} average loss: {avg_epoch_loss:.4f}")


def train_batch(spectrograms, labels, model, optimizer, criterion, device="cpu"):
    spectrograms, labels = spectrograms.to(device), labels.to(device)

    # Forward pass
    # The model expects (batch, time, features)
    # For the baseline model, we need to prepare dummy prev_tokens
    batch_size = spectrograms.size(0)
    dummy_prev_tokens = torch.zeros((batch_size, 1), dtype=torch.long, device=device)

    outputs = model(spectrograms, dummy_prev_tokens)

    # For classification, we use the last output (or mean of all outputs)
    # If outputs has shape [batch, time, classes], take the mean along time dimension
    if outputs.dim() > 2:
        outputs = outputs.mean(dim=1)  # Average across time dimension

    loss = criterion(outputs, labels)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()

    # Step with optimizer
    optimizer.step()

    return loss


def train_log(loss, example_ct, epoch):
    # Where the magic happens
    wandb.log({"epoch": epoch, "batch_loss": loss.item()}, step=example_ct)
    print(f"Loss after {str(example_ct).zfill(5)} examples: {loss.item():.3f}")
