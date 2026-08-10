import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
import random
import numpy as np
from data_loader import get_dataloaders
from model import SimpleCNN


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_model(data_dir, num_epochs=50, batch_size=32, learning_rate=0.001,
                 output_path="best_model.pth", seed=42):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, classes = get_dataloaders(
        data_dir, batch_size=batch_size, num_workers=0, seed=seed
    )
    print(f"Data loaded. Classes ({len(classes)}): {classes}")

    model = SimpleCNN(num_classes=len(classes)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_acc = 0.0
    start_time = time.time()

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 10)

        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        model.eval()
        val_loss = 0.0
        val_corrects = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_corrects.double() / len(val_loader.dataset)
        print(f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), output_path)
            print(f"Saved best model to {output_path}.")

    time_elapsed = time.time() - start_time
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best Val Acc: {best_acc:.4f}")
    print(
        "NOTE: this is only a train/val split -- run evaluate.py against a "
        "held-out test set before trusting this number for a deployment decision."
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train the WM811K wafer-map defect classifier.")
    parser.add_argument("--data_dir", type=str, required=True,
                         help="Path to the WM811K image-folder dataset (root/<class>/*.png)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--output", type=str, default="best_model.pth", help="Where to save the best checkpoint")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        raise SystemExit(f"Dataset path not found: {args.data_dir}")

    train_model(args.data_dir, num_epochs=args.epochs, batch_size=args.batch_size,
                learning_rate=args.lr, output_path=args.output, seed=args.seed)
