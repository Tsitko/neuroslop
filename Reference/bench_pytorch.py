#!/usr/bin/env python3
"""
Reference benchmark: PyTorch with MPS (Metal Performance Shaders) backend.
Same MLP architecture as our Swift implementation: 784 -> 128 -> 64 -> 10.
Uses Metal GPU via MPS backend.
"""
import time
import os
import resource
import numpy as np
import torch
import torch.nn as tnn

def get_rss_mb():
    """Get resident set size in MB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)

def load_mnist(data_dir="Data/mnist"):
    """Load MNIST from IDX files (same files as Swift implementation)."""
    def read_idx_images(path):
        with open(path, 'rb') as f:
            magic = int.from_bytes(f.read(4), 'big')
            assert magic == 2051
            n = int.from_bytes(f.read(4), 'big')
            rows = int.from_bytes(f.read(4), 'big')
            cols = int.from_bytes(f.read(4), 'big')
            data = np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows * cols)
            return data.astype(np.float32) / 255.0

    def read_idx_labels(path):
        with open(path, 'rb') as f:
            magic = int.from_bytes(f.read(4), 'big')
            assert magic == 2049
            n = int.from_bytes(f.read(4), 'big')
            labels = np.frombuffer(f.read(), dtype=np.uint8)
            return labels

    train_images = read_idx_images(f"{data_dir}/train-images-idx3-ubyte")
    train_labels = read_idx_labels(f"{data_dir}/train-labels-idx1-ubyte")
    test_images = read_idx_images(f"{data_dir}/t10k-images-idx3-ubyte")
    test_labels = read_idx_labels(f"{data_dir}/t10k-labels-idx1-ubyte")

    return train_images, train_labels, test_images, test_labels

class MLP(tnn.Module):
    """Same architecture: 784 -> 128 (ReLU) -> 64 (ReLU) -> 10"""
    def __init__(self):
        super().__init__()
        self.fc1 = tnn.Linear(784, 128)
        self.fc2 = tnn.Linear(128, 64)
        self.fc3 = tnn.Linear(64, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # raw logits, CrossEntropyLoss handles softmax
        return x

def compute_accuracy(model, images, labels, device, batch_size=1000):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            x = torch.tensor(images[i:i+batch_size], device=device)
            y = torch.tensor(labels[i:i+batch_size].astype(np.int64), device=device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += len(y)
    model.train()
    return correct / total

def run_benchmark(device_name, train_subset=0, test_subset=0, num_epochs=10):
    device = torch.device(device_name)
    print(f"\n--- PyTorch on {device_name.upper()} ---")
    print(f"Memory RSS: {get_rss_mb():.1f} MB")

    # Load data
    train_images, train_labels, test_images, test_labels = load_mnist()

    if train_subset > 0:
        train_images = train_images[:train_subset]
        train_labels = train_labels[:train_subset]
    if test_subset > 0:
        test_images = test_images[:test_subset]
        test_labels = test_labels[:test_subset]

    print(f"Train: {len(train_images)}, Test: {len(test_images)}")

    # Create model
    model = MLP().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = tnn.CrossEntropyLoss()

    print(f"Training 784->128->64->10 (PyTorch, {device_name})...")
    print(f"Config: lr=0.01, batch_size=64, epochs={num_epochs}, SGD, CrossEntropy")

    batch_size = 64
    num_samples = len(train_images)

    for epoch in range(num_epochs):
        t0 = time.time()

        # Shuffle
        perm = np.random.permutation(num_samples)
        train_images_shuffled = train_images[perm]
        train_labels_shuffled = train_labels[perm]

        epoch_loss = 0.0
        num_batches = 0

        for i in range(0, num_samples, batch_size):
            x = torch.tensor(train_images_shuffled[i:i+batch_size], device=device)
            y = torch.tensor(train_labels_shuffled[i:i+batch_size].astype(np.int64), device=device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        if device_name == "mps":
            torch.mps.synchronize()

        elapsed = (time.time() - t0) * 1000  # ms
        avg_loss = epoch_loss / num_batches
        accuracy = compute_accuracy(model, test_images, test_labels, device)

        print(f"Epoch {epoch+1}/{num_epochs} — loss: {avg_loss:.4f}, accuracy: {accuracy*100:.2f}%, "
              f"time: {elapsed:.1f}ms, mem: {get_rss_mb():.1f} MB")

    final_acc = compute_accuracy(model, test_images, test_labels, device)
    print(f"\nFinal test accuracy: {final_acc*100:.2f}%")
    print(f"Memory RSS: {get_rss_mb():.1f} MB")

def main():
    import sys
    train_subset = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    test_subset = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    num_epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 10

    print(f"=== PyTorch Reference Benchmark ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")

    # CPU baseline
    run_benchmark("cpu", train_subset, test_subset, num_epochs)

    # MPS (Metal GPU)
    if torch.backends.mps.is_available():
        run_benchmark("mps", train_subset, test_subset, num_epochs)
    else:
        print("\nMPS not available, skipping GPU benchmark")

if __name__ == "__main__":
    main()
