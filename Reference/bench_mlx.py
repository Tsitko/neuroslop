#!/usr/bin/env python3
"""
Reference benchmark: MLX (Apple's ML framework optimized for Apple Silicon).
Same MLP architecture as our Swift implementation: 784 -> 128 -> 64 -> 10.
Uses Metal GPU via unified memory.
"""
import time
import os
import resource
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

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
            # One-hot encode
            one_hot = np.zeros((n, 10), dtype=np.float32)
            one_hot[np.arange(n), labels] = 1.0
            return labels, one_hot

    train_images = read_idx_images(f"{data_dir}/train-images-idx3-ubyte")
    train_labels_raw, train_labels = read_idx_labels(f"{data_dir}/train-labels-idx1-ubyte")
    test_images = read_idx_images(f"{data_dir}/t10k-images-idx3-ubyte")
    test_labels_raw, test_labels = read_idx_labels(f"{data_dir}/t10k-labels-idx1-ubyte")

    return (train_images, train_labels_raw, train_labels,
            test_images, test_labels_raw, test_labels)

class MLP(nn.Module):
    """Same architecture: 784 -> 128 (ReLU) -> 64 (ReLU) -> 10"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def __call__(self, x):
        x = nn.relu(self.fc1(x))
        x = nn.relu(self.fc2(x))
        x = self.fc3(x)  # raw logits, softmax in loss
        return x

def loss_fn(model, x, y):
    logits = model(x)
    return mx.mean(nn.losses.cross_entropy(logits, y))

def compute_accuracy(model, images, labels, batch_size=1000):
    correct = 0
    total = 0
    for i in range(0, len(images), batch_size):
        x = mx.array(images[i:i+batch_size])
        y = labels[i:i+batch_size]
        logits = model(x)
        preds = mx.argmax(logits, axis=1)
        mx.eval(preds)
        correct += (np.array(preds) == y).sum()
        total += len(y)
    return correct / total

def main():
    print(f"=== MLX Reference Benchmark ===")
    print(f"MLX backend: {mx.default_device()}")
    print(f"Memory RSS: {get_rss_mb():.1f} MB")

    import sys
    train_subset = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    test_subset = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    num_epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 10

    # Load data
    print("\nLoading MNIST...")
    train_images, train_labels_raw, _, test_images, test_labels_raw, _ = load_mnist()

    if train_subset > 0:
        train_images = train_images[:train_subset]
        train_labels_raw = train_labels_raw[:train_subset]
    if test_subset > 0:
        test_images = test_images[:test_subset]
        test_labels_raw = test_labels_raw[:test_subset]

    print(f"Train: {len(train_images)}, Test: {len(test_images)}")
    print(f"Memory RSS: {get_rss_mb():.1f} MB")

    # Create model
    model = MLP()
    mx.eval(model.parameters())
    optimizer = optim.SGD(learning_rate=0.01)
    loss_and_grad = nn.value_and_grad(model, loss_fn)

    print(f"\nTraining 784->128->64->10 (MLX, {mx.default_device()})...")
    print(f"Config: lr=0.01, batch_size=64, epochs={num_epochs}, SGD, CrossEntropy")

    batch_size = 64
    num_samples = len(train_images)

    for epoch in range(num_epochs):
        t0 = time.time()

        # Shuffle
        perm = np.random.permutation(num_samples)
        train_images_shuffled = train_images[perm]
        train_labels_shuffled = train_labels_raw[perm]

        epoch_loss = 0.0
        num_batches = 0

        for i in range(0, num_samples, batch_size):
            x = mx.array(train_images_shuffled[i:i+batch_size])
            y = mx.array(train_labels_shuffled[i:i+batch_size])

            loss, grads = loss_and_grad(model, x, y)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)

            epoch_loss += loss.item()
            num_batches += 1

        elapsed = (time.time() - t0) * 1000  # ms
        avg_loss = epoch_loss / num_batches
        accuracy = compute_accuracy(model, test_images, test_labels_raw)

        print(f"Epoch {epoch+1}/{num_epochs} — loss: {avg_loss:.4f}, accuracy: {accuracy*100:.2f}%, "
              f"time: {elapsed:.1f}ms, mem: {get_rss_mb():.1f} MB")

    final_acc = compute_accuracy(model, test_images, test_labels_raw)
    print(f"\nFinal test accuracy: {final_acc*100:.2f}%")
    print(f"Memory RSS: {get_rss_mb():.1f} MB")

if __name__ == "__main__":
    main()
