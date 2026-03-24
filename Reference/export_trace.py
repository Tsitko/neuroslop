#!/usr/bin/env python3
"""
Export a deterministic trace from PyTorch for comparison with our implementation.
Saves: initial weights, forward outputs, loss, gradients — step by step.
"""
import numpy as np
import torch
import torch.nn as tnn
import os, struct

def load_mnist_subset(data_dir="Data/mnist", n_train=100, n_test=100):
    """Load a tiny subset for exact comparison."""
    def read_images(path, n):
        with open(path, 'rb') as f:
            f.read(16)  # skip header
            data = np.frombuffer(f.read(n * 784), dtype=np.uint8).reshape(n, 784)
            return data.astype(np.float32) / 255.0

    def read_labels(path, n):
        with open(path, 'rb') as f:
            f.read(8)
            return np.frombuffer(f.read(n), dtype=np.uint8)

    train_x = read_images(f"{data_dir}/train-images-idx3-ubyte", n_train)
    train_y = read_labels(f"{data_dir}/train-labels-idx1-ubyte", n_train)
    test_x = read_images(f"{data_dir}/t10k-images-idx3-ubyte", n_test)
    test_y = read_labels(f"{data_dir}/t10k-labels-idx1-ubyte", n_test)
    return train_x, train_y, test_x, test_y

def save_matrix(path, data, rows, cols):
    """Save matrix as binary: rows(i32) cols(i32) data(float32[])"""
    with open(path, 'wb') as f:
        f.write(struct.pack('<ii', rows, cols))
        f.write(data.astype(np.float32).tobytes())

def main():
    out_dir = "Reference/trace"
    os.makedirs(out_dir, exist_ok=True)

    torch.manual_seed(42)
    np.random.seed(42)

    # Load tiny subset
    train_x, train_y, test_x, test_y = load_mnist_subset()
    batch_size = 16
    batch_x = train_x[:batch_size]
    batch_y = train_y[:batch_size]

    # Save input data
    save_matrix(f"{out_dir}/batch_x.bin", batch_x, batch_size, 784)
    save_matrix(f"{out_dir}/batch_y_indices.bin",
                batch_y.astype(np.float32).reshape(1, -1), 1, batch_size)

    # One-hot for our implementation
    batch_y_onehot = np.zeros((batch_size, 10), dtype=np.float32)
    batch_y_onehot[np.arange(batch_size), batch_y] = 1.0
    save_matrix(f"{out_dir}/batch_y_onehot.bin", batch_y_onehot, batch_size, 10)

    # Create model with KNOWN initialization
    model = tnn.Sequential(
        tnn.Linear(784, 128),
        tnn.ReLU(),
        tnn.Linear(128, 64),
        tnn.ReLU(),
        tnn.Linear(64, 10),
    )

    # Save initial weights (PyTorch stores weights transposed: [out, in])
    # Our implementation: weights are [in, out]
    layers = [model[0], model[2], model[4]]  # Linear layers
    for i, layer in enumerate(layers):
        # PyTorch weight: [out_features, in_features] — need to transpose for our format
        w = layer.weight.detach().numpy().T  # now [in, out]
        b = layer.bias.detach().numpy().reshape(1, -1)  # [1, out]
        save_matrix(f"{out_dir}/w{i}_init.bin", w, w.shape[0], w.shape[1])
        save_matrix(f"{out_dir}/b{i}_init.bin", b, 1, b.shape[1])
        print(f"Layer {i}: weight {w.shape}, bias {b.shape}")
        print(f"  w[0,:5] = {w[0,:5]}")
        print(f"  b[0,:5] = {b[0,:5]}")

    # Forward pass — capture intermediates
    x = torch.tensor(batch_x)

    # Layer 0: linear + relu
    z0 = model[0](x)
    a0 = model[1](z0)
    save_matrix(f"{out_dir}/z0.bin", z0.detach().numpy(), batch_size, 128)
    save_matrix(f"{out_dir}/a0.bin", a0.detach().numpy(), batch_size, 128)

    # Layer 1: linear + relu
    z1 = model[2](a0)
    a1 = model[3](z1)
    save_matrix(f"{out_dir}/z1.bin", z1.detach().numpy(), batch_size, 64)
    save_matrix(f"{out_dir}/a1.bin", a1.detach().numpy(), batch_size, 64)

    # Layer 2: linear (raw logits, no softmax)
    z2 = model[4](a1)
    save_matrix(f"{out_dir}/z2_logits.bin", z2.detach().numpy(), batch_size, 10)

    # Softmax output (what our implementation produces)
    softmax_out = torch.softmax(z2, dim=1)
    save_matrix(f"{out_dir}/softmax_out.bin", softmax_out.detach().numpy(), batch_size, 10)

    print(f"\nForward pass:")
    print(f"  z0[0,:5] = {z0[0,:5].detach().numpy()}")
    print(f"  a0[0,:5] = {a0[0,:5].detach().numpy()}")
    print(f"  z2[0,:5] = {z2[0,:5].detach().numpy()}")
    print(f"  softmax[0] = {softmax_out[0].detach().numpy()}")

    # Loss — CrossEntropy
    # PyTorch: CE(logits, labels) = -log(softmax(logits)[label])
    criterion = tnn.CrossEntropyLoss()
    y_tensor = torch.tensor(batch_y.astype(np.int64))
    loss = criterion(z2, y_tensor)
    print(f"\n  PyTorch CE loss: {loss.item():.6f}")

    # Our CE loss: -sum(target_onehot * log(softmax_output)) / batch_size
    our_loss = -np.sum(batch_y_onehot * np.log(np.clip(softmax_out.detach().numpy(), 1e-7, 1))) / batch_size
    print(f"  Our CE loss formula: {our_loss:.6f}")

    # Backward — compute gradients
    model.zero_grad()
    loss.backward()

    # Save weight gradients
    for i, layer in enumerate(layers):
        wg = layer.weight.grad.numpy().T  # transpose back to [in, out]
        bg = layer.bias.grad.numpy().reshape(1, -1)
        save_matrix(f"{out_dir}/w{i}_grad.bin", wg, wg.shape[0], wg.shape[1])
        save_matrix(f"{out_dir}/b{i}_grad.bin", bg, 1, bg.shape[1])
        print(f"\nLayer {i} gradients:")
        print(f"  dW[0,:5] = {wg[0,:5]}")
        print(f"  db[0,:5] = {bg[0,:5]}")

    # Combined softmax+CE gradient: dL/dz = (softmax - onehot) / batch_size
    combined_grad = (softmax_out.detach().numpy() - batch_y_onehot) / batch_size
    print(f"\n  Combined softmax+CE gradient[0] = {combined_grad[0]}")

    # Save combined gradient for our comparison
    combined_grad_data = combined_grad.astype(np.float32)
    save_matrix(f"{out_dir}/dz2.bin", combined_grad_data, batch_size, 10)

    # SGD: do one step with same initial weights
    lr = 0.01
    torch.manual_seed(42)
    model_step = tnn.Sequential(
        tnn.Linear(784, 128), tnn.ReLU(),
        tnn.Linear(128, 64), tnn.ReLU(),
        tnn.Linear(64, 10),
    )
    opt = torch.optim.SGD(model_step.parameters(), lr=lr)
    opt.zero_grad()
    out = model_step(x)
    loss_step = criterion(out, y_tensor)
    loss_step.backward()
    opt.step()

    layers_step = [model_step[0], model_step[2], model_step[4]]
    for i, layer in enumerate(layers_step):
        w = layer.weight.detach().numpy().T
        b = layer.bias.detach().numpy().reshape(1, -1)
        save_matrix(f"{out_dir}/w{i}_after_step.bin", w, w.shape[0], w.shape[1])
        save_matrix(f"{out_dir}/b{i}_after_step.bin", b, 1, b.shape[1])

    print(f"\nAfter 1 SGD step (lr={lr}):")
    print(f"  loss: {loss_step.item():.6f}")
    for i, layer in enumerate(layers_step):
        print(f"  w{i}[0,:3] = {layer.weight.detach().numpy().T[0,:3]}")

    print(f"\nTrace saved to {out_dir}/")

if __name__ == "__main__":
    main()
