"""
Test script for EvoGrad framework enhancements.
Tests: DataLoader, recursive .to(), initialization, and serialization.
"""

import os
import sys
import torch
import torch.nn as nn

# Add parent dir to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evograd import Tensor, auto_device
from evograd.core import TensorDataset, DataLoader, save_model, load_model
from evograd.nn import Sequential, Linear, ReLU, Softmax
from evograd.nn import init, optim

def test_framework():
    device = auto_device()
    print(f"Testing on device: {device}")

    # 1. Test DataLoader
    print("\n[1] Testing DataLoader...")
    X = Tensor.randn((100, 10))
    y = Tensor.arange(100)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    for i, (bx, by) in enumerate(loader):
        print(f"  Batch {i}: X shape {bx.shape}, y shape {by.shape}")
        assert bx.shape[0] == 32 if i < 3 else bx.shape[0] == 4
    print("  DataLoader OK.")

    # 2. Test Recursive .to()
    print("\n[2] Testing recursive .to()...")
    # Start on CPU
    model = Sequential(
        Linear(10, 64, device="cpu"),
        ReLU(),
        Linear(64, 2, device="cpu"),
        Softmax()
    )
    # Move to target device
    model.to(device)
    for p in model.parameters():
        assert str(p.device).startswith(device) or device == "cpu"
    print(f"  Model moved to {device} recursively OK.")

    # 3. Test Initialization
    print("\n[3] Testing initialization...")
    # Get weights before init
    w_before = model.parameters()[0].data.clone()
    init.xavier_uniform(model)
    w_after = model.parameters()[0].data
    assert not torch.equal(w_before, w_after)
    print("  Xavier initialization applied OK.")

    # 4. Test Serialization
    print("\n[4] Testing serialization...")
    path = "test_model.pt"
    save_model(model, path)
    
    # Create new model and load
    new_model = Sequential(
        Linear(10, 64, device=device),
        ReLU(),
        Linear(64, 2, device=device),
        Softmax()
    )
    load_model(new_model, path)
    
    for p1, p2 in zip(model.parameters(), new_model.parameters()):
        assert torch.equal(p1.data, p2.data)
    
    os.remove(path)
    print("  Save/Load model weights OK.")

    # 5. Test Optimizer
    print("\n[5] Testing Optimizer integration...")
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    bx, by = next(iter(loader))
    bx, by = bx.to(device), by.to(device)
    
    logits = model(bx)
    loss = torch.nn.functional.cross_entropy(logits.data, by.data.long())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("  Optimizer step OK.")

    print("\n" + "="*30)
    print("  ALL FRAMEWORK TESTS PASSED")
    print("="*30)

if __name__ == "__main__":
    # Workaround for OpenMP error on some macOS versions
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    test_framework()
