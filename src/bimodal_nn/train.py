from data import Data
from model import SimpleNet
from loss import loss_min_proj as loss_fn
import torch
from torch import nn, optim
import matplotlib.pyplot as plt
import numpy as np

learning_rate = 1e-3
batch_size = 1000
epochs = 50

data = Data("out.npz", batch_size=batch_size)
model = SimpleNet()

try:
    model.load_state_dict(torch.load("backup.model", weights_only=True))
except FileNotFoundError:
    print("no backup model found, starting from scratch")

optimizer = optim.Adam(model.parameters(), lr=learning_rate)


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    if model:
        model.eval()
    num_batches = len(dataloader)
    test_loss = 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    err_modal = []
    with torch.no_grad():
        for X, y in dataloader:
            if model:
                pred = model(X)
            else:
                pred = y*0.0
            test_loss += loss_fn(pred, y).item()
            err_modal.append((pred.abs()-y.abs()).abs().mean(dim=0)[None, ...])
    err = torch.concat(err_modal, dim=0).mean(dim=0)
    plt.plot(err)
    plt.savefig("tmp.png")
    test_loss /= num_batches
    print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")

test_loop(data.test_dataloader, None, loss_fn)
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(data.train_dataloader, model, loss_fn, optimizer)
    test_loop(data.test_dataloader, model, loss_fn)

print("Done!")
# not sure if this will work but who cares:
torch.save(model.state_dict(), "backup.model")