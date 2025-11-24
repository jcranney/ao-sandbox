from data import Data
from model import SimpleNet
from loss import loss_min_proj as loss_fn
# from loss import loss_abs_only as loss_fn
import torch
from torch import optim
import matplotlib.pyplot as plt
import os

if os.environ.get("TORCH_DEVICE"):
    device = int(os.environ["TORCH_DEVICE"])
else:
    device = "cpu"

learning_rate = 1e-3
batch_size = 1000
epochs = 50

data = Data("out.npz", batch_size=batch_size)
model = SimpleNet().to(device=device)

# try:
#     model.load_state_dict(torch.load("backup.model", weights_only=True))
# except FileNotFoundError:
#     print("no backup model found, starting from scratch")

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
    out_list = []
    with torch.no_grad():
        for X, y in dataloader:
            if model:
                pred = model(X)
            else:
                pred = y*0.0
            test_loss += loss_fn(pred, y).item()
            out_list.append(torch.concat([y[None, ...],pred[None, ...]], dim=0))
    out = torch.concat(out_list, dim=1)
    y_true = out[0, ...].std(dim=0)
    y_pred = out[1, ...].std(dim=0)
    err = (out[0, ...] - out[1, ...]).std(dim=0)
    plt.close("all")
    plt.plot(y_true.T, "b", label="y_true")
    plt.plot(y_pred.T, "g", label="y_pred")
    plt.plot(err.T, "r", label="err")
    plt.legend()
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