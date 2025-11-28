from data import Data
from model import SimpleNet
from loss import loss_min_proj as loss_fn
from loss import rms_wfe
import datetime
# from loss import loss_abs_only as loss_fn
import torch
from torch import optim
import matplotlib.pyplot as plt
import os
import seaborn
from torch.utils.tensorboard import SummaryWriter

learning_rate = float(os.environ.get("LEARNING_RATE", default="1e-3"))
batch_size = int(os.environ.get("BATCH_SIZE", default="1000"))
epochs = 1000
DO_PLOTS: bool = False


if os.environ.get("TORCH_DEVICE"):
    device = int(os.environ["TORCH_DEVICE"])
else:
    device = "cpu"

data = Data("out.npz", batch_size=batch_size)
model = SimpleNet().to(device=device)

experiment_name: str = (
    f"{datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}_"
    f"LR{learning_rate:0.4f}_"
    f"BS{batch_size:0d}_"
    f"MW{model.width:0d}_"
    f"MD{model.depth:0d}"
)

writer = SummaryWriter(log_dir=f"./runs/{experiment_name}")

seaborn.set_theme()


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
    batch = 1
    wfe = 0.0
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        wfe += rms_wfe(pred, y).item()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    wfe /= (batch+1)
    print(f"RMS WFE (train): {wfe:>7f}")
    return wfe


def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    if model:
        model.eval()
    num_batches = len(dataloader)
    test_loss = 0
    wfe = 0.0

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
            out_list.append(
                torch.stack([y, pred], dim=0
            ).clone().cpu().detach())
            wfe += rms_wfe(pred, y).item()
    y_true, y_pred = torch.concat(out_list, dim=1)
    # plot the modal scatters:
    if DO_PLOTS:
        plt.close("all")
        _, axs = plt.subplots(4, 3, figsize=(10,10))
        for i, ax in enumerate(axs.flatten()[:10]):
            ax.plot(y_true[:, i], y_pred[:, i], "k.")
            ax.axline((0.0, 0.0), slope=1.0)
            ax.set_title(f"mode: {i}")
            ax.set_aspect("equal")
            range = 1.1*y_true[:, i].abs().max()+1e-3
            ax.set_xlim([-range, range])
            ax.set_ylim([-range, range])
        plt.tight_layout()
        plt.savefig("tmp2.png", dpi=300)
        
        
        # y_true.shape = [NSAMPLES, NMODES]
        y_true_std = y_true.std(dim=0)
        y_pred_std = y_pred.std(dim=0)
        y_pred_flip = y_pred.clone()
        if y_pred_flip.shape[1] > 10:
            raise ValueError(
                "Number of modes is more than 10, so the error calculation"
                " will be invalid."
            )
        y_pred_flip[:, 3:6] *= -1.0
        err_std = torch.amin(
            torch.stack([
                y_pred - y_true,
                y_pred_flip - y_true,
            ])**2,
            dim=0
        ).mean(dim=0)**0.5

        plt.close("all")
        plt.plot(y_true_std, label="y_true")
        plt.plot(y_pred_std, label="y_pred")
        plt.plot(err_std, label="err")
        plt.xlabel("Zernike mode (#)")
        plt.ylabel("Mode amplitude RMS (rad)")
        plt.legend()
        plt.tight_layout()
        plt.savefig("tmp.png", dpi=200)
    
    test_loss /= num_batches
    print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")
    wfe /= num_batches
    print(f"RMS WFE (test):  {wfe:>7f}")
    return wfe

test_loss = 0.0
test_loop(data.test_dataloader, None, loss_fn)
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loss = train_loop(data.train_dataloader, model, loss_fn, optimizer)
    writer.add_scalar("Loss/train", train_loss, t)
    test_loss = test_loop(data.test_dataloader, model, loss_fn)
    writer.add_scalar("Loss/test", test_loss, t)
    writer.flush()

print("Done!")
print(f"{test_loss:0.4e}")
# not sure if this will work but who cares:
torch.save(model.state_dict(), "backup.model")
