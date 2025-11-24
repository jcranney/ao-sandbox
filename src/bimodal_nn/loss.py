import os
import numpy as np
import torch

if os.environ.get("TORCH_DEVICE"):
    device = int(os.environ["TORCH_DEVICE"])
else:
    device = "cpu"

zernikes = torch.tensor(
    np.load("zernikes.npy").astype(np.float32),
    device=device,
)
pupil = zernikes[0, ...] > 0.5


def loss_min_proj(pred, y):
    phi_pred_a = phi_from_modes(pred)
    phi_pred_b = get_phi_cosolution(phi_pred_a)
    phi_y = phi_from_modes(y)
    l1 = ((phi_pred_a - phi_y)**2)[..., pupil].mean()
    l2 = ((phi_pred_b - phi_y)**2)[..., pupil].mean()
    return torch.min(l1, l2)

def loss_abs_only(pred, y):
    return ((pred.abs() - y.abs())**2).mean()

def get_phi_cosolution(phi_a):
    phi_b = -torch.flip(phi_a, (-2, -1))
    return phi_b

def phi_from_modes(modes):
    phi = torch.einsum("ijk,...i->...jk", zernikes, modes)
    return phi


if __name__ == "__main__":
    y_pred = torch.tensor(
        np.array([0,0,0,1.0,0,0,0,0,0,0])
    #    np.random.standard_normal((1, zernikes.shape[0]))
    ).to(torch.float32).to(device=device)
    phi_a = phi_from_modes(y_pred)
    phi_b = get_phi_cosolution(phi_a=phi_a)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(phi_a.squeeze())
    ax[1].imshow(phi_b.squeeze())
    plt.savefig("tmp.png")
    
    y_true = torch.tensor(
        np.array([0,0,0,-1.0,0,0,0,0,0,0])
    #    np.random.standard_normal((1, zernikes.shape[0]))
    ).to(torch.float32).to(device=device)
    print(loss_abs_only(y_pred, y_true))
    print(loss_min_proj(y_pred, y_true))
    