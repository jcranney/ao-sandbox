import numpy as np
import torch

zernikes = torch.tensor(np.load("zernikes.npy").astype(np.float32))
print(zernikes.shape)

def loss_min_proj(pred, y):
    phi_pred = torch.einsum("ijk,...i->...jk", zernikes, pred)
    phi_y = torch.einsum("ijk,...i->...jk", zernikes, y)
    l1 = ((phi_pred - phi_y)**2).mean()
    l2 = ((-torch.flip(phi_pred, (-2, -1)) - phi_y)**2).mean()
    return torch.min(l1, l2)

def loss_abs_only(pred, y):
    return ((pred.abs() - y.abs())**2).mean()
