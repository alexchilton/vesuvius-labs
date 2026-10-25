"""The pieces notebook 03 builds by hand, packaged for notebook 04: data, spatial split, model, training, evaluation."""
import time

import numpy as np
import tifffile
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

import vlab

DEV = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
L0, L1, P = 6, 22, 64                     # 16 layers around the surface, 64 px patches


def load(ink_tif=None):
    """Surface volume of segment w025 at level 1 (normalised), the resized pseudo-label, and the valid mask."""
    X = vlab.crop(vlab.segment_surface(1), "seg_l1", slice(None), slice(None), slice(None))
    ink = tifffile.imread(ink_tif or vlab.fetch_ink())
    Y = np.asarray(Image.fromarray(ink).resize((X.shape[2], X.shape[1]), Image.BILINEAR), np.float32) / 255.
    valid = X.max(0) > 0
    Xn = (X[L0:L1].astype(np.float32) / 255. - 0.5) / 0.25
    return Xn, Y, valid


def tiles(valid, cut, size=128):
    """Corners of the size x size tiles left of column `cut` that are at least 90% valid: the unit of labelling."""
    H = valid.shape[0]
    return [(i, j) for i in range(0, H - size + 1, size) for j in range(0, cut - size + 1, size) if valid[i:i + size, j:j + size].mean() >= 0.9]


def sample(Xn, Y, corners, n, rng, size=128):
    """n random P x P patches from inside the given tiles."""
    xs, ys = [], []
    for k in rng.integers(0, len(corners), n):
        i, j = corners[k]; a, b = rng.integers(0, size - P + 1, 2)
        xs.append(Xn[:, i + a:i + a + P, j + b:j + b + P]); ys.append(Y[i + a:i + a + P, j + b:j + b + P])
    return torch.tensor(np.stack(xs)), torch.tensor(np.stack(ys))[:, None]


def val_set(Xn, Y, valid, cut, n=512, seed=0):
    """A fixed set of patches right of `cut`, never used for training or pretraining."""
    rng, xs, ys = np.random.default_rng(seed), [], []
    H, W = valid.shape
    while len(xs) < n:
        i, j = rng.integers(0, H - P), rng.integers(cut, W - P)
        if valid[i:i + P, j:j + P].mean() >= 0.9:
            xs.append(Xn[:, i:i + P, j:j + P]); ys.append(Y[i:i + P, j:j + P])
    return torch.tensor(np.stack(xs)), torch.tensor(np.stack(ys))[:, None]


class Encoder(nn.Module):
    """3D convolutions over (layer, y, x), then a max over layers: (B, layers, H, W) -> (B, 2c, H, W)."""
    def __init__(s, c=16):
        super().__init__()
        s.net = nn.Sequential(nn.Conv3d(1, c, 3, padding=1), nn.GELU(), nn.Conv3d(c, c, 3, padding=1), nn.GELU(),
                              nn.Conv3d(c, 2 * c, 3, padding=1), nn.GELU())
    def forward(s, x):
        return s.net(x[:, None]).amax(2)


def head(c_in=32, c_out=1):
    return nn.Sequential(nn.Conv2d(c_in, c_in, 3, padding=1), nn.GELU(), nn.Conv2d(c_in, c_in, 3, padding=2, dilation=2), nn.GELU(), nn.Conv2d(c_in, c_out, 1))


def corr(net, xv, yv):
    net.eval(); out = []
    with torch.no_grad():
        for k in range(0, len(xv), 64):
            out.append(torch.sigmoid(net(xv[k:k + 64].to(DEV))).cpu())
    return float(np.corrcoef(torch.cat(out).ravel().numpy(), yv.ravel().numpy())[0, 1])


def finetune(enc, Xn, Y, corners, xv, yv, steps, seed, lr=2e-3, log_every=0):
    """Ink head on top of `enc`, trained on patches from `corners` only. Returns the held-out correlation."""
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    net = nn.Sequential(enc, head()).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr, weight_decay=1e-4); t = time.time()
    for step in range(1, steps + 1):
        net.train(); xb, yb = sample(Xn, Y, corners, 16, rng)
        loss = F.binary_cross_entropy_with_logits(net(xb.to(DEV)), yb.to(DEV))
        opt.zero_grad(); loss.backward(); opt.step()
        if log_every and step % log_every == 0:
            print(f"  step {step} loss {loss.item():.4f} val corr {corr(net, xv, yv):.3f} ({time.time() - t:.0f}s)")
    return corr(net, xv, yv), net
