"""Notebook 05: crushing a scroll cross-section as a flow, and learning to undo it.

Coordinates are normalised to [-1, 1] (x right, y down), as torch's grid_sample uses them. A deformation is the flow of a
*stationary velocity field* v: integrate dx/dt = v(x) for unit time. Its inverse is the flow of -v, and it can never fold.

    ideal spirals ──crush: flow of v──► deformed image ──U-Net──► u ≈ v ──flow of u──► uncrushed ≈ ideal
"""
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
SIZE = 256
CKPT = Path(__file__).resolve().parent / "data" / "cache" / "flow05"


def grid(size=SIZE, device=DEV):
    """The identity map as a (1, H, W, 2) grid of normalised (x, y)."""
    t = torch.linspace(-1, 1, size, device=device)
    yy, xx = torch.meshgrid(t, t, indexing="ij")
    return torch.stack([xx, yy], -1)[None]


# ------------------------------------------------------------------ the ideal scroll
def spiral_params(n, rng):
    """Random Archimedean spirals r = r0 + pitch * (angle / 2 pi): pitch in normalised units, out to radius rmax."""
    return dict(r0=rng.uniform(0.02, 0.08, n), pitch=rng.uniform(0.016, 0.045, n), rmax=rng.uniform(0.55, 0.75, n),
                phase=rng.uniform(0, 2 * math.pi, n), width=rng.uniform(0.004, 0.008, n), seed=rng.integers(0, 2**31, n))


def spiral_image(p, size=SIZE, device=DEV):
    """Render spirals with a little texture: bright sheet on dark air. Returns (B, 1, H, W) in [0, 1] and the turn index map."""
    g = grid(size, device)[0]
    r = g.norm(dim=-1)[None]; th = torch.atan2(g[..., 1], g[..., 0])[None]
    P = {k: torch.as_tensor(np.asarray(v), dtype=torch.float32, device=device)[:, None, None] for k, v in p.items() if k != "seed"}
    s = (r - P["r0"]) / P["pitch"] - ((th - P["phase"]) % (2 * math.pi)) / (2 * math.pi)   # turns along the spiral
    k = torch.round(s)
    d = (s - k).abs() * P["pitch"]                                                          # distance to the nearest wrap
    img = torch.exp(-0.5 * (d / P["width"]) ** 2) * ((r > P["r0"]) & (r < P["rmax"]) & (k >= 0)).float()
    gen = torch.Generator(device="cpu")
    tex = []
    for sd in p["seed"]:
        gen.manual_seed(int(sd))
        tex.append(torch.rand(1, 1, 16, 16, generator=gen))
    tex = F.interpolate(torch.cat(tex).to(device), size=(size, size), mode="bicubic", align_corners=True)[:, 0]
    img = img * (0.55 + 0.9 * tex).clamp(0, 1.5) * (tex > 0.18).float()                    # fibre brightness, a few gaps
    return img[:, None].clamp(0, 1), k


def spiral_points(p, i, n, rng):
    """n points along spiral i (uniform in arc length), with the turn each one belongs to."""
    r0, pitch, rmax, ph = (float(p[k][i]) for k in ("r0", "pitch", "rmax", "phase"))
    turns = (rmax - r0) / pitch
    u = np.sqrt(rng.uniform(0, 1, n)) * turns                     # arc length grows ~ r, so sample turns with density ~ r
    r = r0 + pitch * u
    th = ph + 2 * math.pi * (u % 1)
    return np.stack([r * np.cos(th), r * np.sin(th)], 1).astype(np.float32), np.floor(u).astype(int)


def turn_of(pts, p, i):
    """Which wrap of the ideal spiral i a point is closest to (in the radial direction)."""
    r0, pitch, ph = (float(p[k][i]) for k in ("r0", "pitch", "phase"))
    r = np.linalg.norm(pts, axis=1); th = np.arctan2(pts[:, 1], pts[:, 0])
    return np.round((r - r0) / pitch - ((th - ph) % (2 * math.pi)) / (2 * math.pi)).astype(int)


# ------------------------------------------------------------------ crushing as a flow
def random_velocity(n, rng, crush=(0.55, 1.1), wobble=0.12, size=SIZE, device=DEV):
    """v(x) = A x + t + smooth noise. A = log of a squash along a random axis (symmetric: no rotation, which a spiral
    could not tell apart from a different start angle). Returns (B, 2, H, W)."""
    g = grid(size, device)[0].permute(2, 0, 1)                                      # (2, H, W)
    out = []
    for _ in range(n):
        a, b = np.log(rng.uniform(*crush, 2)); ang = rng.uniform(0, math.pi)
        R = np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]])
        A = torch.tensor(R @ np.diag([a, b]) @ R.T, dtype=torch.float32, device=device)
        t = torch.tensor(rng.normal(0, 0.04, 2), dtype=torch.float32, device=device)
        lin = torch.einsum("ij,jhw->ihw", A, g) + t[:, None, None]
        noise = torch.tensor(rng.normal(0, 1, (1, 2, 6, 6)), dtype=torch.float32, device=device)
        noise = F.interpolate(noise, size=(size, size), mode="bicubic", align_corners=True)[0]
        out.append(lin + wobble * rng.uniform(0, 1) * noise)
    return torch.stack(out)


def flow(v, steps=7):
    """exp(v): the map x -> x(1) for dx/dt = v(x), by scaling and squaring (2**steps Euler steps, composed). (B, H, W, 2)."""
    idg = grid(v.shape[-1], v.device)
    disp = v.permute(0, 2, 3, 1) / 2 ** steps
    for _ in range(steps):
        disp = disp + F.grid_sample(disp.permute(0, 3, 1, 2), idg + disp, align_corners=True, padding_mode="border").permute(0, 2, 3, 1)
    return idg + disp


def warp(img, phi):
    """img sampled at phi(x): the image whose point x shows what img had at phi(x)."""
    return F.grid_sample(img, phi, align_corners=True, padding_mode="zeros")


def at_points(phi, pts):
    """Evaluate a map (1, H, W, 2) at points (n, 2) in normalised coordinates."""
    q = torch.as_tensor(pts, device=phi.device)[None, :, None, :]
    return F.grid_sample(phi.permute(0, 3, 1, 2), q, align_corners=True, padding_mode="border")[0, :, :, 0].T.cpu().numpy()


def jacobian_det(phi):
    """det of d phi / dx per pixel, in units where the identity has det 1. <= 0 means the map folds the sheet onto itself."""
    h = 2 / (phi.shape[1] - 1)
    dx = (phi[:, 1:-1, 2:] - phi[:, 1:-1, :-2]) / (2 * h); dy = (phi[:, 2:, 1:-1] - phi[:, :-2, 1:-1]) / (2 * h)
    return dx[..., 0] * dy[..., 1] - dx[..., 1] * dy[..., 0]


def batch(n, rng, **kw):
    """Synthetic training pairs: (deformed image, true velocity, ideal image)."""
    p = spiral_params(n, rng)
    ideal, _ = spiral_image(p)
    v = random_velocity(n, rng, **kw)
    deformed = warp(ideal, flow(-v))               # deformed(x) = ideal(exp(-v)(x)): a sheet point x appears at exp(v)(x)
    deformed = (deformed + 0.06 * torch.randn_like(deformed)).clamp(0, 1)
    return deformed, v, ideal


# ------------------------------------------------------------------ the network
def block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.GroupNorm(8, o), nn.SiLU(), nn.Conv2d(o, o, 3, padding=1), nn.GroupNorm(8, o), nn.SiLU())


class UNet(nn.Module):
    """Deformed image (1 channel) -> velocity field (2 channels), predicted on a coarse x coarse grid and upsampled bicubically."""
    def __init__(self, w=(16, 32, 64, 128, 128), coarse=16):
        super().__init__(); self.coarse = coarse
        self.down = nn.ModuleList([block(1 if i == 0 else w[i - 1], c) for i, c in enumerate(w)])
        self.up = nn.ModuleList([block(w[i + 1] + w[i], w[i]) for i in reversed(range(len(w) - 1))])
        self.head = nn.Conv2d(w[0], 2, 3, padding=1); nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)

    def forward(self, x):
        skips = []
        for i, b in enumerate(self.down):
            x = b(x if i == 0 else F.avg_pool2d(x, 2)); skips.append(x)
        for b, s in zip(self.up, reversed(skips[:-1])):
            x = b(torch.cat([F.interpolate(x, size=s.shape[-2:], mode="bilinear", align_corners=False), s], 1))
        u = F.adaptive_avg_pool2d(self.head(x), self.coarse)       # a crush is smooth: predict it on a coarse grid,
        return F.interpolate(u, size=x.shape[-2:], mode="bicubic", align_corners=True)   # then upsample smoothly


def train(steps=3000, bs=12, lr=2e-3, seed=0, every=250, log=print):
    """Train the U-Net on fresh synthetic pairs. Loss: velocity MSE + image MSE after uncrushing.
    Writes CKPT/latest.pt every `every` steps (weights, optimiser, step, RNG state) and resumes from it; CKPT/final.pt at the end."""
    CKPT.mkdir(parents=True, exist_ok=True)
    net = UNet().to(DEV); opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.05)
    rng, step, hist = np.random.default_rng(seed), 0, []
    if (CKPT / "final.pt").exists():
        s = torch.load(CKPT / "final.pt", map_location=DEV, weights_only=False); net.load_state_dict(s["net"])
        log(f"loaded the trained model ({s['step']} steps)"); return net.eval(), s["hist"]
    if (CKPT / "latest.pt").exists():
        s = torch.load(CKPT / "latest.pt", map_location=DEV, weights_only=False)
        net.load_state_dict(s["net"]); opt.load_state_dict(s["opt"]); sched.load_state_dict(s["sched"])
        step, hist = s["step"], s["hist"]; rng.bit_generator.state = s["rng"]
        log(f"RESUMED at step {step} of {steps} ({len(hist)} logged losses)")
    net.train()
    while step < steps:
        deformed, v, ideal = batch(bs, rng)
        u = net(deformed)
        loss_v = F.mse_loss(u, v); loss_i = F.mse_loss(warp(deformed, flow(u)), ideal)
        loss = loss_v + 4 * loss_i
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step(); step += 1
        hist.append((step, float(loss_v), float(loss_i)))
        if step % every == 0 or step == steps:
            torch.save(dict(net=net.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), step=step, hist=hist,
                            rng=rng.bit_generator.state), CKPT / "latest.pt")
            log(f"step {step:5d}  velocity MSE {np.mean([h[1] for h in hist[-every:]]):.5f}  image MSE {np.mean([h[2] for h in hist[-every:]]):.5f}")
    torch.save(dict(net=net.state_dict(), step=step, hist=hist), CKPT / "final.pt")
    return net.eval(), hist


# ------------------------------------------------------------------ unpaired: flow matching with OT couplings
def ot_pairs(a, b):
    """Exact optimal-transport matching between two equal-size point sets (squared distance)."""
    from scipy.optimize import linear_sum_assignment
    c = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
    i, j = linear_sum_assignment(c)
    return j[np.argsort(i)]


class VelocityMLP(nn.Module):
    def __init__(self, h=128):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(3, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, 2))

    def forward(self, x, t):
        return self.f(torch.cat([x, t], 1))


def ot_flow_matching(src, dst, steps=2000, bs=256, seed=0):
    """OT-CFM (Tong et al. 2023): learn a velocity field carrying the src cloud to the dst cloud with NO correspondences.
    Each step: minibatches from both, OT-match them, regress v(x_t, t) onto (x1 - x0) along the straight line."""
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    m = VelocityMLP(); opt = torch.optim.Adam(m.parameters(), 1e-3)
    for _ in range(steps):
        a = src[rng.integers(0, len(src), bs)]; b = dst[rng.integers(0, len(dst), bs)]
        b = b[ot_pairs(a, b)]
        x0, x1 = torch.tensor(a), torch.tensor(b); t = torch.rand(bs, 1)
        loss = ((m(x0 + t * (x1 - x0), t) - (x1 - x0)) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return m


@torch.no_grad()
def transport(m, x, n=60):
    x = torch.tensor(x)
    for k in range(n):
        x = x + m(x, torch.full((len(x), 1), k / n)) / n
    return x.numpy()


# ------------------------------------------------------------------ the real check
def roundness(pts):
    """How far one traced wrap is from a circle: fit a centre by least squares, then the radius as a linear function of
    the angle (a spiral wrap grows slowly); return the residual std / mean radius. 0 = a perfect spiral wrap."""
    A = np.c_[2 * pts, np.ones(len(pts))]; b = (pts ** 2).sum(1)
    cx, cy, _ = np.linalg.lstsq(A, b, rcond=None)[0]
    d = pts - [cx, cy]; r = np.linalg.norm(d, axis=1); th = np.unwrap(np.arctan2(d[:, 1], d[:, 0])[np.argsort(np.arctan2(d[:, 1], d[:, 0]))])
    r = r[np.argsort(np.arctan2(d[:, 1], d[:, 0]))]
    fit = np.polyval(np.polyfit(th, r, 1), th)
    return float(np.std(r - fit) / np.mean(r))
