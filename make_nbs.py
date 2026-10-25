"""Build the tutorial notebooks: .venv/bin/python make_nbs.py  (then execute with nbconvert, see README)."""
import nbformat as nbf

md = lambda s: nbf.v4.new_markdown_cell(s.strip())
code = lambda s: nbf.v4.new_code_cell(s.strip())
SETUP = code("""
import sys, time, numpy as np, matplotlib.pyplot as plt
sys.path.insert(0, '..')
import vlab
plt.rcParams.update({'figure.dpi': 110, 'image.cmap': 'gray'})
""")
COLAB = code("""
# On Google Colab: fetch the code and the two missing libraries. Locally this cell does nothing.
import sys, os
if 'google.colab' in sys.modules:
    if not os.path.exists('/content/vesuvius-labs'):
        !git clone -q https://github.com/alexchilton/vesuvius-labs /content/vesuvius-labs
        !pip -q install "zarr<3" s3fs tifffile
    os.chdir('/content/vesuvius-labs/notebooks')
""")
NB = {}

# ---------------------------------------------------------------- 00
NB["00_the_problem"] = [
md("""
# 00 · Reading a scroll nobody can open

**The problem.** In 79 AD Vesuvius buried a library at Herculaneum. Its papyrus scrolls were carbonised into lumps that
crumble if you try to unroll them. The Vesuvius Challenge reads them *virtually*: X-ray CT the scroll, find the papyrus
sheet inside the 3D scan, flatten it, and detect the ink.

**The catch.** The ink is carbon on carbonised papyrus. In X-ray it has almost the same density as the sheet, so you
cannot simply "see" it. It shows up as a faint texture that machine-learning models learn to spot.

**This notebook** draws the pipeline, then streams a real scroll (PHerc. 139) from the public S3 bucket and looks at it.
Nothing is downloaded in bulk; we read a few chunks.

| notebook | step of the pipeline |
|---|---|
| 00 | the problem, the scan |
| 01 | the volume: pyramids, chunks, air vs papyrus |
| 02 | flattening: render a sheet from its mesh (virtual unwrapping by hand) |
| 03 | ink detection: a small 2.5D CNN |
"""),
SETUP,
md("## 1. The pipeline"),
code("""
fig, ax = plt.subplots(figsize=(12, 2.2)); ax.axis('off')
steps = ['CT scan\\n(3D voxels,\\n9.4 / 2.4 µm)', 'surface\\nprediction\\n(where is papyrus?)', 'mesh tracing\\n(one sheet,\\nno layer jumps)',
         'flattening\\n(sample along mesh\\n+ normals)', 'ink detection\\n(2.5D model)', 'papyrologists\\nread it']
for i, s in enumerate(steps):
    ax.add_patch(plt.Rectangle((i * 2, 0), 1.6, 1, fc='#f3e6cc' if i in (3, 4) else '#e8eef6', ec='k'))
    ax.text(i * 2 + .8, .5, s, ha='center', va='center', fontsize=9)
    if i < len(steps) - 1: ax.annotate('', (i * 2 + 2, .5), (i * 2 + 1.6, .5), arrowprops=dict(arrowstyle='->'))
ax.set_xlim(-.2, 12); ax.set_ylim(-.2, 1.2); ax.set_title('the Vesuvius pipeline (shaded: what notebooks 02 and 03 do by hand)'); plt.show()
"""),
md("""
## 2. The data

The scans live in `s3://vesuvius-challenge-open-data/` (anonymous read, CC-BY-NC 4.0), one folder per scroll.
Each scan is an **OME-Zarr** pyramid: level 0 is full resolution, every further level halves each axis.
The array axes are `(z, y, x)`; `z` runs along the scroll's long axis.
"""),
code("""
for lvl in range(6):
    a = vlab.scroll_volume(lvl)
    print(f"level {lvl}: shape {a.shape}, {np.prod(a.shape) / 1e9:8.2f} GB as uint8, chunks {a.chunks}")
"""),
md("""
Level 0 is about **920 GB**. That is why everything here streams small pieces: a slice at level 4 is a handful of
128³ chunks.

## 3. A cross-section: the spiral
"""),
code("""
v4 = vlab.scroll_volume(4)
t = time.time(); sl = vlab.crop(v4, 'v4', v4.shape[0] // 2)
print(f'slice {sl.shape} in {time.time() - t:.1f}s (cached after the first run); value range {sl.min()}..{sl.max()}')
fig, ax = plt.subplots(1, 2, figsize=(11, 5))
ax[0].imshow(sl); ax[0].set_title('cross-section at the scroll middle (level 4, 150 µm voxels)')
v2 = vlab.scroll_volume(2); z2 = v2.shape[0] // 2
zoomed = vlab.crop(v2, 'v2mid', z2, slice(600, 1000), slice(600, 1000))
ax[1].imshow(zoomed); ax[1].set_title('zoom at level 2 (37 µm): the wraps of the sheet')
plt.show()
"""),
md("""
The bright spiral is papyrus; dark is air. In the zoom you can count wraps, and see where they touch or are crushed
together. Following **one** wrap through thousands of such slices, without jumping to its neighbour, is the
segmentation problem (notebook 02 uses a sheet someone already traced).

## 4. Along the scroll
"""),
code("""
side = vlab.crop(v4, 'v4side', slice(None), v4.shape[1] // 2)
plt.figure(figsize=(12, 3)); plt.imshow(side.T, aspect='auto'); plt.title('a slice along the scroll axis (z →), level 4'); plt.show()
"""),
md("""
## Check yourself

1. Why does the challenge use a pyramid instead of one big array?
2. Ink is carbon on carbon. Why does that make it harder than reading pencil on paper in an X-ray?
3. In the zoomed cross-section, where would a tracer most likely jump from one wrap to the next?
"""),
]

# ---------------------------------------------------------------- 01
NB["01_volumes"] = [
md("""
# 01 · The volume: chunks, air and papyrus

**Goal.** Read one small 3D block of the scroll, and learn the three things every later step depends on:
the pyramid and chunk layout, what intensities mean, and how a sheet looks in three orthogonal views.

**Data.** PHerc. 139, the 9.362 µm scan, pyramid level 1 (18.7 µm voxels), one 256³ block near the middle.
"""),
SETUP,
code("""
v1 = vlab.scroll_volume(1)
zc, yc, xc = v1.shape[0] // 2, 1500, 1500
t = time.time(); B = vlab.crop(v1, 'v1block', slice(zc, zc + 256), slice(yc, yc + 256), slice(xc, xc + 256))
print(f'block {B.shape} {B.dtype}, {B.nbytes / 1e6:.0f} MB, streamed in {time.time() - t:.1f}s')
print('chunks touched:', np.prod([int(np.ceil(256 / c)) + 1 for c in v1.chunks]), 'at most (128³ each)')
"""),
md("## 1. What the numbers mean"),
code("""
h = np.bincount(B.ravel(), minlength=256)
plt.figure(figsize=(9, 3)); plt.semilogy(h); plt.xlabel('voxel value (uint8)'); plt.ylabel('count (log)')
thr = 255 * 0.35
plt.axvline(thr, c='r', ls='--'); plt.title('intensity histogram: a big air peak, then papyrus'); plt.show()
print(f'fraction above {thr:.0f}: {(B > thr).mean():.3f}')
"""),
md("""
The volume is **masked** (outside the scroll is 0) and quantised to 8 bits. Air sits low; papyrus higher.
A threshold separates them roughly; it fails where sheets touch, which is exactly where the real models are needed.

## 2. Three orthogonal views
"""),
code("""
fig, ax = plt.subplots(1, 3, figsize=(13, 4.3))
for a, (img, t) in zip(ax, [(B[128], 'z (across the scroll)'), (B[:, 128], 'y'), (B[:, :, 128], 'x')]):
    a.imshow(img); a.set_title(f'slice through {t}')
plt.show()
fig, ax = plt.subplots(1, 3, figsize=(13, 4.3))
for a, (img, t) in zip(ax, [(B[128] > thr, 'z'), (B[:, 128] > thr, 'y'), (B[:, :, 128] > thr, 'x')]):
    a.imshow(img); a.set_title(f'threshold, {t}')
plt.show()
"""),
md("""
## 3. Sheets have a direction

Papyrus is a thin sheet, so locally the intensity changes fast *across* it and slowly *along* it.
The **structure tensor** (averaged outer product of the gradient) captures that: its largest eigenvector is the sheet
normal. This is the classical, model-free way to find sheets, and the normal is what notebook 02 samples along.
"""),
code("""
from scipy.ndimage import gaussian_filter, sobel
Bf = gaussian_filter(B.astype(np.float32), 1.0)
g = [sobel(Bf, axis=i) for i in range(3)]
J = np.empty(B.shape + (3, 3), np.float32)
for i in range(3):
    for j in range(3):
        J[..., i, j] = gaussian_filter(g[i] * g[j], 3.0)
w, v = np.linalg.eigh(J[128])                      # one z-slice, 256 x 256 tensors
coh = (w[..., 2] - w[..., 1]) / (w[..., 2] + w[..., 1] + 1e-6)   # 1 = clean sheet, 0 = isotropic
fig, ax = plt.subplots(1, 2, figsize=(11, 5))
ax[0].imshow(B[128]); ax[0].set_title('slice')
ax[1].imshow(coh * (B[128] > thr), cmap='magma'); ax[1].set_title('sheet coherence (bright = clearly a sheet)')
ys, xs = np.mgrid[8:256:16, 8:256:16]; n = v[ys, xs, :, 2]
ax[0].quiver(xs, ys, n[..., 2], -n[..., 1], color='r', scale=25); plt.show()
"""),
md("""
Red arrows: the estimated normal (in-plane part). They point across the wraps. Coherence drops where wraps touch
or the sheet is damaged.

## Check yourself

1. Why is a global threshold not enough to separate two touching wraps?
2. The structure tensor's normal has a sign ambiguity. Why does that matter when you sample "in front of" a sheet?
3. Which pyramid level would you use to find the scroll's outline, and which to see fibres?
"""),
]

# ---------------------------------------------------------------- 02
NB["02_flattening"] = [
md("""
# 02 · Virtual unwrapping by hand

**Goal.** Take a sheet someone traced through the scroll (a *mesh*) and render it flat ourselves: sample the 3D volume
along the mesh and along its normals. Then check our result against the official **surface volume** the challenge
publishes for the same segment.

**Data.**

| object | what it is | size |
|---|---|---|
| `mesh/*.tifxyz` | three float images `x, y, z`: grid point (i, j) is a 3D position in the scroll (level-0 voxels) | 1.5 MB |
| scroll volume, level 0 | the 9.362 µm scan, streamed as one small crop | ~35 MB here |
| official surface volume | 28 layers × 7,280 × 6,800, the challenge's own rendering of this sheet | we read one tile |

The mesh grid is stored at `scale = 0.05`: one grid step is 20 canvas pixels.
"""),
SETUP,
code("""
x, y, z, meta = vlab.segment_mesh()
print('grid', x.shape, '| scale', meta['scale'], '| valid points', f'{(x > 0).mean():.1%}', '| bbox (x, y, z)', np.round(meta['bbox']).tolist())
print('first valid row of points:'); r = np.argwhere(x > 0)[0]
print(' (x, y, z) =', [round(float(c[r[0], r[1] + k]), 1) for k in range(3) for c in (x, y, z)][:9])
"""),
md("## 1. The sheet in 3D"),
code("""
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
m = x > 0; step = 6
fig = plt.figure(figsize=(8, 6)); ax = fig.add_subplot(projection='3d')
ax.scatter(x[::step, ::step][m[::step, ::step]], y[::step, ::step][m[::step, ::step]], z[::step, ::step][m[::step, ::step]],
           s=1, c=z[::step, ::step][m[::step, ::step]], cmap='viridis')
ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z (scroll axis)'); ax.set_title('the traced sheet: part of one wrap'); plt.show()
"""),
md("""
## 2. Render a patch

For a 24 × 24 grid patch (480 × 480 pixels):

1. **Interpolate** the grid to every canvas pixel. Pixel (u, v) sits at grid coordinate (u/20, v/20).
2. **Normals**: the cross product of the two in-plane derivatives.
3. **Sample** the volume at `point + k · normal` for k = −14 … 13 (28 layers), trilinear.
"""),
code("""
from scipy.ndimage import map_coordinates
i0, j0, g, up = 30, 113, 24, 20
GI, GJ = np.meshgrid(i0 + np.arange(g * up) / up, j0 + np.arange(g * up) / up, indexing='ij')
X, Y, Z = [map_coordinates(c.astype(np.float64), [GI, GJ], order=1) for c in (x, y, z)]
P = np.stack([X, Y, Z], -1)
n = np.cross(np.gradient(P, axis=0), np.gradient(P, axis=1)); n /= np.linalg.norm(n, axis=-1, keepdims=True)
lo = np.floor(np.array([Z.min(), Y.min(), X.min()]) - 20).astype(int); hi = np.ceil(np.array([Z.max(), Y.max(), X.max()]) + 20).astype(int)
t = time.time(); V = vlab.crop(vlab.scroll_volume(0), 'nb02', *[slice(a, b) for a, b in zip(lo, hi)])
print(f'volume crop {V.shape} ({V.nbytes / 1e6:.0f} MB) in {time.time() - t:.1f}s')
Pz = np.stack([Z, Y, X], -1) - lo; N = n[..., [2, 1, 0]]
mine = np.stack([map_coordinates(V, (Pz + k * N).reshape(-1, 3).T, order=1).reshape(X.shape) for k in np.arange(28) - 14])
print('our surface volume', mine.shape)
"""),
md("## 3. Check it against the official rendering"),
code("""
S = vlab.segment_surface(0)
ref = vlab.crop(S, 'nb02ref', slice(None), slice(i0 * up, (i0 + g) * up), slice(j0 * up, (j0 + g) * up)).astype(float)
c = [np.corrcoef(mine[k][40:-40, 40:-40].ravel(), ref[k][40:-40, 40:-40].ravel())[0, 1] for k in range(28)]
print(f'correlation per layer: mean {np.mean(c):.3f}, min {min(c):.3f}, max {max(c):.3f}')
fig, ax = plt.subplots(2, 3, figsize=(12, 8))
for col, k in enumerate((8, 14, 20)):
    ax[0, col].imshow(mine[k]); ax[0, col].set_title(f'ours, layer {k}')
    ax[1, col].imshow(ref[k]); ax[1, col].set_title(f'official, layer {k}')
plt.show()
"""),
md("""
A correlation around 0.93 means we reproduced the challenge's renderer from its mesh and the raw scan.
The rest is interpolation detail (their resampling and normal smoothing differ slightly from ours).

**A trap we hit building this.** Upsampling the grid with `scipy.ndimage.zoom` instead of interpolating at
`pixel / 20` gives correlation **0.37**: `zoom` stretches the grid so its corners align, which shifts every interior
point by up to half a cell. The geometry must be resampled at exactly the pixel positions the format defines.

## 4. Why 28 layers?
"""),
code("""
prof = mine.reshape(28, -1).mean(1)
plt.figure(figsize=(7, 3)); plt.plot(np.arange(28) - 14, prof, 'o-'); plt.axvline(0, c='r', ls='--')
plt.xlabel('offset along the normal (voxels)'); plt.ylabel('mean intensity'); plt.title('the sheet is the bright band around offset 0'); plt.show()
"""),
md("""
The papyrus is a few voxels thick and the mesh is not exactly on it. Rendering a slab of layers lets the ink model
look *into* the sheet and forgive a mesh that is off by a few voxels.

## Check yourself

1. What happens to the rendering if a normal flips sign halfway across the patch?
2. Why does the slab have to be wider when the mesh was traced at a coarser resolution?
3. How would a sheet switch (the mesh jumping to the neighbouring wrap) look in the rendered layers?
"""),
]

# ---------------------------------------------------------------- 03
NB["03_ink_detection"] = [
md("""
# 03 · Ink detection: a small 2.5D CNN

**Goal.** Train a model that looks at a stack of surface layers and predicts ink, and measure how far a tiny model gets.

**Data.**

| object | shape | role |
|---|---|---|
| surface volume of segment w025, level 1 | 28 × 3,640 × 3,400 (18.7 µm pixels) | input |
| official ink prediction (2.4 µm model), resized to the same canvas | 3,640 × 3,400, 0..1 | **pseudo-label** |

**Honest framing.** These labels are another model's output, not human ink labels. Training on them is
*distillation*: our small model learns to imitate a big one from a coarser scan. What it can show is whether the
9 µm data carries the signal the 2.4 µm model found, on regions it never trained on.

**The split is spatial.** Neighbouring patches share pixels, so a random patch split leaks. We train on the left
60% of the sheet and validate on the right 40%.
"""),
SETUP,
code("""
import torch, torch.nn as nn, torch.nn.functional as F, tifffile
from PIL import Image
S1 = vlab.segment_surface(1)
t = time.time(); X = vlab.crop(S1, 'seg_l1', slice(None), slice(None), slice(None)); print(f'surface volume {X.shape}, {X.nbytes / 1e6:.0f} MB, {time.time() - t:.0f}s')
ink = tifffile.imread(vlab.fetch_ink())
Yl = np.asarray(Image.fromarray(ink).resize((X.shape[2], X.shape[1]), Image.BILINEAR), np.float32) / 255.
valid = X.max(0) > 0
print('label', Yl.shape, f'mean {Yl[valid].mean():.3f}', '| valid pixels', f'{valid.mean():.1%}')
fig, ax = plt.subplots(1, 2, figsize=(12, 6)); ax[0].imshow(X[14]); ax[0].set_title('layer 14 (the surface)')
ax[1].imshow(Yl, cmap='magma'); ax[1].set_title('pseudo-label: official 2.4 µm ink prediction'); plt.show()
"""),
md("""
Layer 14 alone shows no letters: this is the point of the challenge. The label shows a line of Greek letters near the
top and a lot of speckle lower down. The speckle is uncertainty in the big model, and our model will learn it too.

## 1. Patches and the split
"""),
code("""
dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
L0, L1 = 6, 22                                     # 16 layers around the surface
Xn = (X[L0:L1].astype(np.float32) / 255. - 0.5) / 0.25
P, H, W = 64, *X.shape[1:]
cut = int(W * 0.6)
def sample(n, train, rng):
    xs, ys = [], []
    while len(xs) < n:
        i = rng.integers(0, H - P); j = rng.integers(0, cut - P) if train else rng.integers(cut, W - P)
        if valid[i:i + P, j:j + P].mean() < 0.9: continue
        xs.append(Xn[:, i:i + P, j:j + P]); ys.append(Yl[i:i + P, j:j + P])
    return torch.tensor(np.stack(xs)), torch.tensor(np.stack(ys))[:, None]
rng = np.random.default_rng(0); xv, yv = sample(512, False, rng)
print('device', dev, '| input layers', L1 - L0, '| patch', P, '| train columns <', cut, '| val patches', len(xv))
"""),
md("""
## 2. The model

A small 2.5D net: 3D convolutions read the layer stack, a max over depth collapses it to 2D
(the challenge's own `z_projection_mode: max`), and a few 2D convolutions produce the ink map.
"""),
code("""
class Ink25D(nn.Module):
    def __init__(s, c=16):
        super().__init__()
        s.enc = nn.Sequential(nn.Conv3d(1, c, 3, padding=1), nn.GELU(), nn.Conv3d(c, c, 3, padding=1), nn.GELU(),
                              nn.Conv3d(c, 2 * c, (3, 3, 3), padding=1), nn.GELU())
        s.dec = nn.Sequential(nn.Conv2d(2 * c, 2 * c, 3, padding=1), nn.GELU(), nn.Conv2d(2 * c, 2 * c, 3, padding=2, dilation=2),
                              nn.GELU(), nn.Conv2d(2 * c, 1, 1))
    def forward(s, x):                             # x: (B, layers, H, W)
        return s.dec(s.enc(x[:, None]).amax(2))    # max over depth -> (B, 2c, H, W) -> logits
net = Ink25D().to(dev); print(sum(p.numel() for p in net.parameters()), 'parameters')
"""),
code("""
opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-4)
def val_corr():
    net.eval(); out = []
    with torch.no_grad():
        for k in range(0, len(xv), 64): out.append(torch.sigmoid(net(xv[k:k + 64].to(dev))).cpu())
    p = torch.cat(out).ravel().numpy(); return np.corrcoef(p, yv.ravel().numpy())[0, 1]
t = time.time(); hist = []
for step in range(1, 1501):
    net.train(); xb, yb = sample(16, True, rng)
    loss = F.binary_cross_entropy_with_logits(net(xb.to(dev)), yb.to(dev))
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 250 == 0:
        hist.append((step, loss.item(), val_corr())); print(f'step {step} loss {loss.item():.4f} val corr {hist[-1][2]:.3f} ({time.time() - t:.0f}s)')
"""),
md("""
## 3. Baseline: is it just brightness?

If ink were simply denser papyrus, one layer's brightness would predict the label. Compare.
"""),
code("""
base = np.corrcoef(xv[:, 14 - L0 - 2:14 - L0 + 3].mean(1).ravel().numpy(), yv.ravel().numpy())[0, 1]
print(f'held-out correlation with the pseudo-label: brightness of the surface layers {base:.3f}  vs  2.5D CNN {hist[-1][2]:.3f}')
"""),
md("## 4. Predict the whole sheet"),
code("""
net.eval(); pred = np.zeros((H, W), np.float32); cnt = np.zeros((H, W), np.float32); T = 256
with torch.no_grad():
    for i in range(0, H - T + 1, T // 2):
        for j in range(0, W - T + 1, T // 2):
            if valid[i:i + T, j:j + T].mean() < 0.2: continue
            p = torch.sigmoid(net(torch.tensor(Xn[None, :, i:i + T, j:j + T]).to(dev)))[0, 0].cpu().numpy()
            pred[i:i + T, j:j + T] += p; cnt[i:i + T, j:j + T] += 1
pred = np.where(cnt > 0, pred / np.maximum(cnt, 1), 0) * valid
fig, ax = plt.subplots(1, 2, figsize=(13, 6.5))
ax[0].imshow(pred, cmap='magma'); ax[0].axvline(cut, c='c', ls='--'); ax[0].set_title('our 2.5D CNN (left of the line: trained on)')
ax[1].imshow(Yl * valid, cmap='magma'); ax[1].axvline(cut, c='c', ls='--'); ax[1].set_title('pseudo-label'); plt.show()
top = slice(200, 900)
fig, ax = plt.subplots(2, 1, figsize=(13, 6)); ax[0].imshow(pred[top], cmap='magma'); ax[0].set_title('zoom: the line of letters, ours')
ax[1].imshow((Yl * valid)[top], cmap='magma'); ax[1].set_title('zoom: pseudo-label'); plt.show()
"""),
md("""
## What to take away

- The letters are invisible in any single layer. Brightness alone correlates 0.11 with the label on the held-out strip;
  the CNN reaches about 0.31 after 1,500 steps and is still rising. The signal is texture across depth, not brightness.
- 0.31 is not readable letters: the prediction picks up ink-like texture and the big model's speckle, but the strokes
  are blurred. Closing that gap with few labels is what notebook 04 tests.
- Everything right of the dashed line was never trained on. That is the only part of the picture that counts.
- This model imitates another model. Its letters are only as real as the big model's; the challenge's prizes demand
  held-out validation and papyrologist review for exactly this reason.

## Check yourself

1. Why would a random train/validation patch split overstate the score here?
2. What does `amax` over depth assume about where the ink sits in the stack?
3. The pseudo-label is speckled in the lower half. How would you stop the model learning the speckle?
4. Notebook 04 trains on one fragment and tests on another. What do you expect to happen to the correlation, and why?
"""),
]

# ---------------------------------------------------------------- 04
NB["04_self_supervised"] = [
md("""
# 04 · Few labels: does self-supervised pretraining help?

**The question.** Labels are scarce in this challenge; unlabelled scan is not. If we first teach the encoder papyrus
texture from **unlabelled** layers, does it then learn ink from fewer labels than an encoder trained from scratch?

**The pretext task: masked image modelling** (SimMIM-style). Hide 60% of the 8 × 8 pixel blocks of a 16-layer patch
(all layers at once) and train the network to reconstruct the hidden voxels. To do that it must model fibre texture and
the depth profile of the sheet, and it never sees a label.

**The experiment.**

| | |
|---|---|
| data | segment w025, 16 surface layers, level 1 (from notebook 03) |
| label | the official ink prediction (pseudo-label), as in notebook 03 |
| labelled pool | 128 × 128 tiles from the left 60% of the sheet; we reveal 2%, 10%, 50% or 100% of them |
| pretraining data | every tile of the left 60%, **labels unused** |
| validation | the same 512 patches from the right 40%, never seen by either stage |
| arms | scratch vs pretrained encoder, the same head, the same steps, the same seed |

**What would make SSL win:** its held-out correlation is above scratch at small label fractions, and the gap shrinks
as the fraction grows. If the curves overlap, the pretext did not teach anything ink-relevant at this scale.
"""),
SETUP,
code("""
import torch, torch.nn as nn, torch.nn.functional as F
import inklab as K
Xn, Y, valid = K.load()
cut = int(Xn.shape[2] * 0.6)
pool = K.tiles(valid, cut); xv, yv = K.val_set(Xn, Y, valid, cut)
print('device', K.DEV, '| layers', Xn.shape[0], '| labelled pool', len(pool), 'tiles of 128 x 128 | val patches', len(xv))
"""),
md("## 1. Pretext: reconstruct masked blocks"),
code("""
def mask_blocks(x, ratio, rng, b=8):
    B, L, H, W = x.shape
    m = torch.tensor(rng.random((B, 1, H // b, W // b)) < ratio, dtype=torch.float32)
    m = m.repeat_interleave(b, 2).repeat_interleave(b, 3)        # (B, 1, H, W): 1 = hidden
    return x * (1 - m), m
rng = np.random.default_rng(0); xb, _ = K.sample(Xn, Y, pool, 4, rng); xm, m = mask_blocks(xb, 0.6, rng)
fig, ax = plt.subplots(1, 3, figsize=(10, 3.4))
ax[0].imshow(xb[0, 8]); ax[0].set_title('a patch, layer 8'); ax[1].imshow(m[0, 0], cmap='gray_r'); ax[1].set_title('60% of blocks hidden')
ax[2].imshow(xm[0, 8]); ax[2].set_title('what the encoder sees'); plt.show()
"""),
code("""
torch.manual_seed(0); rng = np.random.default_rng(1)
enc_ssl = K.Encoder().to(K.DEV); rec = K.head(32, Xn.shape[0]).to(K.DEV)
opt = torch.optim.AdamW(list(enc_ssl.parameters()) + list(rec.parameters()), 2e-3, weight_decay=1e-4)
t = time.time(); curve = []
for step in range(1, 1501):
    xb, _ = K.sample(Xn, Y, pool, 16, rng); xm, m = mask_blocks(xb, 0.6, rng)
    out = rec(enc_ssl(xm.to(K.DEV)))
    loss = (((out - xb.to(K.DEV)) ** 2) * m.to(K.DEV)).sum() / (m.sum() * Xn.shape[0]).to(K.DEV)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 250 == 0: curve.append((step, loss.item())); print(f'pretrain step {step} masked MSE {loss.item():.4f} ({time.time() - t:.0f}s)')
SSL_STATE = {k: v.detach().clone() for k, v in enc_ssl.state_dict().items()}
"""),
code("""
with torch.no_grad():
    xb, _ = K.sample(Xn, Y, pool, 1, np.random.default_rng(5)); xm, m = mask_blocks(xb, 0.6, np.random.default_rng(6))
    out = rec(enc_ssl(xm.to(K.DEV))).cpu()
fig, ax = plt.subplots(1, 3, figsize=(10, 3.4))
for a, im, t in zip(ax, [xm[0, 8], out[0, 8], xb[0, 8]], ['masked input', 'reconstruction', 'truth']): a.imshow(im, vmin=-2, vmax=2); a.set_title(t)
plt.show()
"""),
md("""
If the reconstruction fills the hidden blocks with plausible fibre texture, the encoder has learned something about
papyrus without a single label. Whether that something is *ink-relevant* is what the next section measures.

## 2. Fine-tune with few labels: scratch vs pretrained
"""),
code("""
def encoder(pretrained):
    e = K.Encoder()
    if pretrained: e.load_state_dict(SSL_STATE)
    return e
fracs, STEPS, rows = [0.02, 0.10, 0.50, 1.00], 600, []
for f in fracs:
    sub = [pool[i] for i in np.random.default_rng(42).permutation(len(pool))[:max(1, int(round(f * len(pool))))]]
    for arm in ('scratch', 'pretrained'):
        t = time.time(); c, _ = K.finetune(encoder(arm == 'pretrained'), Xn, Y, sub, xv, yv, STEPS, seed=0)
        rows.append((f, len(sub), arm, c)); print(f'labels {f:4.0%} ({len(sub):3d} tiles)  {arm:10s}  held-out corr {c:.3f}  ({time.time() - t:.0f}s)')
"""),
code("""
import pandas as pd
R = pd.DataFrame(rows, columns=['fraction', 'tiles', 'arm', 'corr']).pivot(index=['fraction', 'tiles'], columns='arm', values='corr')
R['pretrained - scratch'] = R['pretrained'] - R['scratch']; print(R.round(3).to_string())
plt.figure(figsize=(6.5, 4))
for arm, mk in (('scratch', 'o-'), ('pretrained', 's-')): plt.semilogx([f for f, _ in R.index], R[arm], mk, label=arm)
plt.xlabel('fraction of labelled tiles'); plt.ylabel('held-out correlation'); plt.legend(); plt.title('does pretraining buy label efficiency?'); plt.show()
"""),
md("""
## Reading the result

- **One seed per cell.** Two runs differing only in their seed can move by a few hundredths at 600 steps, so a gap
  smaller than that is not evidence. Rerun with seeds 1 and 2 before believing a small difference.
- The label is a pseudo-label (notebook 03). "Better" means "closer to the 2.4 µm model", not "closer to the true ink".
- Pretraining used only the left 60% too. Pretraining on the validation strip's unlabelled pixels is allowed in
  practice (you always have the unlabelled scroll you want to read), and is a fair follow-up: it is test-time
  self-supervision, aimed at the cross-scroll problem.

## Try next

1. Mask ratio 0.4 vs 0.8; block size 4 vs 16.
2. Mask whole *layers* instead of blocks: the model must infer depth structure.
3. SimCLR instead of masking: two augmented views of one patch as positives.
4. Pretrain on a different segment, fine-tune on this one: does the texture knowledge transfer?

## Check yourself

1. Why is the validation strip excluded from *pretraining* in this experiment?
2. Why does the gap between the arms have to shrink as labels grow, if SSL is working?
3. What would it mean if the pretrained arm were **worse** at 100% labels?
"""),
]

import sys
for name, cells in NB.items():
    if len(sys.argv) > 1 and name not in sys.argv[1:]: continue
    first = next(k for k, c in enumerate(cells) if c.cell_type == 'code')
    nb = nbf.v4.new_notebook(); nb["cells"] = cells[:first] + [COLAB] + cells[first:]
    nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nbf.write(nb, f"notebooks/{name}.ipynb"); print("wrote", name, len(cells), "cells")
