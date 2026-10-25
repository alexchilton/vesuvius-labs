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

**The problem.** In 79 AD Vesuvius buried a villa at Herculaneum under some twenty metres of volcanic material. The
villa held a library. Its papyrus scrolls were carbonised: they survived as black lumps that crumble if you try to
unroll them. Roughly a thousand are still unread. The Vesuvius Challenge reads them **without opening them**: scan the
scroll with X-rays, find the papyrus sheet inside the 3D scan, flatten it, and detect the ink.

**Why this is hard, in one table.**

| step | what goes wrong |
|---|---|
| scanning | ink and papyrus are both carbon, so they absorb X-rays almost equally |
| finding the sheet | wraps are crushed together, torn, and stuck to each other |
| following one sheet | a tracer that jumps to the neighbouring wrap reads text from the wrong page |
| flattening | the sheet is stretched and folded, so a flat image distorts the letters |
| ink detection | the ink shows only as a faint surface texture, and a model trained on one scroll often fails on the next |

**This notebook** covers the history, how the scans are made, the full pipeline and the data, then streams a real
scroll (PHerc. 139) from the public S3 bucket and measures it. Nothing is downloaded in bulk.

| notebook | step of the pipeline | open problem it touches (see section 6) |
|---|---|---|
| 00 | the problem, the scan, the data | 10 data infrastructure |
| 01 | the volume: pyramids, chunks, air vs papyrus, sheet normals | 2 surface prediction |
| 02 | flattening: render a sheet from its mesh, measure its distortion | 3 mesh tracing |
| 03 | ink detection: a small 2.5D CNN on pseudo-labels | 7 ink generalisation |
| 04 | few labels: does self-supervised pretraining help? | 9 self-supervised representations |
| 05 | uncrushing a cross-section with flows | 6 spiral fit and global priors |
"""),
SETUP,
md("""
## 1. How we got here

| when | what happened |
|---|---|
| 79 AD | Vesuvius erupts; the Villa of the Papyri and its library are buried |
| 18th century | the villa is excavated and the scrolls are found; attempts to unroll them by hand and by machine destroy many and expose only scraps |
| 2015 | Brent Seales' team (University of Kentucky) virtually unwraps the En-Gedi scroll and reads Leviticus. Its ink contained metal, so it was visible in X-ray; Herculaneum ink is not |
| March 2023 | the Vesuvius Challenge launches with public synchrotron scans and a prize fund |
| August 2023 | Casey Handmer spots the **crackle pattern**: a faint cracked texture where ink sits on the papyrus. First convincing evidence of ink inside a closed scroll ($10,000) |
| October 2023 | Luke Farritor trains a model on crackle and reads the first word from inside a scroll: ΠΟΡΦΥΡΑϹ, "purple" |
| 5 February 2024 | 2023 Grand Prize ($700,000) to Youssef Nader, Luke Farritor and Julian Schilliger: 15 columns, more than 2,000 characters, about 5% of PHerc. Paris. 4. The text is about pleasure, likely by the Epicurean Philodemus |
| December 2024 | first automated segmentation prize ($60,000): the 2023 result reproduced with far less manual tracing |
| May 2025 | first title of a sealed scroll read ($60,000, Marcel Roth and Micha Nowak) |
| March 2026 | Kaggle Surface Detection competition ($200,000): topologically correct papyrus detection |
| October 2026 | First Letters prize for PHerc. 343 ($50,000, Erwin Nieuwlaar): "πᾶν τὸ γινόμε[νον", "everything that happens" |
| 2027 | the Grand Prize ($1,000,000) asks for a **fully** unrolled scroll, with at most 8 hours of human annotation |

Total awarded since 2023: $1,958,500 (scrollprize.org/winners, October 2026). The shift over these years is the
lesson: 2023 was won with careful *human* tracing plus ML ink detection on 5% of one scroll; the open question now is
**automation at scale**.

## 2. How a scroll is scanned

```
   X-ray source ═══════► rotating scroll ═══════► detector
   (synchrotron beam)    (hundreds to thousands    (one 2D projection
                          of angles)                per angle)
                                     │
                         tomographic reconstruction
                                     ▼
                  a 3D grid of voxels: z = along the scroll axis
```

The scrolls were scanned at particle accelerators (Diamond Light Source in the UK, ESRF in France), not hospital CT.
A synchrotron beam is bright and coherent enough for **phase-contrast** imaging: besides absorption it records how the
beam is bent at edges, which makes thin structures such as the ink layer stand out. The price of resolution is data
size, which grows with the cube of the voxel count:
"""),
code("""
level0_gb, base_um = 20974 * 6621 * 6621 / 1e9, 9.362            # PHerc. 139 at 9.362 µm, measured in section 4
for um in (9.362, 4.0, 2.4, 1.1):
    print(f'{um:6.3f} µm voxels: {level0_gb * (base_um / um) ** 3 / 1e3:9.1f} TB for the whole scroll as uint8')
"""),
md("""
A 2.4 µm scan of one scroll is tens of terabytes. This is why the challenge stores scans as **pyramids of chunks**
(section 4, notebook 01) and why most work happens on small crops.

## 3. The pipeline
"""),
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
| stage | input → output | how it is done today | why it is hard |
|---|---|---|---|
| surface prediction | volume → "is this voxel papyrus surface?" | 3D segmentation networks (nnU-Net ensembles) | touching wraps merge; holes; the label itself is approximate |
| mesh tracing | surface prediction → one continuous sheet as a mesh | semi-automatic tracers in Volume Cartographer (VC3D), patch growing, human correction | one wrong step jumps to the next wrap; errors compound over thousands of slices |
| flattening | mesh → a flat parametrisation | minimise stretch (like flattening an orange peel) | a crumpled sheet cannot be flattened without distortion |
| rendering | mesh + volume → **surface volume**: ~28–65 layers sampled along the normals | trilinear sampling (we do it by hand in 02) | the mesh is only roughly on the sheet, so a slab of layers is sampled |
| ink detection | surface volume → 2D ink map | 2.5D or 3D CNNs trained on few, noisy labels | ink is a faint texture; models trained on one scroll fail on another |

## 4. The data

The scans live in `s3://vesuvius-challenge-open-data/` (anonymous read, CC-BY-NC 4.0), one folder per scroll or
fragment. Each scan is an **OME-Zarr** pyramid: level 0 is full resolution, every further level halves each axis.
The array axes are `(z, y, x)`; `z` runs along the scroll's long axis.
"""),
code("""
top = sorted(p.split('/')[-1] for p in vlab.FS.ls(vlab.BUCKET))
print(len(top), 'top-level folders, e.g.', top[:8])
s = vlab.SCROLL
print(f'{s}:', [p.split('/')[-1] for p in vlab.FS.ls(f'{vlab.BUCKET}/{s}')])
segs = vlab.FS.ls(f'{vlab.BUCKET}/{s}/segments')
print(f'{s} has {len(segs)} traced segments, e.g.', [p.split('/')[-1] for p in segs[:3]])
for lvl in range(6):
    a = vlab.scroll_volume(lvl)
    print(f"level {lvl}: shape {a.shape}, {np.prod(a.shape) / 1e9:8.2f} GB as uint8, chunks {a.chunks}")
"""),
md("""
Level 0 is about **920 GB**. A slice at level 4 is a handful of 128³ chunks.

| word | meaning |
|---|---|
| scroll | a closed papyrus roll (PHerc. = Herculaneum papyrus, followed by its inventory number) |
| fragment | a piece that was physically opened in the past; its surface can be photographed, including in infrared, which gives real ink labels |
| volume | one 3D scan of a scroll at one resolution and energy |
| segment | one traced piece of sheet: a mesh, plus the surface volume rendered from it |
| wrap | one turn of the spiral |
| sheet switch | a tracing error where the mesh jumps from one wrap to its neighbour |
| surface volume | the flattened sheet as a stack of layers parallel to the surface |
| crackle | the cracked texture of dried ink on papyrus, the main visual cue models learn |
| recto / verso | the inner (written) and outer face of the sheet |

## 5. A cross-section: the spiral
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

**How many wraps?** Walk outwards from the centre along a line and count the bright crossings. Each crossing is one
wrap (if the wraps are separated; crushed regions merge two into one, so the count is a lower bound).
"""),
code("""
from scipy.signal import find_peaks
v2s = vlab.crop(v2, 'v2full', z2).astype(np.float32)
mask = v2s > 0; cy, cx = [int(np.mean(i)) for i in np.nonzero(mask)]
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5)); ax[0].imshow(v2s)
counts = []
for ang in np.deg2rad([0, 60, 120, 180, 240, 300]):
    r = np.arange(0, 1200); yy, xx = (cy + r * np.sin(ang)).astype(int), (cx + r * np.cos(ang)).astype(int)
    ok = (yy >= 0) & (yy < v2s.shape[0]) & (xx >= 0) & (xx < v2s.shape[1]); yy, xx = yy[ok], xx[ok]
    prof = v2s[yy, xx]; inside = prof > 0
    pk, _ = find_peaks(np.convolve(prof, np.ones(3) / 3, 'same'), prominence=25, distance=4)
    pk = pk[inside[pk]]; counts.append(len(pk))
    ax[0].plot(xx, yy, lw=.8); ax[0].plot(xx[pk], yy[pk], 'r.', ms=2)
    if ang == 0: ax[1].plot(prof, lw=.7); ax[1].plot(pk, prof[pk], 'r.'); ax[1].set_xlabel('distance from the centre (level-2 voxels)')
ax[0].set_title('six rays from the centre; red = a wrap crossing'); ax[1].set_title('intensity along the first ray')
plt.show()
print('wrap crossings per ray:', counts, f'-> roughly {int(np.median(counts))} wraps visible at 37 µm (a lower bound)')
"""),
md("""
## 6. Along the scroll
"""),
code("""
side = vlab.crop(v4, 'v4side', slice(None), v4.shape[1] // 2)
plt.figure(figsize=(12, 3)); plt.imshow(side.T, aspect='auto'); plt.title('a slice along the scroll axis (z →), level 4'); plt.show()
"""),
md("""
The scroll is not a neat cylinder: its cross-section changes along z, and the ends are damaged. Every stage above has
to cope with that.

## 7. What is still open (the 2026 open problems)

The challenge publishes its list of open problems (scrollprize.org/2026_open_problems). In short:

| # | problem | what a solution looks like |
|---|---|---|
| 1 | scan quality in compressed regions | better reconstruction (phase retrieval) or scan recipes where wraps are crushed |
| 2 | surface prediction and topology | papyrus masks without holes, merges or switches; topology-aware losses |
| 3 | mesh tracing and geometry | tracers that follow one sheet automatically over a whole scroll |
| 4 | label quality | snapping approximate human meshes to the real sheet; active learning |
| 5 | fibre tracing | papyrus fibres run in two directions and give the sheet's U and V axes |
| 6 | spiral fit and global priors | fit the whole scroll as one (deformed) spiral; winding numbers to detect switches |
| 7 | ink detection generalisation | models that work on scrolls they were not trained on |
| 8 | direct 3D ink segmentation | find ink in the 2.4 µm volume without flattening first |
| 9 | self-supervised 3D representations | pretrained 3D encoders (the challenge's "dinovol") for every downstream task |
| 10 | data infrastructure | streaming and processing terabyte volumes |

**Where a newcomer fits.** Besides the big prizes there are monthly progress prizes for released tools, models,
documentation and bug fixes. Code: github.com/ScrollPrize/villa. Models: huggingface.co/scrollprize. Community: the
Discord and scrollprize.substack.com.

## Check yourself

1. Why does the challenge use a pyramid instead of one big array?
2. Ink is carbon on carbon. Why does that make it harder than reading pencil on paper in an X-ray?
3. The En-Gedi scroll was read in 2015. What made it easier than a Herculaneum scroll?
4. In the zoomed cross-section, where would a tracer most likely jump from one wrap to the next?
5. Why is the wrap count from the rays a lower bound and not the true count?
6. A 2.4 µm scan shows more detail. Name two costs of using it instead of the 9.4 µm scan.
"""),
]

# ---------------------------------------------------------------- 01
NB["01_volumes"] = [
md("""
# 01 · The volume: chunks, air and papyrus

**The problem.** Every later step reads the scan. Before tracing a sheet or detecting ink, you need to know how the
scan is stored (so that reading it is cheap), what its numbers mean (so that "bright" is not mistaken for "papyrus"),
and how a sheet looks locally (so that you can find its direction without any model).

**Data.**

| object | what it is | size here |
|---|---|---|
| PHerc. 139, the 9.362 µm scan | OME-Zarr pyramid, 6 levels, uint8, 128³ chunks | level 0 is about 920 GB |
| level 1 (18.7 µm voxels) | the same scan with every axis halved | one 256³ block, 17 MB |

**What you will measure:** the cost of a badly aligned read, the air/papyrus histogram, three orthogonal views, and the
sheet normal and its reliability (coherence) from the structure tensor.
"""),
SETUP,
md("""
## 1. How the scan is stored

```
 level 0   20,974 × 6,621 × 6,621  ─┐
 level 1   10,487 × 3,311 × 3,311   │  every level halves each axis:
 level 2    5,244 × 1,656 × 1,656   │  1/8 of the voxels of the level below
   ...                              │
 level 5      656 ×   207 ×   207  ─┘

 each level is cut into 128 × 128 × 128 chunks; a chunk is one file in the bucket,
 so a read costs one request per chunk it touches, whether you need 1 voxel of it or all 2 million
```

That last line is the practical rule: **align your reads to the chunk grid.** A 256³ block that starts on a chunk
boundary touches 2³ = 8 chunks; shift it by half a chunk and it touches 3³ = 27, more than three times the data.
"""),
code("""
v1 = vlab.scroll_volume(1)
zc, yc, xc = (v1.shape[0] // 2) // 128 * 128, 1536, 1536            # chunk-aligned start
for name, off in (('aligned to the chunk grid', 0), ('shifted by half a chunk', 64)):
    sl = tuple(slice(c + off, c + off + 256) for c in (zc, yc, xc))
    n = np.prod([len(range(s.start // 128, (s.stop - 1) // 128 + 1)) for s in sl])
    t = time.time(); _ = v1[sl]; dt = time.time() - t
    print(f'256³ block {name:27s}: {n:2d} chunks, {n * 128 ** 3 / 1e6:5.0f} MB fetched for 17 MB used, {dt:.1f}s (uncached)')
t = time.time(); B = vlab.crop(v1, 'v1block', slice(zc, zc + 256), slice(yc, yc + 256), slice(xc, xc + 256))
print(f'block {B.shape} {B.dtype}, {B.nbytes / 1e6:.0f} MB, from the local cache in {time.time() - t:.1f}s')
"""),
md("""
## 2. What the numbers mean
"""),
code("""
h = np.bincount(B.ravel(), minlength=256)
plt.figure(figsize=(9, 3)); plt.semilogy(h); plt.xlabel('voxel value (uint8)'); plt.ylabel('count (log)')
thr = 255 * 0.35
plt.axvline(thr, c='r', ls='--'); plt.title('intensity histogram: a big air peak, then papyrus'); plt.show()
print(f'fraction above {thr:.0f}: {(B > thr).mean():.3f}')
"""),
md("""
The volume is **masked** (outside the scroll is 0) and quantised to 8 bits. Air sits low; papyrus higher. Three
cautions from the challenge's *Scroll Anatomy* notes:

- **CT intensity is a reconstructed value, not a material label or a density.** The same papyrus can read brighter or
  darker in another scan, at another energy, or in another part of the same scroll.
- **Phase contrast adds fringes.** The scans use phase-contrast imaging (notebook 00), which draws bright and dark
  halos at material boundaries. A halo is not papyrus.
- **Distance in the scan is not distance on the sheet.** Two voxels a few microns apart can lie on wraps that are
  centimetres apart once unrolled.

A threshold separates air from papyrus roughly; it fails where sheets touch, which is exactly where the real models are
needed.

## 3. Three orthogonal views
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
**How papyrus is built, and why that matters for the views.** A sheet is two layers of pith strips pressed together at
right angles. The faces are the **recto** (inside, written) and **verso**. A fibre stays within its own layer. In the
rolled scroll, recto fibres run around the roll and verso fibres run along its axis. Sheets (*kollemata*) were glued end
to end; at a joint (*kollesis*) two sheets overlap within **one** wrap, so "another sheet here" does not mean "another
wrap". The roll winds around a central axis, the **umbilicus**.

## 4. Sheets have a direction: the structure tensor

Locally a sheet is a thin bright layer. Its direction is the direction in which intensity changes fastest. The obvious
estimate, the gradient, fails, and the reason is the whole idea of the structure tensor.

**Counterexample: average the gradient over a sheet.** On a thin bright sheet the gradient points *into* the sheet from
both faces: up on the lower face, down on the upper face. Average it over a neighbourhood and the two cancel.
"""),
code("""
yy = np.arange(64)[:, None] * np.ones((1, 64)); sheet = np.exp(-0.5 * ((yy - 32) / 2.0) ** 2)   # a horizontal sheet
gy, gx = np.gradient(sheet)
print(f'mean gradient over the patch:      ({gx.mean():+.4f}, {gy.mean():+.4f})   <- the two faces cancel')
J = np.array([[np.mean(gx * gx), np.mean(gx * gy)], [np.mean(gx * gy), np.mean(gy * gy)]])
w, V = np.linalg.eigh(J)
print(f'structure tensor eigenvalues {w.round(5)}, top eigenvector {V[:, 1].round(3)}   <- points across the sheet (y)')
fig, ax = plt.subplots(1, 2, figsize=(9, 3.5)); ax[0].imshow(sheet); ax[0].set_title('a thin sheet')
ax[1].plot(gy[:, 32], label='gradient across the sheet'); ax[1].plot(gy[:, 32] ** 2, label='its square'); ax[1].legend()
ax[1].set_title('the gradient changes sign, its square does not'); plt.show()
"""),
md("""
The **structure tensor** averages the outer product g gᵀ instead of g. Squaring removes the sign, so both faces add up.
Its top eigenvector is the sheet normal (up to sign), and the gap between its eigenvalues says how sheet-like the
neighbourhood is:

| eigenvalues λ₁ ≥ λ₂ ≥ λ₃ | neighbourhood |
|---|---|
| λ₁ ≫ λ₂ ≈ λ₃ | a sheet: one strong direction |
| λ₁ ≈ λ₂ ≫ λ₃ | a fibre or a fold line: two strong directions |
| λ₁ ≈ λ₂ ≈ λ₃ | noise or a blob: no direction |

Two scales matter: the smoothing before the gradient (here 1 voxel) and the window of the average (here 3 voxels).
Too small a window sees noise; too large a window averages two wraps into one.
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
print(f'median coherence on papyrus voxels: {np.median(coh[B[128] > thr]):.2f}')
"""),
md("""
Red arrows: the estimated normal (in-plane part). They point across the wraps. Coherence drops where wraps touch or
the sheet is damaged: exactly where tracers go wrong.

## 5. What the challenge uses instead

| task | classical (this notebook) | what the challenge runs now |
|---|---|---|
| where is papyrus? | a threshold | **surface prediction**: 3D nnU-Net ensembles (`surface_m7_nnunet` on huggingface.co/scrollprize); the March 2026 Kaggle Surface Detection competition targeted topologically correct masks |
| which way is the sheet? | structure tensor | **lasagna normals**: predicted surface normals stored as zarr volumes, used by VC3D and the spiral fit |
| which wrap is this? | nothing local can tell | **winding constraints** + the global spiral fit (notebook 05) |

The structure tensor still matters: it needs no training data, it is fast, and it is a sanity check on any learned
normal field.

## Ideas to try

1. Compute coherence at window sizes 1, 3 and 6 and watch where neighbouring wraps merge.
2. Use the second eigenvector at a smaller scale to find **fibre** direction inside a sheet (recto vs verso).
3. Read the same block at level 0 and level 2: how does coherence change with resolution?

## Check yourself

1. Why is a global threshold not enough to separate two touching wraps?
2. Why does averaging the gradient over a sheet give almost zero, and why does the structure tensor not?
3. The structure tensor's normal has a sign ambiguity. Why does that matter when you sample "in front of" a sheet?
4. A read of a 256³ block takes 27 chunks instead of 8. What changed, and how do you avoid it?
5. Which pyramid level would you use to find the scroll's outline, and which to see fibres?
"""),
]

# ---------------------------------------------------------------- 02
NB["02_flattening"] = [
md("""
# 02 · Virtual unwrapping by hand

**The problem.** Ink detection needs the papyrus as a flat image. But in the scan the sheet is a curved, crushed
surface inside a 3D block. **Virtual unwrapping** has two halves: *segmentation* (find the sheet and describe it as a
surface) and *rendering* (sample the volume along that surface to get a flat stack of layers). This notebook takes a
sheet someone already traced and does the rendering ourselves, then checks the result against the official one, and
measures how distorted the traced sheet is.

**Data.**

| object | what it is | size |
|---|---|---|
| `mesh/*.tifxyz` | three float images `x, y, z`: grid point (i, j) is a 3D position in the scroll (level-0 voxels) | 1.5 MB |
| scroll volume, level 0 | the 9.362 µm scan, streamed as one small crop | ~35 MB here |
| official surface volume | 28 layers × 7,280 × 6,800, the challenge's own rendering of this sheet | we read one tile |

The mesh grid is stored at `scale = 0.05`: one grid step is 20 canvas pixels.

```
 flat canvas (u, v)          tifxyz grid (i, j) = (u, v) · 0.05          3D scan
 ┌────────────────┐          x[i, j], y[i, j], z[i, j]                    the sheet, curved
 │ pixel (u, v)   │ ──────►  interpolate at (u/20, v/20)  ──────►  point P(u, v) + k · normal, k = −14 … 13
 └────────────────┘                                                       → 28 layers of the surface volume
```
"""),
SETUP,
code("""
x, y, z, meta = vlab.segment_mesh()
print('grid', x.shape, '| scale', meta['scale'], '| valid points', f'{(x > 0).mean():.1%}', '| bbox (x, y, z)', np.round(meta['bbox']).tolist())
print('first valid row of points:'); r = np.argwhere(x > 0)[0]
print(' (x, y, z) =', [round(float(c[r[0], r[1] + k]), 1) for k in range(3) for c in (x, y, z)][:9])
"""),
md("""
**Where meshes come from.** Segments like this one are made in **VC3D** (Volume Cartographer 3D), the challenge's
desktop tool. The workflow: load a volume, overlay the pre-computed surface predictions (notebook 01, section 5), grow a
patch from a seed point with **GrowPatch** (one growth step is about 20 voxels, which is why the grid spacing here is
20), correct it by hand, add winding and fibre annotations, export. Every segment VC3D writes, whatever its size, is a
tifxyz like this one.

## 1. The sheet in 3D
"""),
code("""
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
m = x > 0; step = 6
fig = plt.figure(figsize=(8, 6)); ax = fig.add_subplot(projection='3d')
ax.scatter(x[::step, ::step][m[::step, ::step]], y[::step, ::step][m[::step, ::step]], z[::step, ::step][m[::step, ::step]],
           s=1, c=z[::step, ::step][m[::step, ::step]], cmap='viridis')
ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z (scroll axis)'); ax.set_title('the traced sheet: part of one wrap'); plt.show()
"""),
md("""
## 2. How distorted is the flat image?

You cannot flatten a curved, crumpled surface without stretching it (think of flattening an orange peel). The tifxyz
grid *is* the flattening: neighbouring grid points are one flat step apart (20 canvas pixels), whatever their distance in
3D. So the 3D distance between neighbours, divided by 20, is the **stretch** of the flat image there: 1 means the flat
image is true to scale, 1.3 means the papyrus is 30% longer than it looks.
"""),
code("""
P3 = np.stack([x, y, z], -1).astype(np.float64); ok = x > 0
di = np.linalg.norm(P3[1:, :] - P3[:-1, :], axis=-1) / 20; di[~(ok[1:] & ok[:-1])] = np.nan
dj = np.linalg.norm(P3[:, 1:] - P3[:, :-1], axis=-1) / 20; dj[~(ok[:, 1:] & ok[:, :-1])] = np.nan
cr = np.linalg.norm(np.cross(P3[1:, :-1] - P3[:-1, :-1], P3[:-1, 1:] - P3[:-1, :-1]), axis=-1) / 400
cr[~(ok[1:, :-1] & ok[:-1, 1:] & ok[:-1, :-1])] = np.nan
a3, b3 = P3[1:, :-1] - P3[:-1, :-1], P3[:-1, 1:] - P3[:-1, :-1]
with np.errstate(invalid='ignore'):
    ang = np.degrees(np.arccos(np.clip((a3 * b3).sum(-1) / np.linalg.norm(a3, axis=-1) / np.linalg.norm(b3, axis=-1), -1, 1)))
ang[np.isnan(cr)] = np.nan
q = np.nanpercentile(ang, [1, 50, 99]); print(f'angle between grid directions: median {q[1]:.2f}°, 1-99% {q[0]:.2f}° … {q[2]:.2f}°')
for name, s in (('stretch along i', di), ('stretch along j', dj), ('area ratio', cr)):
    q = np.nanpercentile(s, [5, 50, 95]); print(f'{name:16s}: median {q[1]:.3f}, 5-95% {q[0]:.3f} … {q[2]:.3f}, beyond ±25%: {np.nanmean(np.abs(s - 1) > .25):.1%}')
fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
for a, s, name in zip(ax, (di, dj, cr), ('stretch along i', 'stretch along j', 'area ratio')):
    im = a.imshow(s, cmap='coolwarm', vmin=0.9, vmax=1.1); a.set_title(name); plt.colorbar(im, ax=a, shrink=.8)
plt.show()
"""),
md("""
**Measured: this mesh is almost perfectly isometric.** Stretch stays within about ±3% (5–95%), grid angles stay within
a few degrees of 90°, and the total 3D area is within half a percent of the flat area. The maps (colour range ±10%) are nearly white.
That is not luck: the tracer grows the grid in fixed steps of 20 voxels and keeps its cells square, so the flat image is
true to scale *by construction*.

So where did the distortion go? A curved sheet that cannot be flattened (a dome, a crumple) cannot be covered by a
square grid without stretching. A tracer that refuses to stretch has to **stop**: the grid gets holes or ragged edges
instead. **HYPOTHESIS (unverified):** this is one reason segments are patches of a wrap rather than whole wraps, and why the invalid points (−1)
in the grid matter as much as the valid ones. A flattening that covers a whole crumpled wrap has to trade stretch
against holes.

## 3. Render a patch

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
md("## 4. Check it against the official rendering"),
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

**The challenge's own quality check** (from the VC3D tutorial): a good segment follows the sheet in the cross-section
views, *and* shows the horizontal and vertical **fibres** clearly in the flattened view. Fibres are the papyrus's own
grid, so if they look straight and crossed at right angles in the rendering, the mesh is on the sheet.

## 5. Why 28 layers?
"""),
code("""
prof = mine.reshape(28, -1).mean(1)
plt.figure(figsize=(7, 3)); plt.plot(np.arange(28) - 14, prof, 'o-'); plt.axvline(0, c='r', ls='--')
plt.xlabel('offset along the normal (voxels)'); plt.ylabel('mean intensity'); plt.title('the sheet is the bright band around offset 0'); plt.show()
"""),
md("""
The papyrus is a few voxels thick and the mesh is not exactly on it. Rendering a slab of layers lets the ink model
look *into* the sheet and forgive a mesh that is off by a few voxels.

## 6. How tracing goes wrong

| failure | what happens | how it shows in the rendering |
|---|---|---|
| **sheet switch** | the mesh leaves its wrap for the neighbouring one, often where wraps touch | text jumps mid-line; fibres change direction; letters from another page appear |
| **wandering** | the mesh oscillates along the normal and briefly touches neighbours | the sheet drifts out of the central layers and back |
| **stretch** | the flat grid is far from isometric (section 2) | letters squashed or elongated |
| **holes** | the tracer could not continue | blank regions |

Local information cannot always tell which wrap you are on: two touching wraps look the same. That is why the
challenge now fits the **whole scroll** as one deformed spiral (scrollprize.org/tutorial_spiral), using sparse
*winding constraints*: "these points are on the same wrap", "this point is one wrap outward of that one".
Notebook 05 builds a toy version of that idea.

## Ideas to try

1. Render the same patch with normals smoothed over 3 × 3 grid points: does the correlation go up?
2. Plot where the grid is invalid (−1) and compare with the stretch map: do holes sit next to the most distorted cells?
3. Render a patch along a different segment of PHerc. 139 (39 are in the bucket, notebook 00).

## Check yourself

1. What happens to the rendering if a normal flips sign halfway across the patch?
2. Why does the slab have to be wider when the mesh was traced at a coarser resolution?
3. How would a sheet switch look in the rendered layers?
4. The stretch map is almost white. Does that prove the flattening is right? (Hint: what would a sheet switch do to it?)
5. Why are fibres a better check of a mesh than the brightness of the central layer?
"""),
]

# ---------------------------------------------------------------- 03
NB["03_ink_detection"] = [
md("""
# 03 · Ink detection: a small 2.5D CNN

**The problem.** After flattening (notebook 02) the sheet is a stack of layers parallel to the papyrus. Somewhere in
that stack is ink, but Herculaneum ink is carbon on carbonised papyrus: it absorbs X-rays almost exactly as the sheet
does. Nobody can see the letters in a single layer. Models find them from **texture across depth**.

**Goal.** Train a model that looks at a stack of surface layers and predicts ink, and measure how far a tiny model gets.

**What ink looks like in the scan.** From the challenge's *Ink Detection* and *Scroll Anatomy* notes: ink may show as
a surface deposit, a raised texture, a **crackle** ("a texture like cracked mud, raised slightly above the papyrus"), or
subtler changes of contrast and texture. On PHerc. Paris. 4 the crackle is crisp; on other scrolls the signal is "more
subtle and heterogeneous". Casey Handmer first spotted crackle by eye in 2023 (notebook 00).

**Where labels come from.**

| source | how | strength | weakness |
|---|---|---|---|
| detached fragments | pieces opened long ago are photographed in infrared (1000 nm), where the ink is crisp, and aligned to their scan | real ground truth | fragments are few, and their surface differs from the inside of a closed scroll |
| crackle by eye | humans mark strokes where crackle is visible in the surface volume | works inside closed scrolls | only where crackle is clear |
| iterative pseudo-labels | run a model, a human labels the strokes that are clearly visible, retrain, repeat | scaled from single letters to whole scrolls (PHerc. 1667) | errors can reinforce themselves |
| another model's output (this notebook) | the official 2.4 µm ink prediction as the label | free, dense | we learn that model's mistakes too |

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

```
 ┌──────────── sheet (3,400 columns) ────────────┐
 │  train: columns 0 … 2,039       │  validate:  │
 │  (random 64 × 64 patches)        │  2,040 …    │
 └──────────────────────────────────┴─────────────┘
```
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

```
 16 layers × 64 × 64 ──3D conv ×3──► 32 channels × 16 × 64 × 64 ──max over depth──► 32 × 64 × 64 ──2D conv ×3──► ink logit 64 × 64
```

**Why a max over depth?** The ink sits at some depth near the surface, but the mesh is off by a few voxels and not by the
same amount everywhere. The max asks "is there an ink-like feature at *any* depth here?", so it forgives a mesh that
wanders. The challenge's trainer also offers `mean`, `logsumexp` and a learned MLP.

**How this compares with the real models** (from the challenge's tutorial and the 2023 Kaggle competition):

| model | input | how depth is handled |
|---|---|---|
| ours | 16 layers × 64², 18.7 µm | 3D convs, then max |
| challenge tutorial, 2.5D | `[64, 256, 256]` surface-volume patches | U-Net (`vesuvius_unet`), max projection |
| challenge, native 3D (`full_3d`) | `[80, 128, 128]` crops of the raw scan, labels projected ±8 voxels along the normal | fully 3D; `full_3d_single_wrap` adds a channel saying which wrap each voxel belongs to |
| challenge 9 µm cross-scroll model (`scrollprize/ink_9um`) | 17-slice patches | small 3D stem pools depth to a 2D map, then a 2D U-Net |
| 2023 Kaggle ink detection, 1st place | fragment surface volumes | 3D CNN / 3D U-Net / UNETR encoders, flattened along depth, then a 2D SegFormer; ensemble of 9 models |
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

## The traps the challenge guards against

| trap | what goes wrong | the guard |
|---|---|---|
| **a model that draws letters** | a model with a big enough view learns what Greek letters look like and paints plausible ones from noise | patches *smaller than a letter*, so the model only ever judges strokes, never letterforms |
| **leaky validation** | neighbouring patches share pixels | spatially held-out regions, excluded from the loss entirely |
| **self-reinforcing labels** | a pseudo-label loop confirms its own mistakes | label only strokes that are clearly and repeatably visible; papyrologists review the final reading |
| **one-scroll models** | a model trained on PHerc. Paris. 4's crisp crackle fails on a scroll with a subtler signal | train on several scrolls at once: the 9 µm model mixes four scrolls with fixed per-batch quotas (29/22/11/2 of 64) |

**Generalisation across scrolls is open problem 7**, and it is the closest to an ordinary Kaggle question: a model that
works on the scrolls it saw is easy; one that works on the next scroll is the prize.

## Ideas to try

1. Replace the max over depth with a mean and with logsumexp; which holds up on the held-out strip?
2. Feed 8, 16 and 24 layers: where does the ink signal live in depth?
3. Train for 5,000 steps: the curve was still rising at 1,500.
4. Weight the loss by the label's confidence to stop learning the speckle.

## Check yourself

1. Why would a random train/validation patch split overstate the score here?
2. What does `amax` over depth assume about where the ink sits in the stack?
3. The pseudo-label is speckled in the lower half. How would you stop the model learning the speckle?
4. A model trained on one fragment is tested on another. What do you expect to happen to the correlation, and why?
"""),
]

# ---------------------------------------------------------------- 04
NB["04_self_supervised"] = [
md("""
# 04 · Few labels: does self-supervised pretraining help?

**The question.** Labels are scarce in this challenge; unlabelled scan is not. If we first teach the encoder papyrus
texture from **unlabelled** layers, does it then learn ink from fewer labels than an encoder trained from scratch?

**Why it is worth asking.** A closed scroll holds terabytes of scan and only a few labelled strokes. Every scroll is a
new domain (open problem 7). If an encoder can learn papyrus from the scroll itself, without labels, it might adapt to a
new scroll where no labels exist. The challenge is betting on this too: open problem 9 is self-supervised 3D
representations, and its "dinovol" model is a DINO-style 3D encoder for the volumes.

**The family of methods.**

| method | pretext task | what it must learn | here |
|---|---|---|---|
| masked image modelling (MAE, SimMIM) | hide blocks, reconstruct them | local texture, how layers continue in depth | **this notebook** |
| contrastive (SimCLR) | two augmented views of one patch must match, other patches must not | what survives the augmentations | try next |
| self-distillation (DINO, dinovol) | a student matches a slowly updated teacher across views | invariant features without negatives | the challenge's 3D encoder |

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

**Measured:** pretraining gives no consistent gain. The difference "pretrained − scratch" is −0.007, +0.022, −0.068 and
+0.009 at 2%, 10%, 50% and 100% of the labels: no direction, and no shrinking gap as labels grow. That is the pattern
the experiment called a null in its opening table.

- **One seed per cell.** Two runs differing only in their seed can move by a few hundredths at 600 steps, so a gap
  smaller than that is not evidence. Rerun with seeds 1 and 2 before believing a small difference.
- **Why scratch at 100% scores 0.196 here and 0.315 in notebook 03.** This notebook fine-tunes for 600 steps; notebook
  03 trains for 1,500 and its curve was still rising. Both arms here are undertrained by the same amount, which is fair
  between them but not comparable with 03.
- The label is a pseudo-label (notebook 03). "Better" means "closer to the 2.4 µm model", not "closer to the true ink".
- Pretraining used only the left 60% too. Pretraining on the validation strip's unlabelled pixels is allowed in
  practice (you always have the unlabelled scroll you want to read), and is a fair follow-up: it is test-time
  self-supervision, aimed at the cross-scroll problem.

**Why masking might not help here, as hypotheses** (HYPOTHESIS, unverified):

1. Reconstructing 8 × 8 blocks of fibre texture can be done from the visible neighbours without modelling ink at all.
   Ink is a small fraction of the pixels; a reconstruction loss barely notices it.
2. 1,500 pretraining steps on one segment is tiny next to MAE-scale pretraining.
3. The pseudo-label measures closeness to another model, so a better ink feature can still score worse.

## Try next

1. Mask ratio 0.4 vs 0.8; block size 4 vs 16.
2. Mask whole *layers* instead of blocks: the model must infer depth structure.
3. SimCLR instead of masking: two augmented views of one patch as positives.
4. Pretrain on a different segment, fine-tune on this one: does the texture knowledge transfer?
5. Three seeds per cell, so a 0.02 difference can be told from noise.
6. Use the challenge's pretrained `dinovol` encoder as the frozen backbone instead of our own pretraining.

## Check yourself

1. Why is the validation strip excluded from *pretraining* in this experiment?
2. Why does the gap between the arms have to shrink as labels grow, if SSL is working?
3. What would it mean if the pretrained arm were **worse** at 100% labels?
"""),
]

# ---------------------------------------------------------------- 05
NB["05_uncrushing_flows"] = [
md("""
# 05 · Uncrushing a scroll with flows

**The problem.** A scroll was a neat spiral of papyrus before Vesuvius crushed it. If we could undo the crush, every wrap
would be a clean circle-like turn, and following one sheet would be easy. Nobody has an uncrushed copy of a real scroll,
so there is nothing to learn the inverse from directly.

**The idea (a friend's question).** Simulate the crush on perfect synthetic spirals, which gives unlimited
(crushed, ideal) pairs, and train a network to undo it. The worry: the network only learns the inverse of *our
simulator*, so it works on real scrolls only if real crushes look like simulated ones. We test exactly that at the end.

```
 synthetic ideal spiral ──crush: flow of a random velocity field v──► crushed image
                                                                          │
                                                       U-Net predicts u ≈ v
                                                                          ▼
          uncrushed image  ◄──── run the flow of u on the crushed image ──┘
```

**Why flows.** A crush that never tears the sheet is a smooth invertible map. If we model it as the *flow* of a
velocity field (integrate dx/dt = v(x) for unit time), three things come for free:

| property | why it matters here |
|---|---|
| the inverse is the flow of −v | uncrushing is the same machinery run backwards |
| trajectories cannot cross | two wraps can never be pushed through each other |
| the Jacobian determinant stays > 0 | the map never folds the sheet onto itself |

**The challenge already does this, by optimisation.** Its *Spiral Fitting* method (scrollprize.org/tutorial_spiral)
models the ideal scroll as an Archimedean spiral r(θ) = (ω / 2π) θ extruded along the axis, and the crush as global
scale and shear followed by the flow of a velocity field, dφ/dt = u(φ), integrated with Runge–Kutta steps. It fits all
of it with plain Adam, **per scroll**, against sparse evidence: traced patches, fibres, and *winding constraints* ("these
points are on one wrap", "this point is k wraps outward of that one"). For PHerc. Paris. 4 it fitted 130 windings.

This notebook asks the amortised version of the same question: can a network trained once on synthetic crushes predict
the velocity field directly from an image, with no per-scroll fitting and no annotations?

| | spiral fit (the challenge) | this notebook |
|---|---|---|
| deformation model | scale/shear + velocity-field flow + gap scaling | squash + shift + velocity-field flow |
| how u is found | optimised per scroll | predicted by a U-Net trained on synthetic pairs |
| evidence | patches, fibres, winding constraints, normals | the image only |
| dimension | 3D | 2D cross-sections |

**Data.** Synthetic spirals made in `flowlab.py` (no download), then one real cross-section of PHerc. 139 at pyramid
level 4 (150 µm voxels, streamed from S3) with the sheet someone traced through it (the mesh of notebook 02).
"""),
SETUP,
code("""
import torch, pandas as pd, flowlab as fl
print('device', fl.DEV, '| image size', fl.SIZE)
rng = np.random.default_rng(0)
deformed, v, ideal = fl.batch(4, rng)
p = fl.spiral_params(4, np.random.default_rng(0))
pd.DataFrame({k: np.round(p[k], 4) for k in ('r0', 'pitch', 'rmax', 'width')}).assign(
    wraps=lambda d: ((d.rmax - d.r0) / d.pitch).round(1))
"""),
md("""
The spiral parameters, in normalised units (the image spans −1 … 1, so 0.01 ≈ 1.3 pixels): inner radius `r0`, gap
between wraps `pitch`, outer radius `rmax`, sheet thickness `width`. Below: ideal (top) and crushed (bottom).
"""),
code("""
fig, ax = plt.subplots(2, 4, figsize=(13, 6.5))
for i in range(4):
    ax[0, i].imshow(ideal[i, 0].cpu()); ax[1, i].imshow(deformed[i, 0].cpu())
    ax[0, i].set_title(f'ideal {i}'); ax[1, i].set_title(f'crushed {i}')
for a in ax.ravel(): a.axis('off')
plt.tight_layout(); plt.show()
"""),
md("""
## 1. A crush is a flow

`random_velocity` builds v(x) = A·x + t + smooth noise: a squash along a random axis (A symmetric, so no rotation),
a small shift, and a wobble. `flow` integrates it by **scaling and squaring**: take v / 2⁷ as a tiny displacement, then
compose the map with itself 7 times (2⁷ = 128 Euler steps for the price of 7). The grid shows what the map does to the
ideal image's coordinates.
"""),
code("""
phi = fl.flow(v[:1]); back = fl.flow(-v[:1])
g = phi[0].cpu().numpy()
fig, ax = plt.subplots(1, 2, figsize=(11, 5))
for k in range(0, fl.SIZE, 16):
    ax[0].plot(g[k, :, 0], g[k, :, 1], 'b', lw=.6); ax[0].plot(g[:, k, 0], g[:, k, 1], 'b', lw=.6)
ax[0].set_aspect('equal'); ax[0].invert_yaxis(); ax[0].set_title('the crush exp(v) applied to a square grid')
ax[1].imshow(deformed[0, 0].cpu()); ax[1].set_title('crushed image 0'); ax[1].axis('off'); plt.show()
comp = fl.warp(phi.permute(0, 3, 1, 2), back).permute(0, 2, 3, 1)
inside = (back.abs() < 0.95).all(-1)
err = (comp - fl.grid())[inside].norm(dim=-1)
print(f'exp(v)(exp(-v)(x)) vs x: median error {err.median():.5f}, max {err.max():.5f} (one pixel = {2 / 255:.5f}), '
      f'on the {inside.float().mean():.0%} of points whose inverse stays inside the image')
"""),
md("""
**Counterexample: why not just predict a displacement?** A displacement field d gives the map x + d(x). Make the
deformation three times stronger and compare it with the flow of the same field.
"""),
code("""
big = 3 * v
dd = fl.jacobian_det(fl.grid() + big.permute(0, 2, 3, 1)).cpu(); df = fl.jacobian_det(fl.flow(big)).cpu()
print('pixels with det <= 0, per sample:  displacement', [f'{f:.1%}' for f in (dd <= 0).float().mean((1, 2))],
      '| flow', [f'{f:.1%}' for f in (df <= 0).float().mean((1, 2))])
i = int((dd <= 0).float().mean((1, 2)).argmax()); det_disp, det_flow = dd[i], df[i]
fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
for a, d, name in ((ax[0], det_disp, 'x + 3v(x)'), (ax[1], det_flow, 'flow of 3v')):
    im = a.imshow(d, cmap='RdBu', vmin=-2, vmax=2); a.set_title(f'sample {i}, {name}: det ≤ 0 on {(d <= 0).float().mean():.1%} of pixels'); a.axis('off')
plt.colorbar(im, ax=ax, shrink=.8); plt.show()
"""),
md("""
Red = negative determinant = the map turned the sheet inside out there: two wraps passed through each other. The
displacement folds; the flow of the same field does not, however strong.

## 2. Learn to uncrush (supervised, on synthetic pairs)

A small U-Net reads the crushed image and predicts a velocity field u. The loss compares u with the true v, and the
uncrushed image `warp(crushed, flow(u))` with the ideal. Each step draws 12 fresh pairs, so the network never sees the
same spiral twice. 3,000 steps take about 12 minutes on Apple MPS; the weights are saved under `data/cache/flow05/`
(every 250 steps, with resume), and a rerun loads them.
"""),
code("""
t = time.time(); net, hist = fl.train(); print(f'{time.time() - t:.0f}s')
H = np.array(hist)
plt.figure(figsize=(8, 3)); plt.semilogy(H[:, 0], H[:, 1], label='velocity MSE'); plt.semilogy(H[:, 0], H[:, 2], label='image MSE')
plt.xlabel('step'); plt.legend(); plt.title('training on fresh synthetic pairs'); plt.show()
"""),
md("""
**Held-out synthetic test.** New spirals and new crushes (seed 123). How close to the ideal is the crushed image, and
the uncrushed one? And the question that matters for reading a scroll: does each sheet point land on the **right wrap**?
"""),
code("""
@torch.no_grad()
def uncrush(img):
    u = net(img); return fl.warp(img, fl.flow(u)), u

rng = np.random.default_rng(123); P = fl.spiral_params(16, rng); I0, _ = fl.spiral_image(P)
V = fl.random_velocity(16, rng); D = (fl.warp(I0, fl.flow(-V)) + 0.06 * torch.randn_like(I0)).clamp(0, 1)
U_img, U = uncrush(D)
corr = lambda a, b: float(np.corrcoef(a.cpu().numpy().ravel(), b.cpu().numpy().ravel())[0, 1])
rows = []
for i in range(16):
    x, turn = fl.spiral_points(P, i, 1500, rng)
    y = fl.at_points(fl.flow(V[i:i + 1]), x)                 # where each sheet point sits in the crushed image
    back_pts = fl.at_points(fl.flow(-U[i:i + 1]), y)         # our uncrush, applied to those points
    rows.append(dict(crushed_vs_ideal=corr(D[i], I0[i]), uncrushed_vs_ideal=corr(U_img[i], I0[i]),
                     right_wrap_before=(fl.turn_of(y, P, i) == turn).mean(), right_wrap_after=(fl.turn_of(back_pts, P, i) == turn).mean(),
                     within_one_wrap_after=(np.abs(fl.turn_of(back_pts, P, i) - turn) <= 1).mean()))
R = pd.DataFrame(rows); print(R.mean().round(3).to_string())
fig, ax = plt.subplots(3, 4, figsize=(13, 10))
for c, i in enumerate(range(4)):
    for r, (img, name) in enumerate(((D, 'crushed'), (U_img, 'uncrushed'), (I0, 'ideal'))):
        ax[r, c].imshow(img[i, 0].cpu()); ax[r, c].set_title(f'{name} {i}'); ax[r, c].axis('off')
plt.tight_layout(); plt.show()
"""),
md("""
## 3. No pairs at all: flow matching with optimal-transport couplings

On a real scroll we have the crushed sheet and only a *description* of the ideal (a spiral), never the pairing. Flow
matching can still learn a map between two unpaired point clouds: draw a minibatch from each, pair them by **optimal
transport** (the matching that moves the least squared distance), and regress a velocity field onto the straight lines
between the pairs (OT-CFM, Tong et al. 2023). We also run exact OT on all 1,500 points, the limit the minibatches
approximate.

**The catch, predicted before running it.** OT pairs each point with the *nearest* free ideal point. A real crush is not
least-movement: squash a circle into an ellipse, and the points on the short axis are closest to an *inner* wrap of the
ideal spiral. So OT should put many points on the wrong wrap, and more so the harder the crush.
"""),
code("""
rows, keep = [], None
for name, crush, wob in (('mild', (0.9, 1.0), 0.0), ('medium', (0.75, 0.95), 0.06), ('hard', (0.55, 0.75), 0.12)):
    rng = np.random.default_rng(7); p = fl.spiral_params(1, rng); vv = fl.random_velocity(1, rng, crush=crush, wobble=wob)
    x, turn = fl.spiral_points(p, 0, 1500, rng); y = fl.at_points(fl.flow(vv), x)
    x_free, _ = fl.spiral_points(p, 0, 1500, rng)                       # an UNPAIRED sample of the ideal
    ot = x_free[fl.ot_pairs(y, x_free)]
    cfm = fl.transport(fl.ot_flow_matching(y, x_free, steps=1500), y)
    img = (fl.warp(fl.spiral_image(p)[0], fl.flow(-vv)) + 0.06 * torch.randn(1, 1, fl.SIZE, fl.SIZE, device=fl.DEV)).clamp(0, 1)
    sup = fl.at_points(fl.flow(-uncrush(img)[1]), y)
    ok = lambda z: (fl.turn_of(z, p, 0) == turn).mean()
    rows.append(dict(crush=name, identity=ok(y), exact_OT=ok(ot), OT_flow_matching=ok(cfm), supervised_UNet=ok(sup)))
    if name == 'hard': keep = (y, ot, cfm, sup, turn)
print('fraction of sheet points on the right wrap'); print(pd.DataFrame(rows).set_index('crush').round(2).to_string())
y, ot, cfm, sup, turn = keep
fig, ax = plt.subplots(1, 4, figsize=(16, 4.3))
for a, z, name in zip(ax, (y, ot, cfm, sup), ('crushed (hard)', 'exact OT', 'OT flow matching', 'supervised U-Net')):
    a.scatter(z[:, 0], z[:, 1], c=turn % 7, cmap='tab10', s=2); a.set_aspect('equal'); a.invert_yaxis(); a.set_title(name); a.axis('off')
plt.suptitle('colour = the wrap each point really belongs to; a clean result shows clean coloured rings'); plt.show()
"""),
md("""
## 4. The real test: a crushed cross-section of PHerc. 139

A level-4 slice through the middle of the traced segment w025, and the traced sheet itself (mesh points within half a
level-4 voxel of the slice). If the network learned something about crushes in general, the traced wrap should come
out **rounder**. `roundness` fits a circle centre and a slowly growing radius, and returns the leftover wobble divided by
the radius: 0 is a perfect wrap.
"""),
code("""
x, y, z, meta = vlab.segment_mesh(); m = x > 0
z0 = float(np.median(z[m])); k4 = int(round(z0 / 16))
sl = vlab.crop(vlab.scroll_volume(4), 'v4z', k4).astype(np.float32)
sel = m & (np.abs(z - z0) < 8); curve = np.stack([x[sel], y[sel]], 1) / 16          # level-0 voxels -> level 4
ys, xs = np.where(sl > np.percentile(sl[sl > 0], 30)); pad = 8
c0, r0_ = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2; half = max(xs.max() - xs.min(), ys.max() - ys.min()) / 2 + pad
x0, y0 = int(c0 - half), int(r0_ - half); side = int(2 * half)
crop_ = np.zeros((side, side), np.float32); src = sl[max(y0, 0):y0 + side, max(x0, 0):x0 + side]
crop_[max(-y0, 0):max(-y0, 0) + src.shape[0], max(-x0, 0):max(-x0, 0) + src.shape[1]] = src
lo, hi = np.percentile(crop_[crop_ > 0], [5, 99.5])
img = torch.tensor(np.clip((crop_ - lo) / (hi - lo), 0, 1), dtype=torch.float32)[None, None]
img = torch.nn.functional.interpolate(img, size=(fl.SIZE, fl.SIZE), mode='area').to(fl.DEV)
cn = ((curve - [x0, y0]) / (side - 1) * 2 - 1).astype(np.float32)                    # the traced wrap, normalised
real_un, u_real = uncrush(img)
cn_after = fl.at_points(fl.flow(-u_real), cn)
print(f'slice z = {k4} (level 4) | crop {side}×{side} -> {fl.SIZE} | traced points {len(cn)}')
print(f'roundness of the traced wrap: before {fl.roundness(cn):.4f}, after uncrushing {fl.roundness(cn_after):.4f}  (0 = perfect)')
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
for a, im, c, name in ((ax[0], img, cn, 'real crushed slice + traced wrap'), (ax[1], real_un, cn_after, 'after our uncrush')):
    a.imshow(im[0, 0].cpu(), extent=(-1, 1, 1, -1)); a.plot(c[:, 0], c[:, 1], 'r.', ms=2); a.set_title(name); a.axis('off')
plt.show()
"""),
md(r"""
## 5. Reading the results

**Measured** (one training run, seed 0, 3,000 steps; held-out synthetic seed 123; OT seed 7):

| test | result |
|---|---|
| invertibility: exp(v) after exp(−v) | median error 0.00007, i.e. 1% of a pixel |
| folds at 3× strength | displacement: 0.9%, 23.4%, 51.8%, 0.0% of pixels on the four samples; flow: 0% on all |
| synthetic, image correlation with the ideal | crushed 0.456 → uncrushed 0.588 |
| synthetic, points on the **right wrap** | 13.0% before → 22.4% after; within one wrap 64.0% |
| unpaired, right wrap on a hard crush | exact OT 48%, OT flow matching 26%, supervised U-Net 11% |
| real PHerc. 139 slice, roundness of the traced wrap | 0.196 before → **0.233 after (worse)** |

1. **The flow does what it promises.** Invertible to 1% of a pixel, and never folds, even where the same field used as
   a displacement folds half the image. This part is solid.
2. **The supervised uncrusher learns the coarse shape, not the wraps.** The outline comes back round and the image
   correlation rises, but only about a fifth of the sheet points land on their own wrap. Two wraps are 2–6 pixels
   apart here; the predicted velocity is off by about 13 pixels (RMS, from the training loss). Wrap-level accuracy needs
   an error below half a wrap spacing, and this model is far from that.
3. **A trade-off we measured, not guessed.** A first version predicted the velocity at full resolution: image
   correlation 0.665 but only 14.3% on the right wrap, and smeared images, because pixel-level noise in u scrambles the
   sheet. Predicting u on a 16 × 16 grid (this version) lowers the image score to 0.588 but raises the right-wrap
   fraction to 22.4% (within one wrap: 40.8% → 64.0%). A crush is smooth; the model should be too.
4. **OT beat the U-Net on wraps, but the comparison flatters OT.** It was given an exact sample of the *true* ideal
   spiral (its spacing, centre and start angle), which the U-Net never sees. Even so, on a hard crush it put 52% of the
   points on the wrong wrap while drawing perfect-looking rings: the plot looks solved and is half wrong. That is the
   catch of least-movement matching, and why the spiral fit relies on winding constraints rather than on matching alone.
5. **The real test failed.** On the real slice the traced wrap came out *less* round (0.196 → 0.233). The synthetic
   spirals are thin bright lines on black; the real cross-section is dense grey papyrus with a crushed, lumpy outline
   and about 30–50 wraps (notebook 00). The network learned the inverse of *our* simulator, which is exactly the worry
   in the opening question, and the real data is outside it.

**What would have to change for this to work on real scans**, as hypotheses to test (HYPOTHESIS, unverified):

- a simulator closer to the scan: real wrap spacing and count, grey papyrus texture, touching and delaminated wraps,
  the scan's noise, and domain randomisation wide enough that the real slice is one case among many;
- an error measured in wrap spacings, and a loss that punishes wrap swaps directly (for example, a winding-number term);
- using the network only as the **initial guess** for the challenge's per-scroll spiral fit, which then corrects it
  against real constraints: amortised inference plus refinement, rather than either alone.

## Ideas to try

1. Train on spirals with 30–50 wraps and real-looking texture (sample patches of the real scan as the fill).
2. Feed two inputs: the image and a structure-tensor normal field (notebook 01), so the network sees sheet direction.
3. Add a few winding constraints as an extra input channel: does 1% of annotated points fix the wrap error?
4. Replace the U-Net with OT flow matching conditioned on the image.

## Check yourself

1. Why is the inverse of exp(v) exactly exp(−v) for a stationary velocity field, and why is that not true for a
   displacement field?
2. Why do trajectories of a flow never cross, and why does that stop two wraps passing through each other?
3. The OT result drew perfect rings with half the points on the wrong ring. Why does least-movement matching produce
   that?
4. Why did predicting the velocity on a coarse grid lower the image score but raise the wrap score?
5. What does the failed real test tell you about the simulator, and what is the cheapest change to test first?
"""),
]

import sys
for name, cells in NB.items():
    if len(sys.argv) > 1 and name not in sys.argv[1:]: continue
    first = next(k for k, c in enumerate(cells) if c.cell_type == 'code')
    nb = nbf.v4.new_notebook(); nb["cells"] = cells[:first] + [COLAB] + cells[first:]
    nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nbf.write(nb, f"notebooks/{name}.ipynb"); print("wrote", name, len(cells), "cells")
