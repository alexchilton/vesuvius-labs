# Vesuvius Challenge: what it is, what is open, what we can run here

Researched 2026-10-25 from scrollprize.org (front page, /prizes, /data, /tutorial, /tutorial5, /2026_open_problems) and the open S3 bucket.
Numbers marked *measured* were read from the bucket in this session; everything else is quoted from the site.

## The problem in one paragraph

In 79 AD Vesuvius buried the library of a villa at Herculaneum.
Its papyrus scrolls were carbonised into lumps that crumble if unrolled.
The challenge reads them without opening them: CT-scan the scroll, find the papyrus sheet inside the 3D volume, flatten it into a 2D image, and detect the ink.
The ink is carbon on carbon, so it is nearly invisible in X-ray; models learn it from texture ("crackle").

## The pipeline

```
scan (CT, 9.4 / 2.4 / 1.1 um voxels)
  -> surface prediction (where is papyrus? 3D segmentation)
  -> mesh tracing (one continuous sheet, no layer switches)
  -> flattening + rendering (sample the volume along the mesh and its normals -> "surface volume", ~28-65 layers)
  -> ink detection (2.5D / 3D model: surface volume patch -> 2D ink map)
  -> papyrologists read it
```

## Prizes (from /prizes)

| prize | amount | deadline | what wins |
|---|---|---|---|
| 2027 Grand Prize | $1M ($800k first) | 2027-06-25 | fully unroll one of 13 eligible scrolls, 100% of the recto, 70% of preserved characters legible, open-source pipeline integrated in VC3D, at most 8 h of human annotation |
| First Letters | $50k per scroll, up to 9 | 2027-06-25 | 10 letters in one 4 cm^2 area of an eligible scroll, no train/predict overlap, false-positive strategy |
| PHerc. Paris. 4 title | $50k | 2027-06-25 | the title of Scroll 1, legible, held-out validated |
| Monthly progress | $20k guaranteed best + $250 to $20k others | last day of each month | released tools and weights, adoption, measured improvement, bug fixes, documentation, the GitHub wishlist |

The monthly prizes reward open tools and documentation, the realistic entry point for a newcomer.

## The ten open problems (from /2026_open_problems)

1. Scan quality in compressed regions (phase retrieval, scan recipes).
2. Surface prediction and topology (nnU-Net ensembles; holes, mergers, sheet switches; topology-aware losses).
3. Mesh tracing and geometry (GrowPatch, lasagna, VC3D).
4. Label quality (human meshes are approximate; label snapping, active learning).
5. Fiber tracing (papyrus fibres give the U/V axes).
6. Spiral fit and global priors (diffeomorphic spiral fitting, winding numbers).
7. **Ink detection generalisation** (models fail across scrolls; the closest to our Kaggle experience).
8. Direct 3D ink segmentation (2.4 um; ink_3d_dino_guided).
9. Self-supervised 3D representations (3D DINO, "dinovol").
10. Data infrastructure (TB volumes, OME-Zarr, streaming).

Code: github.com/ScrollPrize/villa (monorepo: vesuvius package, VC3D, lasagna, spiral fitting, dinovol).
Models: huggingface.co/scrollprize (ink_9um, ink_canonical_2um, surface_m7_nnunet, dinovol, fiber models).
Community: Discord, scrollprize.substack.com.

## Data we can reach from this Mac (measured)

Bucket `s3://vesuvius-challenge-open-data/`, anonymous access, CC-BY-NC 4.0.
47 objects at the top level, one per scroll or fragment (PHerc0139, PHercParis4, ...); fragment folders in this bucket hold only photos.

Scroll PHerc0139, volume `20250728140407-9.362um-1.2m-113keV-masked.zarr`:
OME-Zarr, 6 pyramid levels, uint8, 128^3 chunks.
Level 0 is 20,974 x 6,621 x 6,621 voxels (about 920 GB); level 4 is 1,311 x 414 x 414.
One full slice at level 2 streamed in 12 s.

Segment `PHerc0139/segments/20250108000000-w025_2025010863/` (133.5 GB in total):

| part | size | use here |
|---|---|---|
| `mesh/*-9.362um.tifxyz` (x.tif, y.tif, z.tif, meta.json) | 1.5 MB | the sheet: a grid of 3D points, stored at scale 0.05 |
| `surface-volumes/9.362um-...zarr` | 1.6 GB | the flattened sheet: 28 layers x 7,280 x 6,800, 6 levels |
| `surface-volumes/2.399um-...zarr` | 91 GB | too big for this laptop |
| `ink-detection/*2.399um*.tif` | 68 MB | the official ink prediction, 28,320 x 26,480 (3.89x the 9 um canvas) |
| `ink-detection/downsampled/*.jpg` | 1.3 MB | previews; one line of Greek letters is visible near the top |

The 2023 Kaggle competition `vesuvius-challenge-ink-detection` has 3 fragments with real infrared ink labels.
Downloading returns 403 until the account accepts that competition's rules (one click on its rules page).

## What the notebooks do

| notebook | data | lesson |
|---|---|---|
| 00 the problem | PHerc0139 low pyramid levels | the pipeline; cross-sections show the spiral |
| 01 volumes | one 3D crop | OME-Zarr pyramids and chunks; air vs papyrus; three orthogonal views |
| 02 flattening | segment w025 mesh + scroll crop | virtual unwrapping by hand, checked against the official surface volume |
| 03 ink detection | w025 surface volume + official ink map | a small 2.5D CNN trained on pseudo-labels, spatial train/val split |
| 04 self-supervised | w025 surface volume + official ink map | masked-reconstruction pretraining, then the 03 CNN with 2-100% of the labels |
| 05 domain shift (next) | 2023 Kaggle fragments | train on one fragment, test on another |
