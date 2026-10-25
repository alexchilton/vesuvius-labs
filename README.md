# vesuvius-labs

Tutorial notebooks for the [Vesuvius Challenge](https://scrollprize.org): reading carbonised Herculaneum scrolls from
X-ray CT scans, step by step, on a laptop.

Everything streams small pieces of the challenge's public data (`s3://vesuvius-challenge-open-data/`, anonymous,
CC-BY-NC 4.0); no account or bulk download is needed. Pieces read once are cached under `data/cache/`.

| notebook | what you do | data read |
|---|---|---|
| [00 the problem](notebooks/00_the_problem.ipynb) | the pipeline as a diagram; cross-sections of a real scroll showing the spiral | a few slices of PHerc. 139 |
| [01 volumes](notebooks/01_volumes.ipynb) | OME-Zarr pyramids and chunks, air vs papyrus, sheet normals from the structure tensor | one 256³ block |
| [02 flattening](notebooks/02_flattening.ipynb) | **virtual unwrapping by hand**: render a traced sheet from its mesh and the raw scan, then match the official rendering (correlation 0.93) | a 1.5 MB mesh, one ~35 MB crop |
| [03 ink detection](notebooks/03_ink_detection.ipynb) | a small 2.5D CNN on 16 surface layers, trained on the official ink prediction as a pseudo-label, validated on a spatially held-out strip | the segment's surface volume (350 MB) + a 68 MB ink map |
| [04 self-supervised](notebooks/04_self_supervised.ipynb) | **few labels**: masked-reconstruction pretraining on unlabelled surface tiles, then the 03 CNN with 2%, 10%, 50% and 100% of the labels; pretraining is within noise (−0.068 to +0.022) | the same files as 03 |
| [05 uncrushing with flows](notebooks/05_uncrushing_flows.ipynb) | crush synthetic spirals with a velocity-field **flow** (invertible, never folds), train a U-Net to undo it, compare with unpaired OT flow matching, then test on a real crushed cross-section; the real test fails (roundness 0.196 → 0.233), and the notebook says why | one level-4 slice + the w025 mesh |

Background, prizes, the ten open problems and the data layout: [docs/OVERVIEW.md](docs/OVERVIEW.md).
Notes from the five tutorials at the top of scrollprize.org/tutorial (VC3D, spiral fitting, winding constraints, ink detection, scroll anatomy): [docs/WEBSITE_NOTES.md](docs/WEBSITE_NOTES.md).

## Run

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
mkdir -p data && aws s3 cp --no-sign-request \
  "s3://vesuvius-challenge-open-data/PHerc0139/segments/20250108000000-w025_2025010863/ink-detection/PHerc0139-20250108000000-2.399um-0.22m-78keV-volume-20260102150214-20260417190342-new_canon_autoresearch_recipe-tile256-stride128.tif" \
  data/w025_ink_2um.tif          # notebooks 03 and 04 need this file
.venv/bin/jupyter lab notebooks/
```

Notebooks 03 and 04 train on Apple MPS if available, else CPU (a few minutes either way).

## Next

- 06 domain shift: train on one 2023 fragment, test on another (public mirror `alvarocasaszan/vesuvius-fragments` on Kaggle).
- 07 surface prediction: compare the structure tensor with the challenge's nnU-Net surface model.

## Credits

Data, meshes, surface volumes and the 2.4 µm ink prediction: the Vesuvius Challenge and its contributors
(scrollprize.org; code at github.com/ScrollPrize/villa). These notebooks only read their public releases.
