# Notes from the five tutorials at the top of scrollprize.org/tutorial

Read 2026-10-25.
Summaries of each page as fetched, with the site's own terms and numbers.
Used to expand notebooks 00 to 05.

## 1. Virtual Unwrapping with VC3D (scrollprize.org/tutorial_VC3D)

VC3D (Volume Cartographer 3D) is the challenge's desktop tool for segmentation.
Workflow: load a volume from the open data catalogue, overlay the pre-computed surface predictions, grow a patch with **GrowPatch**, add winding and fibre annotations, export.
Every segment VC3D makes, whatever its size, is stored in the same `tifxyz` format (the one notebook 02 reads).
GrowPatch size grows with its iteration count; one growth step is about 20 voxels; directional growth uses keys 1 to 5.
Fibre annotation needs **lasagna normals** (predicted surface normals stored as zarr volumes).
Quality check from the page: a good segment follows the sheet in the cross-section views, and shows the horizontal and vertical fibres clearly in the flattened view.
The page was last updated 2026-07-12; pre-built Docker containers are tagged by build date.

## 2. Spiral Fitting (scrollprize.org/tutorial_spiral)

The ideal scroll is an Archimedean spiral extruded along the axis: `r(θ) = (ω / 2π) θ`, one turn per spacing ω.
A **diffeomorphic** deformation maps the ideal spiral into the crushed scan: global scale and shear, then the flow of a velocity field `dφ_t(x)/dt = u(φ_t(x))` integrated with Runge–Kutta steps, then local scaling of the gaps between windings.
Inputs, none of which needs full coverage: tifxyz patches, same-winding point collections (fibres, strips), relative-winding annotations (two points exactly k windings apart), absolute-winding annotations, lasagna normals, gradient magnitude, skeletonised surface-prediction tracks, the umbilicus (the central axis as a function of z), and an optional outer-shell mesh.
All parameters are fitted jointly with plain Adam on a weighted sum of losses (the `loss_weight_*` entries of `default_config`).
Output: the fitted spiral sampled on a regular (θ, z) grid, pushed through the deformation, one tifxyz per winding (`wNNN`, and `wNNN_spliced` with verified patches spliced in).
Example: 130 fitted windings for PHerc. Paris. 4.
Failure modes named: **sheet switches** (the fit leaves winding A for its neighbour B) and **wandering** (it oscillates along the normal and touches neighbours); patches fix both most reliably.
Hard case: shear zones, where the papyrus broke and slid along an axis.

## 3. Winding Constraints (scrollprize.org/open_problems/winding_annotations)

Sparse evidence whose relation to the windings is known: points, lines, fibres or patches.
Three kinds: **same-winding** (these points are on one wrap), **relative-winding** (this point is k wraps outward of that one), **absolute-winding** (this point is on wrap number n).
Any method may produce them: hand drawing, fibre or mesh inference, classical vision, learned proposals.
Untried ideas the page lists: hand-drawn surface paths, ink as same-winding evidence, gaps between text columns, kolleseis (sheet joints).
A good generator is accurate with a measurable confidence, fast enough for a whole volume, easy to verify, and compatible with the spiral fit.

## 4. Ink Detection (scrollprize.org/tutorial5)

Label sources: detached fragments photographed in infrared (1000 nm) and aligned to the surface volume; the **crackle** texture ("like cracked mud, raised slightly above the papyrus"); and **iterative pseudo-labelling** (run a model, a human labels clearly visible strokes, retrain), which scaled from single letters to whole scrolls such as PHerc. 1667.
2.5D: surface-volume patches `[64, 256, 256]` (z, y, x), a U-Net (`vesuvius_unet`), depth collapsed with `z_projection_mode: "max"` (also `mean`, `logsumexp`, `learned_mlp`).
Native 3D: `full_3d` on crops from the raw volume, `[80, 128, 128]`, labels projected ±8 voxels along the normal; `full_3d_single_wrap` adds a channel marking which wrap a voxel belongs to.
9 µm cross-scroll model (`huggingface.co/scrollprize/ink_9um`): a small 3D stem pools a 17-slice patch to a 2D map, then a 2D U-Net; trained on aligned labels from four scrolls (PHerc. 0139, 1667, Paris. 4, 0814) with fixed per-scroll batch quotas 29/22/11/2 of 64, 78,125 iterations; two seeds, seven checkpoints each.
Training: lr 0.01, fp16 with overflow checks, patch overlap 0.5, at least 5% labelled coverage per patch; held-out validation regions excluded from the loss.
Against label leakage: patches smaller than a letter (so the model cannot learn to draw letters), conservative labels, held-out masks, papyrologists review the final reading.
Cross-scroll: Paris. 4 shows crisp crackle; other scrolls show "more subtle and heterogeneous" signals.

## 5. Scroll Anatomy (github.com/ScrollPrize/villa/blob/main/SCROLL_ANATOMY.md)

A papyrus sheet is two perpendicular layers of pith strips pressed together; the two faces are **recto** and **verso**.
A fibre stays within its layer and never passes from recto to verso.
Sheets (**kollemata**) are glued end to end and rolled; a **kollesis** is the overlapping joint, where two sheets overlap within the same winding (another sheet there is not another winding).
The roll winds around the **umbilicus** axis.
Recto fibres run lengthwise in the unrolled roll (circumferential when rolled); verso fibres run across its height (axial when rolled).
Carbonisation caused delamination, folds, compression, tears and breakage; these bend fibres without changing which winding they belong to.
CT intensity is a reconstructed value, not a material label or a density; phase-contrast fringes and halos appear at boundaries.
Ink may show as a surface deposit, raised texture, crackle, or subtler contrast changes.
The distance between two surfaces in the scan says nothing about how far apart they are in the unrolled scroll.
