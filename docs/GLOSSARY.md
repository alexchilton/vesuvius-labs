# Glossary for newcomers

Every term used in the notebooks, in plain words.
"NB" says which notebook uses it most.

## The scrolls

| term | plain meaning | NB |
|---|---|---|
| **Herculaneum** | a Roman town next to Pompeii, buried by Vesuvius in 79 AD | 00 |
| **Villa of the Papyri** | the villa at Herculaneum where the scroll library was found; the only intact library from the ancient world | 00 |
| **papyrus** | ancient paper, made from strips of a reed pressed together | 00 |
| **scroll** | a long papyrus sheet rolled up; the ones here were carbonised and are still closed | 00 |
| **carbonised** | turned to charcoal by heat without burning away; the scroll kept its shape but became brittle and black | 00 |
| **PHerc. 139, PHerc. Paris. 4** | inventory names: "PHerc." means Herculaneum papyrus, then its number (Paris. 4 is kept in Paris) | 00 |
| **fragment** | a piece of a scroll that was opened (and mostly destroyed) in the past; its writing can be photographed, which gives real ink labels | 03 |
| **recto / verso** | the two faces of the sheet; the recto is the inside face, where the text was written | 01 |
| **fibres** | the strands of the papyrus reed; they run in two directions at right angles, like the threads of a woven cloth | 01 |
| **kollema / kollesis** | a single sheet of papyrus / the glued joint where two sheets overlap to make the long roll | 01 |
| **umbilicus** | the central axis the scroll was rolled around | 01, 05 |
| **wrap (winding)** | one full turn of the spiral; a scroll has dozens to over a hundred | 00, 05 |
| **column** | a block of text; a scroll is written in columns side by side, like the pages of a book laid out in a row | 00 |
| **crackle** | the cracked, mud-like texture of dried ink on the papyrus; the main visual clue that ink is there | 03 |
| **Philodemus** | an Epicurean philosopher; the first scroll read (2023) is probably his | 00 |

## Scanning

| term | plain meaning | NB |
|---|---|---|
| **CT (computed tomography)** | take X-ray pictures from hundreds of angles, then compute what the inside looks like in 3D; the same idea as a hospital scan | 00 |
| **synchrotron** | a particle accelerator that makes an extremely bright, narrow X-ray beam (Diamond in the UK, ESRF in France); far sharper than a hospital scanner | 00 |
| **absorption contrast** | how much X-ray a material stops; ink and papyrus are both carbon and stop almost the same amount, which is the core difficulty | 00 |
| **phase contrast** | also recording how the X-ray beam *bends* at edges; makes thin layers like ink stand out, but adds bright and dark halos at boundaries | 00, 01 |
| **voxel** | a 3D pixel: one small cube of the scan with one brightness value | 00 |
| **voxel size (9.4 µm, 2.4 µm)** | the side length of one voxel; 9.4 µm is about a tenth of a hair's width; smaller is sharper but the data grows with the cube | 00 |
| **keV** | the energy of the X-rays used; different energies show materials differently | 00 |
| **reconstruction** | the computation that turns the 2D X-ray pictures into the 3D voxel grid | 00 |

## Data and storage

| term | plain meaning | NB |
|---|---|---|
| **S3 bucket** | an online folder (Amazon storage) where the challenge publishes all its data; anyone can read it | 00 |
| **OME-Zarr** | a file format for huge images, stored as many small files ("chunks") so you can read just the piece you need | 01 |
| **pyramid / level** | the same scan stored at several sizes: level 0 is full resolution, level 1 is half in every direction, and so on; like map zoom levels | 00, 01 |
| **chunk** | one small block of the scan (here 128 × 128 × 128 voxels) stored as one file; reading anything inside it costs the whole block | 01 |
| **uint8** | a number from 0 to 255; each voxel is stored as one | 01 |
| **masked volume** | a scan where everything outside the scroll is set to 0 | 01 |
| **cache** | a local copy of what you already downloaded, so the second read is instant | 00 |

## Geometry: finding and flattening the sheet

| term | plain meaning | NB |
|---|---|---|
| **segmentation** | finding the papyrus sheet inside the 3D scan; "a segment" is one traced piece of sheet | 00, 02 |
| **surface prediction** | a model that marks, for every voxel, whether it is papyrus surface; the first step before tracing | 01 |
| **mesh** | a surface described by points connected into a net, like a fishing net laid along the sheet | 02 |
| **tifxyz** | the challenge's mesh format: three images x, y, z, where pixel (i, j) holds the 3D position of one point of the sheet | 02 |
| **tracing** | following one sheet through the scan to build its mesh; done in **VC3D** with tools like **GrowPatch** | 02 |
| **sheet switch** | a tracing error where the mesh jumps onto the neighbouring wrap; the text then comes from the wrong place | 02, 05 |
| **normal** | the direction pointing straight out of the sheet, at right angles to it | 01, 02 |
| **structure tensor** | a classical formula that finds the sheet's normal from how brightness changes around each voxel, without any training | 01 |
| **coherence** | how clearly a neighbourhood looks like a single sheet (1) rather than noise (0) | 01 |
| **flattening / unwrapping** | turning the curved sheet into a flat image; like peeling an orange and pressing the peel flat | 02 |
| **isometric** | a flattening that keeps every distance true to scale (no stretch) | 02 |
| **surface volume** | the flattened sheet as a stack of layers (typically 28 to 65) parallel to the surface; the input for ink detection | 02, 03 |
| **rendering** | sampling the scan along the mesh and its normals to produce the surface volume | 02 |
| **spiral fit** | fitting the whole scroll at once as one ideal spiral that was then deformed; the challenge's current approach | 05 |
| **winding number / winding constraint** | which wrap a point is on / a hint such as "these two points are on the same wrap" or "this point is one wrap further out" | 05 |

## Machine learning

| term | plain meaning | NB |
|---|---|---|
| **label** | the right answer a model learns from (here: "ink" or "no ink" per pixel) | 03 |
| **pseudo-label** | a label produced by another model instead of a human; cheap, but it carries that model's mistakes | 03 |
| **CNN** | a neural network that slides small filters over an image to find patterns | 03 |
| **2.5D model** | a model that reads a stack of layers (3D) but outputs a flat map (2D) | 03 |
| **U-Net / nnU-Net** | a network shaped like a U that shrinks the image to understand context, then grows it back to mark every pixel; nnU-Net is a self-configuring version used for medical 3D scans | 01, 03 |
| **held-out / validation** | data the model never trains on, kept back to test whether it generalises | 03 |
| **spatial split** | holding out a whole *region* rather than random patches, because neighbouring patches overlap and would leak answers | 03 |
| **generalisation** | working on new data, here a scroll the model never saw; the hardest open problem in ink detection | 03 |
| **self-supervised learning (SSL)** | learning from unlabelled data with a made-up task, e.g. hiding part of an image and predicting it | 04 |
| **masked image modelling (MAE, SimMIM)** | the "hide blocks and reconstruct them" kind of SSL | 04 |
| **SimCLR / DINO / dinovol** | other SSL methods: match two views of the same patch; dinovol is the challenge's 3D version of DINO | 04 |
| **fine-tuning** | training a pretrained model a little more on the real task | 04 |
| **ensemble** | averaging several models' predictions; usually more accurate than any single one | 01 |
| **test-time augmentation (TTA)** | predicting on flipped copies of the input and averaging | 01 |
| **seed** | the number that fixes the randomness of a run; two seeds give two slightly different results | 04 |

## Flows and transport (notebook 05)

| term | plain meaning | NB |
|---|---|---|
| **deformation** | a map that moves every point of a shape somewhere else; a crush is one | 05 |
| **velocity field** | an arrow at every point saying which way and how fast it moves | 05 |
| **flow** | what you get by following the arrows for a fixed time; like dropping leaves on a river and seeing where they end up | 05 |
| **diffeomorphism** | a smooth map that can be undone exactly; it never tears or folds | 05 |
| **Jacobian determinant** | how much a map locally scales area; below zero means the map folded the sheet over itself | 05 |
| **scaling and squaring** | a fast way to compute a flow: take a tiny step, then compose it with itself repeatedly | 05 |
| **optimal transport (OT)** | matching two clouds of points so the total movement is as small as possible | 05 |
| **flow matching (OT-CFM)** | training a velocity field to carry one point cloud onto another, using OT pairs as targets | 05 |
| **amortised** | trained once, then fast on every new case, instead of solving each case from scratch | 05 |
| **domain gap** | the difference between training data (here synthetic) and real data, which can make a model fail | 05 |

## Competition and metrics

| term | plain meaning | NB |
|---|---|---|
| **correlation** | how closely two maps rise and fall together, from −1 to 1; used here to compare a prediction with a label | 02, 03 |
| **Dice** | overlap between a predicted mask and the true mask, from 0 to 1 | 01 |
| **topology** | the "shape type" of a mask: how many pieces, holes and tunnels it has; a bridge between two wraps changes it | 01 |
| **Betti numbers** | counts of pieces (0), tunnels (1) and cavities (2) | 01 |
| **VOI (variation of information)** | how differently two masks split space into connected pieces; high when sheets merge or split | 01 |
| **First Letters / Grand Prize / progress prizes** | the challenge's prizes: find letters in a new scroll / unroll a whole scroll / monthly awards for tools and documentation | 00 |
