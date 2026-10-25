"""Small helpers shared by the notebooks: stream the open Vesuvius data from S3 and cache what we read."""
import hashlib
import json
from pathlib import Path

import numpy as np
import s3fs
import tifffile
import zarr

BUCKET = "vesuvius-challenge-open-data"
SCROLL = "PHerc0139"
VOLUME = "20250728140407-9.362um-1.2m-113keV-masked.zarr"     # 9.362 um, the smallest full-scroll scan
SEGMENT = "20250108000000-w025_2025010863"
SEG_SURFACE = "surface-volumes/9.362um-1.2m-113keV-volume-20250728140407.zarr"
SEG_MESH = "mesh/20250108000000-on-20250728140407-9.362um.tifxyz"
CACHE = Path(__file__).resolve().parent / "data" / "cache"
FS = s3fs.S3FileSystem(anon=True)


def _open(path):
    return zarr.open(s3fs.S3Map(f"{BUCKET}/{path}", s3=FS), mode="r")


def scroll_volume(level=0, scroll=SCROLL, volume=VOLUME):
    """One pyramid level of a scroll scan, axes (z, y, x), uint8. Level 0 is full resolution; each level halves every axis."""
    return _open(f"{scroll}/volumes/{volume}")[str(level)]


def segment_surface(level=0, scroll=SCROLL, segment=SEGMENT):
    """The flattened sheet of one segment: (layer, v, u), uint8. Layer 0..27 runs along the surface normal."""
    return _open(f"{scroll}/segments/{segment}/{SEG_SURFACE}")[str(level)]


def segment_mesh(scroll=SCROLL, segment=SEGMENT):
    """The tifxyz mesh: three float grids x, y, z (scroll voxel coordinates at level 0) and its meta.json.
    Grid point (i, j) sits at canvas pixel (i / scale, j / scale); unset points are -1."""
    root = f"{BUCKET}/{scroll}/segments/{segment}/{SEG_MESH}"
    xyz = [cached(f"mesh_{segment}_{c}", lambda c=c: tifffile.imread(FS.open(f"{root}/{c}.tif", "rb"))) for c in "xyz"]
    meta = json.loads(FS.cat(f"{root}/meta.json"))
    return xyz[0], xyz[1], xyz[2], meta


def cached(key, fn):
    """np.save the result of fn() under data/cache/<key>.npy, and reuse it on the next call."""
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / (hashlib.sha1(key.encode()).hexdigest()[:16] + ".npy")
    if f.exists():
        return np.load(f)
    a = np.asarray(fn())
    np.save(f, a)
    return a


def crop(arr, name, *slices):
    """arr[slices], streamed once and then read from the local cache."""
    return cached(f"{name}{slices}", lambda: arr[slices])


INK_KEY = (f"{SCROLL}/segments/{SEGMENT}/ink-detection/PHerc0139-20250108000000-2.399um-0.22m-78keV-volume-20260102150214-"
           "20260417190342-new_canon_autoresearch_recipe-tile256-stride128.tif")


def fetch_ink(path=Path(__file__).resolve().parent / "data" / "w025_ink_2um.tif"):
    """The official 2.4 um ink prediction for segment w025 (68 MB), downloaded once."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        FS.get(f"{BUCKET}/{INK_KEY}", str(path))
    return path
