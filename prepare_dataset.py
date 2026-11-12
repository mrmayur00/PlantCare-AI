"""
prepare_dataset.py - Build the training / test image folders for PlantCare AI.

Reads dataset_sources.json, which maps every model class to one or more
folders of labelled leaf photos, and writes:

    <out>/train/<class>/*.jpg     used by train_model.py (it splits off validation)
    <out>/test/<class>/*.jpg      independent test images, NEVER used in training
    <out>/manifest.json           counts, sources and licences for your report

What it does for every class:
    1. Reads all JPG / PNG / JFIF / WEBP images from the listed folders.
    2. Shrinks them (longest side 256 px) to keep training fast.
    3. Finds near-duplicate photos with a perceptual hash (dHash). Many public
       datasets contain the same leaf more than once (copies, small crops).
    4. Splits by duplicate GROUP, so a photo and its near-copy always land on
       the same side. Otherwise the test score would be inflated.

Usage:
    python prepare_dataset.py --config dataset_sources.json --out data

Adding a new plant (for example Jamun): download its dataset, then add an
entry to dataset_sources.json such as

    "Jamun___Anthracnose": {"folders": ["${PLANTCARE_SOURCES}/jamun/Anthracnose"]}

Optional keys per class: "max_source" (images read), "max_train",
"recursive" (search sub-folders), "min_green" (drop photos that are not
mostly green plants, used for wheat).

and run this script and train_model.py again (see README, section 3).
"""
import argparse
import json
import os
import random
import shutil

import numpy as np
from PIL import Image, ImageOps

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".jfif", ".webp")
HERE = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    p = argparse.ArgumentParser(description="Prepare PlantCare AI train/test folders")
    p.add_argument("--config", default=os.path.join(HERE, "dataset_sources.json"))
    p.add_argument("--out", default=os.path.join(HERE, "data"))
    p.add_argument("--size", type=int, default=256, help="longest side of saved images")
    p.add_argument("--test-fraction", type=float, default=0.15)
    p.add_argument("--min-test", type=int, default=25, help="minimum test images per class")
    p.add_argument("--max-train", type=int, default=400, help="cap on training images per class")
    p.add_argument("--dup-distance", type=int, default=6,
                   help="max dHash bit difference treated as a near-duplicate")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def dhash(image, size=8):
    """64-bit difference hash: robust to resizing and recompression."""
    small = np.asarray(image.convert("L").resize((size + 1, size), Image.BILINEAR), dtype=np.int16)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return np.packbits(bits)


def group_duplicates(hashes, max_distance):
    """Union-find over images whose hashes differ by <= max_distance bits."""
    n = len(hashes)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    if n:
        mat = np.unpackbits(np.stack(hashes), axis=1).astype(np.uint8)  # n x 64
        for i in range(n):
            dist = np.count_nonzero(mat[i + 1:] != mat[i], axis=1)
            for j in np.nonzero(dist <= max_distance)[0]:
                a, b = find(i), find(i + 1 + j)
                if a != b:
                    parent[b] = a
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def list_images(folders, recursive=False):
    files = []
    for folder in folders:
        if not os.path.isdir(folder):
            raise SystemExit(f"Folder not found: {folder}")
        walker = os.walk(folder) if recursive else [(folder, [], os.listdir(folder))]
        for root, _, names in walker:
            files += [os.path.join(root, n) for n in sorted(names) if n.lower().endswith(IMAGE_EXTS)]
    return files


def green_fraction(image):
    """Share of clearly green pixels; used to keep plant photos and drop e.g. soil or ripe ears."""
    small = image.copy()
    small.thumbnail((128, 128))
    hsv = np.asarray(small.convert("HSV"), dtype=np.float32)
    h, s, v = hsv[..., 0] * 360 / 255, hsv[..., 1] / 255, hsv[..., 2] / 255
    return float(((h >= 60) & (h <= 160) & (s >= 0.2) & (v >= 0.15)).mean())


def load_small(path, size):
    try:
        im = Image.open(path)
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((size, size), Image.BILINEAR)
        return im
    except Exception:  # noqa: BLE001 - unreadable / corrupted files are skipped
        return None


def main():
    args = parse_args()
    with open(args.config, encoding="utf-8") as f:
        config = json.load(f)
    rng = random.Random(args.seed)
    base = os.path.dirname(os.path.abspath(args.config))

    for split in ("train", "test"):
        shutil.rmtree(os.path.join(args.out, split), ignore_errors=True)

    manifest = {"sources": config.get("_sources", {}), "classes": {}}
    for class_name, spec in sorted(config.items()):
        if class_name.startswith("_"):
            continue
        folders = [os.path.join(base, os.path.expandvars(p)) for p in spec["folders"]]
        files = list_images(folders, spec.get("recursive", False))
        rng.shuffle(files)
        if spec.get("max_source"):
            files = files[: spec["max_source"]]

        images, hashes, skipped = [], [], 0
        for path in files:
            im = load_small(path, args.size)
            if im is None or min(im.size) < 32:
                skipped += 1
                continue
            # optional filter, e.g. wheat "healthy" photos must show green plants
            if spec.get("min_green") and green_fraction(Image.open(path).convert("RGB")) < spec["min_green"]:
                skipped += 1
                continue
            images.append(im)
            hashes.append(dhash(im))

        groups = group_duplicates(hashes, args.dup_distance)
        rng.shuffle(groups)
        n_test_target = max(args.min_test, int(round(len(images) * args.test_fraction)))
        test_idx, train_idx = [], []
        for g in groups:
            if len(test_idx) < n_test_target:
                # test set: one photo per group, so every test image is independent
                test_idx.append(g[0])
            else:
                # training set: other shots of the same leaf are useful variety,
                # but exact copies (almost identical hash) are dropped
                kept = []
                for i in g:
                    if all(np.count_nonzero(np.unpackbits(hashes[i] ^ hashes[k])) > 1 for k in kept):
                        kept.append(i)
                train_idx += kept
        rng.shuffle(train_idx)
        train_idx = train_idx[: spec.get("max_train", args.max_train)]

        for split, idx in (("test", test_idx), ("train", train_idx)):
            out_dir = os.path.join(args.out, split, class_name)
            os.makedirs(out_dir, exist_ok=True)
            for k, i in enumerate(idx):
                images[i].save(os.path.join(out_dir, f"{k:04d}.jpg"), "JPEG", quality=90)

        manifest["classes"][class_name] = {
            "source": spec.get("source", ""),
            "read": len(files), "unreadable": skipped,
            "unique_after_dedup": len(groups),
            "train": len(train_idx), "test": len(test_idx),
        }
        print(f"{class_name:45s} read {len(files):5d}  unique {len(groups):5d}  "
              f"train {len(train_idx):4d}  test {len(test_idx):3d}")

    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote {args.out}/train, {args.out}/test and manifest.json")


if __name__ == "__main__":
    main()
