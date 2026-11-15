"""
train_species.py - Train the stage-1 PLANT SPECIES classifier.

Expected data (one folder per class; names must match exactly):

    data/species/
        Tulsi/  Mango/  Jamun/  Custard_Apple/  Sapodilla/
        Curry_Leaf/  Rice/  Wheat/  Rose/  Hibiscus/
        Unsupported/      <- other plants, non-leaf objects, soil, hands, walls ...

Healthy AND diseased leaves of a species go in that species' folder (the
species model must not care about disease). Mix lighting, backgrounds, angles
and, above all, include photos from the same kind of phone/camera users will use.

What this script does
  1. Loads every image once with the SAME preprocessing the app uses.
  2. Finds near-duplicates (perceptual hash) and splits by duplicate GROUP into
     train / validation / test, so copies of one leaf never sit on both sides.
  3. Trains MobileNetV2 (ImageNet transfer learning) with augmentation.
  4. Calibrates the accept thresholds (confidence, top-2 margin) on the
     VALIDATION set so that accepted answers reach --target-precision.
  5. Reports honest numbers on the untouched TEST set: accuracy, per-class
     precision/recall/F1, confusion matrix, share of "Unsupported" photos
     wrongly accepted. Optional --unseen-dir tests extra never-seen plants.
  6. Writes models/species_model.keras, models/species_info.json and
     models/species_report.json.

Usage (Colab GPU recommended, roughly 20-40 minutes):
    python train_species.py --data-dir data/species
    python train_species.py --data-dir data/species --unseen-dir data/unseen_plants
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
from PIL import Image, ImageOps

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from backend.species import SPECIES, UNSUPPORTED_LABEL, to_array  # noqa: E402
from prepare_dataset import dhash, group_duplicates  # noqa: E402

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".jfif", ".webp")


def parse_args():
    p = argparse.ArgumentParser(description="Train the PlantCare AI species classifier")
    p.add_argument("--data-dir", default=os.path.join(HERE, "data", "species"))
    p.add_argument("--output-dir", default=os.path.join(HERE, "models"))
    p.add_argument("--unseen-dir", default="", help="folder of OTHER plants never used in training (rejection test)")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--fine-tune-epochs", type=int, default=6)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--weights", default="imagenet", choices=["imagenet", "none"])
    p.add_argument("--max-per-class", type=int, default=700)
    p.add_argument("--min-images", type=int, default=60, help="refuse to train a class with fewer images")
    p.add_argument("--target-precision", type=float, default=0.97,
                   help="share of ACCEPTED answers that must be right on validation data")
    p.add_argument("--dup-distance", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def list_images(folder):
    return sorted(os.path.join(r, f) for r, _, fs in os.walk(folder) for f in fs
                  if f.lower().endswith(IMAGE_EXTS))


def load_array(path, size):
    try:
        return to_array(ImageOps.exif_transpose(Image.open(path)), size)
    except Exception:  # noqa: BLE001 - unreadable files are skipped
        return None


def load_and_split(args):
    """Return class_names and (X, y) for train / val / test, split by duplicate group."""
    present = sorted(d for d in os.listdir(args.data_dir)
                     if os.path.isdir(os.path.join(args.data_dir, d)) and not d.startswith("."))
    unknown = [d for d in present if d not in SPECIES and d != UNSUPPORTED_LABEL]
    if unknown:
        raise SystemExit(f"Unknown class folders: {unknown}. Allowed: {sorted(SPECIES)} + '{UNSUPPORTED_LABEL}'")
    missing = [c for c in list(SPECIES) + [UNSUPPORTED_LABEL] if c not in present]
    if missing:
        print(f"WARNING: no folder for {missing}. These will NOT be supported by the trained model.")
    if UNSUPPORTED_LABEL not in present:
        raise SystemExit(f"A '{UNSUPPORTED_LABEL}' folder is required (other leaves + non-leaf photos).")

    rng = random.Random(args.seed)
    class_names = present
    split = {k: ([], []) for k in ("train", "val", "test")}
    counts = {}
    for idx, name in enumerate(class_names):
        files = list_images(os.path.join(args.data_dir, name))
        rng.shuffle(files)
        files = files[:args.max_per_class]
        arrays = [a for a in (load_array(f, args.image_size) for f in files) if a is not None]
        if len(arrays) < args.min_images:
            raise SystemExit(f"Class '{name}' has only {len(arrays)} readable images "
                             f"(need >= {args.min_images}). Add more or remove the folder.")
        hashes = [dhash(Image.fromarray(a)) for a in arrays]
        groups = group_duplicates(hashes, args.dup_distance)
        rng.shuffle(groups)
        n = len(arrays)
        t_val, t_test = int(round(n * 0.15)), int(round(n * 0.15))
        buckets = {"train": [], "val": [], "test": []}
        for g in groups:  # whole duplicate groups stay together
            if len(buckets["test"]) < t_test:
                buckets["test"] += g
            elif len(buckets["val"]) < t_val:
                buckets["val"] += g
            else:
                buckets["train"] += g
        for k, ids in buckets.items():
            split[k][0].extend(arrays[i] for i in ids)
            split[k][1].extend([idx] * len(ids))
        counts[name] = {k: len(v) for k, v in buckets.items()}
        print(f"  {name:15s} {n:5d} images, {len(groups):5d} unique -> "
              + ", ".join(f"{k} {len(v)}" for k, v in buckets.items()))
    out = {k: (np.stack(v[0]), np.array(v[1], dtype=np.int64)) for k, v in split.items()}
    return class_names, out, counts


def build_model(num_classes, size, weights, alpha):
    import tensorflow as tf
    L = tf.keras.layers
    base = tf.keras.applications.MobileNetV2(input_shape=(size, size, 3), include_top=False, alpha=alpha,
                                             weights=None if weights == "none" else "imagenet")
    base.trainable = weights == "none"
    inputs = tf.keras.Input(shape=(size, size, 3), name="image")
    x = L.RandomFlip("horizontal_and_vertical")(inputs)
    x = L.RandomRotation(0.15)(x)
    x = L.RandomZoom(0.2)(x)
    x = L.RandomBrightness(0.25, value_range=(0, 255))(x)
    x = L.RandomContrast(0.25)(x)
    x = L.Rescaling(1.0 / 127.5, offset=-1.0, name="preprocess")(x)   # raw 0-255 in, like the app
    x = base(x, training=False if weights != "none" else None)
    x = L.GlobalAveragePooling2D()(x)
    x = L.Dropout(0.3)(x)
    outputs = L.Dense(num_classes, activation="softmax", name="predictions")(x)
    return tf.keras.Model(inputs, outputs, name="plantcare_species"), base


def top2(probs):
    srt = np.sort(probs, axis=1)
    return probs.argmax(1), srt[:, -1], srt[:, -1] - srt[:, -2]


def accepted_mask(probs, unsup_idx, conf_t, margin_t):
    pred, conf, margin = top2(probs)
    return (pred != unsup_idx) & (conf >= conf_t) & (margin >= margin_t)


def calibrate(probs, y, unsup_idx, target):
    """Pick (confidence, margin) giving max coverage while accepted answers stay >= target precision."""
    pred = probs.argmax(1)
    supported = y != unsup_idx
    best = None
    for c in np.arange(0.50, 0.996, 0.01):
        for m in (0.0, 0.1, 0.2, 0.3, 0.4):
            acc = accepted_mask(probs, unsup_idx, c, m)
            if acc.sum() == 0:
                continue
            precision = float((pred[acc] == y[acc]).mean())
            coverage = float((acc & supported & (pred == y)).sum() / max(supported.sum(), 1))
            if precision >= target and (best is None or coverage > best["coverage"]):
                best = {"confidence": round(float(c), 2), "margin": m, "precision": precision, "coverage": coverage}
    if best is None:
        print(f"WARNING: no threshold reaches {target:.0%} precision on validation data; "
              "using strict fallback (0.95 / 0.30). The model needs more/better data.")
        acc = accepted_mask(probs, unsup_idx, 0.95, 0.3)
        best = {"confidence": 0.95, "margin": 0.3, "target_not_reached": True,
                "precision": float((pred[acc] == y[acc]).mean()) if acc.sum() else None,
                "coverage": float((acc & supported & (pred == y)).sum() / max(supported.sum(), 1))}
    return best


def per_class_report(y, pred, class_names):
    k = len(class_names)
    cm = np.zeros((k, k), dtype=int)
    for t, p in zip(y, pred):
        cm[t, p] += 1
    rep = {}
    for i, name in enumerate(class_names):
        tp, fp, fn = cm[i, i], cm[:, i].sum() - cm[i, i], cm[i, :].sum() - cm[i, i]
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        rep[name] = {"precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3),
                     "support": int(cm[i].sum())}
    return cm, rep


def main():
    args = parse_args()
    import tensorflow as tf
    tf.keras.utils.set_random_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Reading {args.data_dir}")
    class_names, data, counts = load_and_split(args)
    unsup_idx = class_names.index(UNSUPPORTED_LABEL)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = data["train"], data["val"], data["test"]
    print(f"\n{len(class_names)} classes | train {len(Xtr)} | val {len(Xva)} | test {len(Xte)}\n")

    # balance classes
    cw = {i: len(ytr) / (len(class_names) * max((ytr == i).sum(), 1)) for i in range(len(class_names))}
    train_ds = (tf.data.Dataset.from_tensor_slices((Xtr.astype("float32"), ytr))
                .shuffle(len(Xtr), seed=args.seed).batch(args.batch_size).prefetch(tf.data.AUTOTUNE))
    val_ds = tf.data.Dataset.from_tensor_slices((Xva.astype("float32"), yva)).batch(args.batch_size)

    model, base = build_model(len(class_names), args.image_size, args.weights, args.alpha)
    model_path = os.path.join(args.output_dir, "species_model.keras")
    cbs = [tf.keras.callbacks.ModelCheckpoint(model_path, monitor="val_accuracy", save_best_only=True),
           tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=4, restore_best_weights=True),
           tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6)]
    start = time.time()
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    print("Phase 1: training the classification head")
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=cbs, class_weight=cw, verbose=2)
    if args.fine_tune_epochs > 0 and args.weights != "none":
        print("\nPhase 2: fine-tuning the top of MobileNetV2")
        base.trainable = True
        for layer in base.layers[:-40]:
            layer.trainable = False
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])
        model.fit(train_ds, validation_data=val_ds, epochs=args.epochs + args.fine_tune_epochs,
                  initial_epoch=args.epochs, callbacks=cbs, class_weight=cw, verbose=2)

    best = tf.keras.models.load_model(model_path, compile=False)
    p_val = best.predict(Xva.astype("float32"), verbose=0)
    p_te = best.predict(Xte.astype("float32"), verbose=0)

    # 4. calibrate on VALIDATION data only
    cal = calibrate(p_val, yva, unsup_idx, args.target_precision)
    print(f"\nCalibrated on validation: min confidence {cal['confidence']}, min margin {cal['margin']} "
          f"-> precision {cal['precision']}, coverage {cal['coverage']:.3f}")

    # 5. honest numbers on TEST data
    pred = p_te.argmax(1)
    cm, rep = per_class_report(yte, pred, class_names)
    acc_mask = accepted_mask(p_te, unsup_idx, cal["confidence"], cal["margin"])
    sup = yte != unsup_idx
    test = {
        "images": int(len(yte)),
        "accuracy": round(float((pred == yte).mean()), 4),
        "per_class": rep,
        "confusion_matrix": {"labels": class_names, "matrix": cm.tolist()},
        "at_thresholds": {
            "accepted_share_of_supported_photos": round(float((acc_mask & sup).sum() / max(sup.sum(), 1)), 4),
            "precision_of_accepted": round(float((pred[acc_mask] == yte[acc_mask]).mean()), 4) if acc_mask.sum() else None,
            "unsupported_photos_wrongly_accepted": int((acc_mask & ~sup).sum()),
            "unsupported_photos_total": int((~sup).sum()),
        },
    }
    if args.unseen_dir and os.path.isdir(args.unseen_dir):
        files = list_images(args.unseen_dir)
        arrs = [a for a in (load_array(f, args.image_size) for f in files) if a is not None]
        if arrs:
            pu = best.predict(np.stack(arrs).astype("float32"), verbose=0)
            wrong = accepted_mask(pu, unsup_idx, cal["confidence"], cal["margin"])
            test["unseen_plants"] = {"photos": len(arrs), "wrongly_accepted": int(wrong.sum())}

    print(f"\nTEST accuracy: {test['accuracy']*100:.1f}% on {len(yte)} images")
    print(f"{'class':15s} {'prec':>6s} {'rec':>6s} {'f1':>6s} {'n':>5s}")
    for name, r in rep.items():
        print(f"{name:15s} {r['precision']:6.2f} {r['recall']:6.2f} {r['f1']:6.2f} {r['support']:5d}")
    print("At the calibrated thresholds:", json.dumps(test["at_thresholds"]))
    if "unseen_plants" in test:
        print("Unseen plants:", test["unseen_plants"])

    info = {
        "class_names": class_names,
        "image_size": args.image_size,
        "input_scale": "0-255",
        "min_confidence": cal["confidence"],
        "min_margin": cal["margin"],
        "calibration": cal,
        "architecture": f"MobileNetV2 (alpha {args.alpha}), weights={args.weights}",
        "images_per_class": counts,
        "training_minutes": round((time.time() - start) / 60, 1),
        "test_metrics": {"accuracy": test["accuracy"], "images": test["images"]},
    }
    with open(os.path.join(args.output_dir, "species_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    with open(os.path.join(args.output_dir, "species_report.json"), "w", encoding="utf-8") as f:
        json.dump(test, f, indent=2)
    print(f"\nSaved {model_path}, species_info.json, species_report.json in {args.output_dir}")


if __name__ == "__main__":
    main()
