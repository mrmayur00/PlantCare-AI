"""
train_model.py - Train the PlantCare AI leaf disease classifier.

Transfer learning with MobileNetV2 (pre-trained on ImageNet) on the
PlantVillage dataset. Produces the two files the app needs:

    models/plant_disease_model.keras   the trained model
    models/model_info.json             class labels (in output order) + input size

Dataset layout expected (one folder per class), for example the
"raw/color" folder of https://github.com/spMohanty/PlantVillage-Dataset:

    raw/color/
        Apple___Apple_scab/        *.JPG
        Apple___Black_rot/
        ...
        Tomato___healthy/

Usage (Google Colab with a GPU is recommended, about 20-30 minutes):

    python train_model.py --data-dir PlantVillage-Dataset/raw/color

Quick test on a laptop CPU (fewer images, fewer epochs):

    python train_model.py --data-dir PlantVillage-Dataset/raw/color --max-per-class 150 --epochs 3 --fine-tune-epochs 0

Preprocessing note: the model contains its own Rescaling layer that maps
pixels from 0-255 to -1..1 (what MobileNetV2 expects). The Flask app
therefore feeds raw 0-255 RGB pixels, exactly as in training.
"""
import argparse
import json
import os
import random
import time

import numpy as np
import tensorflow as tf

IMAGE_EXTS = (".jpg", ".jpeg", ".png")
HERE = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    p = argparse.ArgumentParser(description="Train the PlantCare AI model")
    p.add_argument("--data-dir", required=True, help="folder with one sub-folder per class")
    p.add_argument("--output-dir", default=os.path.join(HERE, "models"))
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=5, help="epochs with the base network frozen")
    p.add_argument("--fine-tune-epochs", type=int, default=3, help="extra epochs with top layers unfrozen")
    p.add_argument("--val-split", type=float, default=0.2)
    p.add_argument("--max-per-class", type=int, default=0, help="limit images per class (0 = all)")
    p.add_argument("--alpha", type=float, default=1.0,
                   help="MobileNetV2 width (1.0 = full size, 0.35 = small and fast)")
    p.add_argument("--weights", default="imagenet", choices=["imagenet", "none"],
                   help="'none' trains from scratch (only if ImageNet weights cannot be downloaded)")
    p.add_argument("--init-from", default="",
                   help="existing .keras model to start from (adds new classes without "
                        "re-learning the old ones from zero; same --alpha/--image-size needed)")
    p.add_argument("--lr", type=float, default=1e-3, help="starting learning rate")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def warm_start(model, base, init_path, class_names):
    """
    Copy what an earlier PlantCare model learned into the new model.

    - The whole MobileNetV2 backbone (all convolution weights) is copied.
    - For classes that existed in the old model, their row of the final
      layer is copied too; new classes start from random weights.
    """
    old = tf.keras.models.load_model(init_path, compile=False)
    old_base = next(l for l in old.layers if isinstance(l, tf.keras.Model))
    base.set_weights(old_base.get_weights())

    info_path = os.path.join(os.path.dirname(init_path), "model_info.json")
    with open(info_path, encoding="utf-8") as f:
        old_classes = json.load(f)["class_names"]
    old_w, old_b = old.get_layer("predictions").get_weights()
    new_w, new_b = model.get_layer("predictions").get_weights()
    reused = 0
    for j, name in enumerate(class_names):
        if name in old_classes:
            i = old_classes.index(name)
            new_w[:, j], new_b[j] = old_w[:, i], old_b[i]
            reused += 1
    model.get_layer("predictions").set_weights([new_w, new_b])
    print(f"Warm start from {init_path}: backbone copied, "
          f"{reused} of {len(class_names)} class outputs reused, "
          f"{len(class_names) - reused} new classes")


def collect_files(data_dir, max_per_class, val_split, seed):
    """List images per class and make a stratified train/validation split."""
    class_names = sorted(
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith(".")
    )
    if not class_names:
        raise SystemExit(f"No class folders found in {data_dir}")

    rng = random.Random(seed)
    train, val = [], []
    for idx, name in enumerate(class_names):
        folder = os.path.join(data_dir, name)
        files = sorted(
            os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(IMAGE_EXTS)
        )
        rng.shuffle(files)
        if max_per_class:
            files = files[:max_per_class]
        n_val = max(1, int(len(files) * val_split))
        val += [(f, idx) for f in files[:n_val]]
        train += [(f, idx) for f in files[n_val:]]
        print(f"  {name:55s} {len(files):5d} images")
    rng.shuffle(train)
    return class_names, train, val


def make_dataset(pairs, image_size, batch_size, shuffle):
    paths = [p for p, _ in pairs]
    labels = [l for _, l in pairs]

    def load(path, label):
        img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        img = tf.image.resize(img, (image_size, image_size), method="bilinear")
        return tf.cast(img, tf.float32), label  # raw 0-255 values, like the app

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(min(len(paths), 5000), reshuffle_each_iteration=True)
    return ds.map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)


def build_model(num_classes, image_size, weights, alpha=1.0):
    """MobileNetV2 backbone + small classification head."""
    base = tf.keras.applications.MobileNetV2(
        input_shape=(image_size, image_size, 3),
        include_top=False,
        alpha=alpha,
        weights=None if weights == "none" else "imagenet",
    )
    base.trainable = weights == "none"  # freeze only when using pre-trained weights
    if weights == "none":
        # Keras' MobileNetV2 uses BatchNorm momentum 0.999, tuned for pre-trained weights.
        # From scratch the running statistics then lag far behind the weights and
        # validation accuracy stays at chance level, so use a faster momentum.
        for layer in base.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.momentum = 0.9

    inputs = tf.keras.Input(shape=(image_size, image_size, 3), name="image")
    # Light augmentation, active only during training
    x = tf.keras.layers.RandomFlip("horizontal_and_vertical")(inputs)
    x = tf.keras.layers.RandomRotation(0.1)(x)
    x = tf.keras.layers.RandomZoom(0.1)(x)
    # MobileNetV2 preprocessing (0-255 -> -1..1) stored inside the model
    x = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1.0, name="preprocess")(x)
    x = base(x, training=False if weights != "none" else None)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    return tf.keras.Model(inputs, outputs, name="plantcare_mobilenetv2"), base


def main():
    args = parse_args()
    tf.keras.utils.set_random_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Reading dataset from {args.data_dir}")
    class_names, train_pairs, val_pairs = collect_files(
        args.data_dir, args.max_per_class, args.val_split, args.seed)
    print(f"\n{len(class_names)} classes | {len(train_pairs)} training | {len(val_pairs)} validation images\n")

    train_ds = make_dataset(train_pairs, args.image_size, args.batch_size, shuffle=True)
    val_ds = make_dataset(val_pairs, args.image_size, args.batch_size, shuffle=False)

    model, base = build_model(len(class_names), args.image_size, args.weights, args.alpha)
    if args.init_from:
        # load before training: the checkpoint below may overwrite the same file
        warm_start(model, base, args.init_from, class_names)
    model_path = os.path.join(args.output_dir, "plant_disease_model.keras")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(model_path, monitor="val_accuracy", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", restore_best_weights=True,
                                         patience=3 if args.weights != "none" else 6),
        # Halve the learning rate when validation loss stops improving
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2,
                                             min_lr=1e-6, verbose=1),
    ]

    start = time.time()
    model.compile(optimizer=tf.keras.optimizers.Adam(args.lr),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    print("Phase 1: training the classification head")
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks, verbose=2)

    if args.fine_tune_epochs > 0 and args.weights != "none":
        print("\nPhase 2: fine-tuning the top of MobileNetV2")
        base.trainable = True
        for layer in base.layers[:-30]:
            layer.trainable = False
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                      loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        model.fit(train_ds, validation_data=val_ds,
                  epochs=args.epochs + args.fine_tune_epochs, initial_epoch=args.epochs,
                  callbacks=callbacks, verbose=2)

    # Reload the best checkpoint and evaluate it
    best = tf.keras.models.load_model(model_path)
    loss, acc = best.evaluate(val_ds, verbose=0)

    # Per-class accuracy, useful for the project report
    y_true, y_pred = [], []
    for images, labels in val_ds:
        y_pred += list(np.argmax(best.predict(images, verbose=0), axis=1))
        y_true += list(labels.numpy())
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    per_class = {
        name: round(float(np.mean(y_pred[y_true == i] == i)) * 100, 1)
        for i, name in enumerate(class_names) if np.any(y_true == i)
    }

    info = {
        "class_names": class_names,
        "image_size": args.image_size,
        "input_scale": "0-255",
        "architecture": (f"MobileNetV2 (alpha {args.alpha}) "
                         + ("transfer learning from ImageNet" if args.weights != "none" else "trained from scratch")
                         + (f", warm-started from {os.path.basename(args.init_from)}" if args.init_from else "")),
        "dataset": os.path.basename(os.path.normpath(args.data_dir)),
        "train_images": len(train_pairs),
        "val_images": len(val_pairs),
        "val_accuracy": round(float(acc) * 100, 2),
        "per_class_val_accuracy": per_class,
        "training_minutes": round((time.time() - start) / 60, 1),
    }
    with open(os.path.join(args.output_dir, "model_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    print(f"\nDone. Validation accuracy: {acc * 100:.2f}%")
    print(f"Saved model : {model_path}")
    print(f"Saved info  : {os.path.join(args.output_dir, 'model_info.json')}")


if __name__ == "__main__":
    main()
