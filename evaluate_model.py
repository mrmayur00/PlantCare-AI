"""
evaluate_model.py - Test the trained model on independent test images.

The test images (data/test/<class>/) are created by prepare_dataset.py and are
never used for training or validation, so this is an honest estimate of how
the model performs on new photos from the same datasets.

Usage:
    python evaluate_model.py                      # uses models/ and data/test
    python evaluate_model.py --model-dir models --test-dir data/test

Writes models/test_report.json and prints, per plant:
    plant      - was the plant species identified correctly?
    status     - was healthy vs diseased correct?
    disease    - was the exact class (plant + disease) correct?
    confident  - share of photos above the confidence threshold, and the
                 accuracy on just those (what the app shows as a diagnosis)
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import config  # noqa: E402
from backend.disease_info import get_result_info  # noqa: E402
from backend.preprocessing import preprocess_for_model  # noqa: E402

IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate PlantCare AI on held-out test images")
    p.add_argument("--model-dir", default=os.path.join(HERE, "models"))
    p.add_argument("--test-dir", default=os.path.join(HERE, "data", "test"))
    p.add_argument("--threshold", type=float, default=config.CONFIDENCE_THRESHOLD)
    p.add_argument("--unseen-dir", default=os.path.join(HERE, "sample_images", "not_supported"),
                   help="photos of plants the model does NOT support; every one should "
                        "come out as uncertain / unsupported")
    return p.parse_args()


def check_unseen(model, info, folder, threshold):
    """How often does the model wrongly 'diagnose' a plant it was never trained on?"""
    from PIL import Image
    from backend.disease_info_india import UNSUPPORTED_CLASS
    files = [os.path.join(r, f) for r, _, fs in os.walk(folder) for f in sorted(fs)
             if f.lower().endswith(IMAGE_EXTS)]
    if not files:
        return None
    wrong = []
    for path in files:
        batch = preprocess_for_model(Image.open(path).convert("RGB"), info.get("image_size", 224),
                                     info.get("input_scale", "0-255"))
        pr = model.predict(batch, verbose=0)[0]
        label, conf = info["class_names"][int(np.argmax(pr))], float(pr.max()) * 100
        if conf >= threshold and label != UNSUPPORTED_CLASS:
            wrong.append({"file": os.path.relpath(path, folder), "predicted": label, "confidence": round(conf, 1)})
    print(f"\nPlants the model was never trained on ({len(files)} photos in {folder}):")
    print(f"  shown as uncertain / unsupported : {len(files) - len(wrong)}")
    print(f"  wrongly given a diagnosis        : {len(wrong)}")
    for w in wrong:
        print(f"    {w['file']:45s} -> {w['predicted']} ({w['confidence']}%)")
    return {"photos": len(files), "wrongly_diagnosed": len(wrong), "details": wrong}


def plant_of(label):
    return get_result_info(label)["plant"]


def status_of(label):
    return get_result_info(label)["status"]


def main():
    args = parse_args()
    import tensorflow as tf
    from PIL import Image

    with open(os.path.join(args.model_dir, "model_info.json"), encoding="utf-8") as f:
        info = json.load(f)
    classes = info["class_names"]
    size = info.get("image_size", 224)
    model = tf.keras.models.load_model(os.path.join(args.model_dir, "plant_disease_model.keras"), compile=False)

    rows = []  # (true_label, predicted_label, confidence)
    for true_label in sorted(os.listdir(args.test_dir)):
        folder = os.path.join(args.test_dir, true_label)
        if not os.path.isdir(folder):
            continue
        if true_label not in classes:
            print(f"  skipping {true_label}: not a class of this model")
            continue
        files = [os.path.join(folder, f) for f in sorted(os.listdir(folder)) if f.lower().endswith(IMAGE_EXTS)]
        for i in range(0, len(files), 64):
            batch = np.concatenate([
                preprocess_for_model(Image.open(p).convert("RGB"), size, info.get("input_scale", "0-255"))
                for p in files[i:i + 64]])
            probs = model.predict(batch, verbose=0)
            for pr in probs:
                k = int(np.argmax(pr))
                rows.append((true_label, classes[k], float(pr[k]) * 100))

    # Aggregate per plant
    stats = defaultdict(lambda: defaultdict(int))
    per_class = defaultdict(lambda: [0, 0])
    confusions = defaultdict(int)
    for true, pred, conf in rows:
        p = plant_of(true)
        s = stats[p]
        s["n"] += 1
        s["plant_ok"] += plant_of(pred) == p
        s["status_ok"] += status_of(pred) == status_of(true) and plant_of(pred) == p
        s["exact_ok"] += pred == true
        if conf >= args.threshold:
            s["confident"] += 1
            s["confident_ok"] += pred == true
        per_class[true][0] += 1
        per_class[true][1] += pred == true
        if pred != true:
            confusions[(true, pred)] += 1

    def pct(a, b):
        return round(100.0 * a / b, 1) if b else None

    report = {"threshold": args.threshold, "test_images": len(rows), "plants": {}, "classes": {}}
    print(f"\n{'Plant':22s} {'test':>5s} {'plant':>7s} {'status':>7s} {'disease':>8s} "
          f"{'shown':>7s} {'acc(shown)':>11s}")
    for plant in sorted(stats):
        s = stats[plant]
        r = {
            "test_images": s["n"],
            "plant_identified_pct": pct(s["plant_ok"], s["n"]),
            "health_status_pct": pct(s["status_ok"], s["n"]),
            "exact_class_pct": pct(s["exact_ok"], s["n"]),
            "confident_pct": pct(s["confident"], s["n"]),
            "accuracy_when_confident_pct": pct(s["confident_ok"], s["confident"]),
        }
        report["plants"][plant] = r
        print(f"{plant:22s} {s['n']:5d} {r['plant_identified_pct']:6.1f}% {r['health_status_pct']:6.1f}% "
              f"{r['exact_class_pct']:7.1f}% {r['confident_pct']:6.1f}% "
              f"{(r['accuracy_when_confident_pct'] or 0):10.1f}%")
    for c, (n, ok) in sorted(per_class.items()):
        report["classes"][c] = {"test_images": n, "accuracy_pct": pct(ok, n)}
    total_ok = sum(ok for n, ok in per_class.values())
    report["overall_exact_class_pct"] = pct(total_ok, len(rows))
    report["top_confusions"] = [
        {"true": t, "predicted": p, "count": n}
        for (t, p), n in sorted(confusions.items(), key=lambda x: -x[1])[:15]]

    print(f"\nOverall exact-class accuracy: {report['overall_exact_class_pct']}% on {len(rows)} test images")
    print("Most common mistakes:")
    for c in report["top_confusions"][:8]:
        print(f"  {c['count']:3d} x {c['true']}  ->  {c['predicted']}")

    if os.path.isdir(args.unseen_dir):
        report["unsupported_plants"] = check_unseen(model, info, args.unseen_dir, args.threshold)

    out = os.path.join(args.model_dir, "test_report.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
