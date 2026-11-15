"""
species.py - Stage 1 of PlantCare AI: which plant is this?

A dedicated image classifier (trained by train_species.py) that tells the
10 supported species apart and has an explicit "Unsupported" class for every
other leaf or object. The disease model (stage 2) is only consulted when this
stage accepts the photo AND the species has disease classes (see app.py).

The same `to_array` function is used for training and for prediction, so the
two can never differ in resizing / colour handling.

Nothing here claims a plant is supported by itself: a species counts as
supported only if it is present in models/species_info.json["class_names"],
which train_species.py writes from the classes it was really trained on.
"""
import json
import os

import numpy as np
from PIL import Image

import config

UNSUPPORTED_LABEL = "Unsupported"

# folder / class name -> display information
# disease_prefix: prefix of the matching classes in the DISEASE model
#                 (None = no disease model exists for this plant)
SPECIES = {
    "Tulsi":         {"plant": "Tulsi",         "local_name": "Tulsi",     "scientific_name": "Ocimum tenuiflorum",   "disease_prefix": "Tulsi"},
    "Mango":         {"plant": "Mango",         "local_name": "Amba",      "scientific_name": "Mangifera indica",     "disease_prefix": "Mango"},
    "Jamun":         {"plant": "Jamun",         "local_name": "Jambh",     "scientific_name": "Syzygium cumini",      "disease_prefix": None},
    "Custard_Apple": {"plant": "Custard Apple", "local_name": "Sitafal",   "scientific_name": "Annona squamosa",      "disease_prefix": None},
    "Sapodilla":     {"plant": "Sapodilla",     "local_name": "Chiku",     "scientific_name": "Manilkara zapota",     "disease_prefix": None},
    "Curry_Leaf":    {"plant": "Curry Leaf",    "local_name": "Kadi Pata", "scientific_name": "Murraya koenigii",     "disease_prefix": None},
    "Rice":          {"plant": "Rice",          "local_name": "Dhan",      "scientific_name": "Oryza sativa",         "disease_prefix": "Rice"},
    "Wheat":         {"plant": "Wheat",         "local_name": "Gahu",      "scientific_name": "Triticum aestivum",    "disease_prefix": "Wheat"},
    "Rose":          {"plant": "Rose",          "local_name": "Gulab Ful", "scientific_name": "Rosa species",         "disease_prefix": None},
    "Hibiscus":      {"plant": "Hibiscus",      "local_name": "Jaswand",   "scientific_name": "Hibiscus rosa-sinensis", "disease_prefix": None},
}


def to_array(image, size):
    """PIL image -> uint8 array (size, size, 3), RGB. Shared by training and serving."""
    return np.asarray(image.convert("RGB").resize((size, size), Image.BILINEAR), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Photo quality checks (run before any model)
# ---------------------------------------------------------------------------
def quality_issue(image):
    """
    Return a user-facing message if the photo is clearly unusable, else None.
    These are simple heuristics with deliberately lenient limits (config.py);
    they only catch extreme cases. Tune them on your own photos.
    """
    gray = np.asarray(image.convert("L").resize((256, 256)), dtype=np.float32)
    mean = float(gray.mean())
    if mean < config.QUALITY_MIN_BRIGHTNESS:
        return "The photo is too dark. Please take it in daylight and upload it again."
    if mean > config.QUALITY_MAX_BRIGHTNESS:
        return "The photo is overexposed (too bright). Please avoid direct glare and upload it again."
    lap = (4 * gray[1:-1, 1:-1] - gray[:-2, 1:-1] - gray[2:, 1:-1]
           - gray[1:-1, :-2] - gray[1:-1, 2:])
    if float(lap.var()) < config.QUALITY_MIN_SHARPNESS:
        return "The photo is blurry. Please hold the camera steady, focus on the leaf and upload it again."
    return None


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------
class SpeciesModel:
    def __init__(self):
        self.model = None
        self.class_names = []
        self.image_size = 224
        self.min_confidence = 0.9   # fraction 0-1
        self.min_margin = 0.2       # fraction 0-1
        self.metrics = {}
        self.error = None

    @property
    def is_ready(self):
        return self.model is not None

    def load(self):
        """Load models/species_model.keras. Absent files are not an error: the app
        then falls back to the old single-model behaviour."""
        if not (os.path.exists(config.SPECIES_MODEL_PATH) and os.path.exists(config.SPECIES_INFO_PATH)):
            self.error = "species model not installed"
            return False
        try:
            import tensorflow as tf
            with open(config.SPECIES_INFO_PATH, encoding="utf-8") as f:
                info = json.load(f)
            model = tf.keras.models.load_model(config.SPECIES_MODEL_PATH, compile=False)
            if model.output_shape[-1] != len(info["class_names"]):
                raise ValueError(f"model has {model.output_shape[-1]} outputs but "
                                 f"species_info.json lists {len(info['class_names'])} classes")
        except Exception as exc:  # noqa: BLE001
            self.error = f"species model could not be loaded: {exc}"
            return False
        self.model = model
        self.class_names = info["class_names"]
        self.image_size = info.get("image_size", 224)
        self.min_confidence = info.get("min_confidence", 0.9)
        self.min_margin = info.get("min_margin", 0.2)
        self.metrics = info.get("test_metrics", {})
        if config.SPECIES_MIN_CONFIDENCE_PCT is not None:
            self.min_confidence = config.SPECIES_MIN_CONFIDENCE_PCT / 100.0
        self.error = None
        return True

    def supported_plants(self):
        return [c for c in self.class_names if c in SPECIES]

    def plant_info(self, label):
        return SPECIES[label]

    def predict(self, image):
        """Returns label, confidence (%), margin (points), accepted, top3 [{label, confidence}]."""
        batch = to_array(image, self.image_size)[None].astype("float32")  # raw 0-255
        probs = self.model.predict(batch, verbose=0)[0]
        order = np.argsort(probs)[::-1]
        top, second = float(probs[order[0]]), float(probs[order[1]])
        label = self.class_names[int(order[0])]
        accepted = (label != UNSUPPORTED_LABEL and top >= self.min_confidence
                    and (top - second) >= self.min_margin)
        return {
            "label": label,
            "confidence": round(top * 100, 1),
            "margin": round((top - second) * 100, 1),
            "accepted": bool(accepted),
            "top3": [{"label": self.class_names[int(i)], "confidence": round(float(probs[i]) * 100, 1)}
                     for i in order[:3]],
        }


species_model = SpeciesModel()
