"""
prediction.py - Loads the trained Keras model once and runs predictions.

The model is loaded a single time when the Flask app starts (see app.py).
If the model file or its metadata is missing, the app still starts, but
every prediction request returns a clear error instead of a fake result.
"""
import json
import os

import numpy as np

import config
from backend.preprocessing import preprocess_for_model


class ModelNotAvailableError(Exception):
    """Raised when a prediction is requested but no working model is loaded."""


class PlantDiseaseModel:
    """Small wrapper that owns the Keras model and its class-label mapping."""

    def __init__(self):
        self.model = None
        self.class_names = []
        self.image_size = 224
        self.input_scale = "0-255"
        self.val_accuracy = None  # from model_info.json, shown on the About page
        self.error = None  # human-readable reason when loading failed

    # ------------------------------------------------------------------
    def load(self, model_path=config.MODEL_PATH, info_path=config.MODEL_INFO_PATH):
        """Load the model file and model_info.json. Safe to call once at startup."""
        if not os.path.exists(model_path):
            self.error = (
                "AI model file not found at models/plant_disease_model.keras. "
                "Train it with train_model.py (see README.md) and restart the server."
            )
            return False

        if not os.path.exists(info_path):
            self.error = (
                "models/model_info.json is missing. It holds the class-label order "
                "the model was trained with, so predictions cannot be decoded without it."
            )
            return False

        try:
            with open(info_path, encoding="utf-8") as f:
                info = json.load(f)
            self.class_names = list(info["class_names"])
            self.image_size = int(info.get("image_size", 224))
            self.input_scale = info.get("input_scale", "0-255")
            self.val_accuracy = info.get("val_accuracy")
        except (ValueError, KeyError) as exc:
            self.error = f"models/model_info.json is invalid: {exc}"
            return False

        try:
            # Imported here so the web pages still work if TensorFlow is not installed
            import tensorflow as tf

            self.model = tf.keras.models.load_model(model_path, compile=False)
        except ImportError:
            self.error = "TensorFlow is not installed. Run: pip install -r requirements.txt"
            return False
        except Exception as exc:  # noqa: BLE001 - any Keras load failure
            self.error = f"The AI model could not be loaded: {exc}"
            self.model = None
            return False

        # The number of model outputs must match the number of labels
        output_units = int(self.model.output_shape[-1])
        if output_units != len(self.class_names):
            self.error = (
                f"Model has {output_units} outputs but model_info.json lists "
                f"{len(self.class_names)} classes. They must match."
            )
            self.model = None
            return False

        # Warm-up run so the first real request is fast
        self.model.predict(np.zeros((1, self.image_size, self.image_size, 3), np.float32), verbose=0)
        self.error = None
        return True

    # ------------------------------------------------------------------
    @property
    def is_ready(self):
        return self.model is not None

    def predict(self, image):
        """
        Classify a PIL RGB image.

        Returns a dict: label (raw class name), confidence (0-100),
        and top3 (list of {label, confidence}) for transparency.
        """
        if not self.is_ready:
            raise ModelNotAvailableError(self.error or "AI model is not loaded.")

        batch = preprocess_for_model(image, self.image_size, self.input_scale)
        probs = self.model.predict(batch, verbose=0)[0].astype(np.float64)

        # If the model's last layer was not a softmax, convert logits to probabilities
        if probs.min() < 0 or abs(probs.sum() - 1.0) > 1e-3:
            exp = np.exp(probs - probs.max())
            probs = exp / exp.sum()

        order = np.argsort(probs)[::-1]
        top3 = [
            {"label": self.class_names[i], "confidence": round(float(probs[i]) * 100, 2)}
            for i in order[:3]
        ]
        return {"label": top3[0]["label"], "confidence": top3[0]["confidence"], "top3": top3}


# One shared instance used by the whole app
disease_model = PlantDiseaseModel()
