"""
preprocessing.py - Image validation and preprocessing.

The same steps used during training (train_model.py) are applied here:
    1. Convert to RGB
    2. Resize to the model's square input size (default 224 x 224)
    3. Feed raw 0-255 pixel values; the model rescales them internally
       (a Rescaling layer inside the saved model), so training and
       prediction can never drift apart.
"""
import io
import os
import uuid

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

import config


class ImageValidationError(Exception):
    """Raised when an uploaded file is not a usable image."""


def allowed_file(filename):
    """Return True if the file name has a JPG / JPEG / PNG extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in config.ALLOWED_EXTENSIONS


def load_image(file_bytes):
    """
    Open the uploaded bytes as a PIL image and check it is a real JPEG/PNG.

    Raises ImageValidationError with a friendly message if anything is wrong.
    """
    if not file_bytes:
        raise ImageValidationError("The uploaded file is empty.")

    max_bytes = config.MAX_FILE_SIZE_MB * 1024 * 1024
    if len(file_bytes) > max_bytes:
        raise ImageValidationError(
            f"Image is too large. Maximum size is {config.MAX_FILE_SIZE_MB} MB."
        )

    try:
        # verify() detects truncated / corrupted files
        probe = Image.open(io.BytesIO(file_bytes))
        probe.verify()
        image = Image.open(io.BytesIO(file_bytes))
        image.load()
    except (UnidentifiedImageError, OSError, SyntaxError):
        raise ImageValidationError("The file is not a valid image or it is corrupted.")

    # Check the real content type, not just the file extension
    if image.format not in ("JPEG", "PNG"):
        raise ImageValidationError("Unsupported image format. Please upload a JPG or PNG image.")

    # Phone cameras store rotation in EXIF; apply it so the leaf is upright
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


def leaf_pixel_ratio(image):
    """
    Fraction of pixels with leaf-like colours (0.0 - 1.0).

    Counts green / yellow-green pixels and yellow-brown pixels (diseased
    tissue). Used to reject photos with no leaf at all (a wall, the sky,
    a screen) before the classifier is asked to choose a disease.
    """
    rgb = np.asarray(image.convert("RGB").resize((128, 128)), dtype=np.float32)

    def to_hsv(array):
        hsv = np.asarray(Image.fromarray(array.astype(np.uint8)).convert("HSV"), dtype=np.float32)
        return hsv[..., 0] * 360.0 / 255.0, hsv[..., 1] / 255.0, hsv[..., 2] / 255.0

    # Living leaves: green, and yellow / brown diseased tissue
    hue, sat, val = to_hsv(rgb)
    green = (hue >= 40) & (hue <= 170) & (sat >= 0.15) & (val >= 0.12)
    yellow_brown = (hue >= 10) & (hue < 40) & (sat >= 0.25) & (val >= 0.12)

    # Dried leaves (for example mango die back) are dull grey-brown. Under a
    # blue or yellow colour cast they look neutral, so first remove the cast
    # with a simple "grey-world" white balance, then look for dull browns.
    balanced = np.clip(rgb * (rgb.mean() / np.maximum(rgb.reshape(-1, 3).mean(axis=0), 1.0)), 0, 255)
    hue_b, sat_b, val_b = to_hsv(balanced)
    dried = (((hue_b <= 65) | (hue_b >= 345)) & (sat_b >= 0.08)
             & (val_b >= 0.12) & (val_b <= 0.65))

    return float((green | yellow_brown | dried).mean())


def preprocess_for_model(image, image_size, input_scale="0-255"):
    """
    Convert a PIL RGB image into the array the model expects.

    Returns a float32 array of shape (1, image_size, image_size, 3).
    """
    resized = image.resize((image_size, image_size), Image.BILINEAR)
    array = np.asarray(resized, dtype=np.float32)
    if input_scale == "0-1":
        # Only for externally trained models that expect 0-1 input
        array = array / 255.0
    return np.expand_dims(array, axis=0)


def save_upload(image):
    """
    Save a downsized JPEG copy of the image into static/uploads.

    Returns the file name (not the full path) so it can be stored in the
    database and served by Flask from /static/uploads/<name>.
    """
    os.makedirs(config.UPLOAD_FOLDER, exist_ok=True)
    copy = image.copy()
    copy.thumbnail((config.SAVED_IMAGE_MAX_SIDE, config.SAVED_IMAGE_MAX_SIDE))
    filename = f"{uuid.uuid4().hex}.jpg"
    copy.save(os.path.join(config.UPLOAD_FOLDER, filename), "JPEG", quality=88)
    return filename
