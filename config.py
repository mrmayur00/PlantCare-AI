"""
config.py - Central configuration for PlantCare AI.

All paths and tunable settings live here so the rest of the code
never has to hard-code them.
"""
import os

# Root folder of the project (the folder that contains this file)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# AI model
# ---------------------------------------------------------------------------
# The trained Keras model. Create it with train_model.py (see README.md).
MODEL_PATH = os.path.join(BASE_DIR, "models", "plant_disease_model.keras")

# Metadata written by train_model.py next to the model:
#   - class_names : list of labels in the exact order of the model's outputs
#   - image_size  : the square input size the model was trained on
#   - input_scale : "0-255" if the model rescales internally, "0-1" otherwise
MODEL_INFO_PATH = os.path.join(BASE_DIR, "models", "model_info.json")

# Predictions below this confidence (in %) are reported as "Uncertain"
# instead of showing a disease name that may be wrong. 75 was chosen from the
# test results: at 75% about 81% of test photos get a diagnosis and 96.5% of
# those are correct, and most photos of plants the model never saw are
# reported as uncertain (see models/test_report.json and README section 8).
CONFIDENCE_THRESHOLD = 75.0

# Second rejection rule: the gap (in percentage points) between the model's top
# and second-best answer must be at least this large. A softmax model must
# spread 100% over its known classes, so a leaf it does not know (e.g. Rose)
# often scores moderately high for the *closest* known class but with a small
# gap to the runner-up. 0 = rule disabled. Do NOT guess a value: run
# `python evaluate_model.py`, read the "margin sweep" it prints, and set
# PLANTCARE_MIN_MARGIN (env var, e.g. on Render) or edit the default below.
MIN_TOP2_MARGIN = float(os.environ.get("PLANTCARE_MIN_MARGIN", "0"))

# ---------------------------------------------------------------------------
# Stage 1: plant species model (created by train_species.py)
# ---------------------------------------------------------------------------
# If these two files are missing the app keeps working with the old single model.
SPECIES_MODEL_PATH = os.path.join(BASE_DIR, "models", "species_model.keras")
SPECIES_INFO_PATH = os.path.join(BASE_DIR, "models", "species_info.json")
# The accept thresholds are calibrated by train_species.py and stored in
# species_info.json. Set this only to override the confidence (in %) on purpose.
_sp = os.environ.get("PLANTCARE_SPECIES_MIN_CONF", "")
SPECIES_MIN_CONFIDENCE_PCT = float(_sp) if _sp else None

# Photo quality limits (simple heuristics, lenient on purpose; adjust on real photos).
# Brightness = mean grey level 0-255; sharpness = variance of the Laplacian at 256 px.
QUALITY_MIN_BRIGHTNESS = float(os.environ.get("PLANTCARE_MIN_BRIGHTNESS", "35"))
QUALITY_MAX_BRIGHTNESS = float(os.environ.get("PLANTCARE_MAX_BRIGHTNESS", "240"))
QUALITY_MIN_SHARPNESS = float(os.environ.get("PLANTCARE_MIN_SHARPNESS", "15"))

# Simple "is there a leaf?" check run before the model. If less than this
# fraction of the image has leaf-like colours (green, yellow, brown), the photo
# is rejected. Calibrated on 1,520 PlantVillage images and on the 3,570 test
# images of the current model: only 4 real leaves (0.1%) fall below 1.5%.
MIN_LEAF_PIXEL_RATIO = 0.015

# ---------------------------------------------------------------------------
# Uploads
# ---------------------------------------------------------------------------
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
MAX_FILE_SIZE_MB = 10
# Saved copies are shrunk to this longest side to keep the uploads folder small
SAVED_IMAGE_MAX_SIDE = 1024

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
DATABASE_PATH = os.path.join(BASE_DIR, "database", "plantcare.db")

# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------
HOST = "0.0.0.0"   # 0.0.0.0 lets phones on the same Wi-Fi reach the app
PORT = 5000
DEBUG = False
