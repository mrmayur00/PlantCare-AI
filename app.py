"""
app.py - PlantCare AI Flask application.

Run:   python app.py            (http://localhost:5000)
       python app.py --https    (self-signed HTTPS, for live camera on phones)

Pages:  /  /scan  /result/<id>  /history  /about
API:    POST /api/predict, GET /api/history, GET/DELETE /api/history/<id>, GET /api/health
"""
import argparse
import os
import socket

from flask import Flask, abort, jsonify, render_template, request
from werkzeug.exceptions import RequestEntityTooLarge

import config
from backend import database
from backend.disease_info import (get_result_info, plants_without_model_support,
                                  set_model_classes, supported_classes)
from backend.prediction import ModelNotAvailableError, disease_model
from backend.preprocessing import (ImageValidationError, allowed_file, leaf_pixel_ratio,
                                   load_image, save_upload)

app = Flask(__name__)
# Flask rejects bigger uploads before they reach our code (small margin for form overhead)
app.config["MAX_CONTENT_LENGTH"] = (config.MAX_FILE_SIZE_MB + 1) * 1024 * 1024

# ---------------------------------------------------------------------------
# Start-up: create the database and load the AI model ONCE
# ---------------------------------------------------------------------------
database.init_db()
os.makedirs(config.UPLOAD_FOLDER, exist_ok=True)
if disease_model.load():
    print(f"[PlantCare AI] Model loaded: {len(disease_model.class_names)} classes, "
          f"input {disease_model.image_size}x{disease_model.image_size}")
    # Tell the knowledge base which classes the AI can really recognise
    set_model_classes(disease_model.class_names)
else:
    print(f"[PlantCare AI] WARNING: {disease_model.error}")


def error_response(message, status):
    """Consistent JSON error format used by every API endpoint."""
    return jsonify({"success": False, "error": message}), status


@app.context_processor
def inject_globals():
    """Values available in every template."""
    if disease_model.is_ready:
        pairs = supported_classes(disease_model.class_names)
        n_plants, n_conditions = len({p for p, _ in pairs}), len(pairs)
    else:
        n_plants, n_conditions = None, None
    return {"model_ready": disease_model.is_ready, "max_mb": config.MAX_FILE_SIZE_MB,
            "n_plants": n_plants, "n_conditions": n_conditions,
            "threshold": int(config.CONFIDENCE_THRESHOLD)}


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html", active="home")


@app.route("/scan")
def scan():
    return render_template("scan.html", active="scan", model_error=disease_model.error)


@app.route("/result/<int:scan_id>")
def result(scan_id):
    # The page loads its data from /api/history/<id> with JavaScript
    return render_template("result.html", active="history", scan_id=scan_id)


@app.route("/history")
def history():
    return render_template("history.html", active="history")


@app.route("/about")
def about():
    if disease_model.is_ready:
        classes = supported_classes(disease_model.class_names)
        model_classes = len(disease_model.class_names)
        not_supported = plants_without_model_support(disease_model.class_names)
    else:
        classes, model_classes, not_supported = supported_classes(), None, []
    return render_template("about.html", active="about", classes=classes, model_classes=model_classes,
                           image_size=disease_model.image_size, val_accuracy=disease_model.val_accuracy,
                           not_supported=not_supported)


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
@app.route("/api/health")
def api_health():
    """Simple status check for the server and the AI model."""
    return jsonify({
        "success": True,
        "status": "running",
        "model_loaded": disease_model.is_ready,
        "model_classes": len(disease_model.class_names) if disease_model.is_ready else 0,
        "model_error": disease_model.error,
    })


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Validate the uploaded leaf image, classify it, store and return the result."""
    # 1. Receive the image
    file = request.files.get("image")
    if file is None or file.filename == "":
        return error_response("No image selected. Please choose or capture a leaf photo.", 400)

    # 2. Validate extension, size and real content
    if not allowed_file(file.filename):
        return error_response("Unsupported file format. Please upload a JPG, JPEG or PNG image.", 400)
    try:
        image = load_image(file.read())
    except ImageValidationError as exc:
        return error_response(str(exc), 400)

    # Reject photos that contain no leaf-coloured area at all
    if leaf_pixel_ratio(image) < config.MIN_LEAF_PIXEL_RATIO:
        return error_response(
            "No plant leaf was detected in this photo. Please photograph a single "
            "leaf so it fills most of the frame.", 422)

    # 3-7. Preprocess and predict (fails clearly if no model is loaded)
    try:
        prediction = disease_model.predict(image)
    except ModelNotAvailableError as exc:
        return error_response(str(exc), 503)
    except Exception as exc:  # noqa: BLE001
        app.logger.exception("Prediction failed")
        return error_response(f"Prediction failed: {exc}", 500)

    confidence = prediction["confidence"]

    # 8-9. Map the label to plant / disease information
    if confidence < config.CONFIDENCE_THRESHOLD:
        # Do not show a disease name the model is unsure about
        result_data = {
            "plant": "Not identified",
            "local_name": None,
            "scientific_name": None,
            "disease": "Not identified",
            "status": "Uncertain",
            "pathogen": None,
            "description": (
                f"The AI is only {confidence:.1f}% confident, which is below the "
                f"{config.CONFIDENCE_THRESHOLD:.0f}% needed to give a reliable answer. "
                "The leaf may belong to a plant this model does not support, or the photo "
                "may be blurry, dark or show more than one leaf."
            ),
            "symptoms": [], "signs": [], "causes": [], "treatment": [],
            "prevention": [
                "Photograph a single leaf filling most of the frame.",
                "Use daylight and a plain background, and keep the camera steady.",
                "Make sure the plant is one of the supported species listed on the About page.",
            ],
        }
    else:
        result_data = get_result_info(prediction["label"])

    result_data.update({
        "success": True,
        "confidence": confidence,
        "predicted_label": prediction["label"],
        "top_predictions": prediction["top3"],
        "threshold": config.CONFIDENCE_THRESHOLD,
    })

    # 10. Store the scan in SQLite
    try:
        filename = save_upload(image)
        scan_id = database.save_scan(result_data, filename)
    except database.DatabaseError as exc:
        return error_response(str(exc), 500)
    except OSError as exc:
        return error_response(f"Could not save the image: {exc}", 500)

    # 11. Return JSON
    result_data["scan_id"] = scan_id
    result_data["image_url"] = f"/static/uploads/{filename}"
    return jsonify(result_data)


@app.route("/api/history", methods=["GET"])
def api_history():
    try:
        return jsonify({"success": True, "scans": database.get_all_scans()})
    except database.DatabaseError as exc:
        return error_response(str(exc), 500)


@app.route("/api/history/<int:scan_id>", methods=["GET"])
def api_history_item(scan_id):
    try:
        scan_data = database.get_scan(scan_id)
    except database.DatabaseError as exc:
        return error_response(str(exc), 500)
    if scan_data is None:
        return error_response("Scan not found. It may have been deleted.", 404)
    scan_data["success"] = True
    return jsonify(scan_data)


@app.route("/api/history/<int:scan_id>", methods=["DELETE"])
def api_history_delete(scan_id):
    try:
        filename = database.delete_scan(scan_id)
    except database.DatabaseError as exc:
        return error_response(str(exc), 500)
    if filename is None:
        return error_response("Scan not found.", 404)
    path = os.path.join(config.UPLOAD_FOLDER, filename)
    if os.path.exists(path):
        os.remove(path)
    return jsonify({"success": True, "deleted_id": scan_id})


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------
@app.errorhandler(RequestEntityTooLarge)
def too_large(_):
    return error_response(f"Image is too large. Maximum size is {config.MAX_FILE_SIZE_MB} MB.", 413)


@app.errorhandler(404)
def not_found(_):
    if request.path.startswith("/api/"):
        return error_response("Endpoint not found.", 404)
    return render_template("404.html", active=None), 404


def local_ip():
    """Find this computer's address on the Wi-Fi network (for testing on a phone)."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return "127.0.0.1"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PlantCare AI")
    parser.add_argument("--https", action="store_true",
                        help="serve over self-signed HTTPS (needed for the live camera on phones)")
    parser.add_argument("--port", type=int, default=config.PORT)
    args = parser.parse_args()

    scheme = "https" if args.https else "http"
    print(f"\n  PlantCare AI running:")
    print(f"    On this computer : {scheme}://localhost:{args.port}")
    print(f"    On your phone    : {scheme}://{local_ip()}:{args.port}  (same Wi-Fi)\n")

    ssl_context = "adhoc" if args.https else None  # 'adhoc' needs: pip install cryptography
    app.run(host=config.HOST, port=args.port, debug=config.DEBUG, ssl_context=ssl_context)
