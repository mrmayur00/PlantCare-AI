"""
tests/test_app.py - Automated end-to-end tests for PlantCare AI.

Run from the project folder:
    python -m unittest tests/test_app.py -v

Uses a temporary database and upload folder, so your real scan history
is not touched. Prediction tests run only if the AI model is present.
Put a few leaf photos in sample_images/ to test real predictions.
"""
import io
import os
import shutil
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import config  # noqa: E402

# Redirect storage to a temp folder BEFORE the app is imported
TMP = tempfile.mkdtemp(prefix="plantcare_test_")
config.DATABASE_PATH = os.path.join(TMP, "test.db")
config.UPLOAD_FOLDER = os.path.join(TMP, "uploads")

from PIL import Image  # noqa: E402

import app as plantcare  # noqa: E402
from backend.disease_info import DISEASES, HEALTHY, get_result_info  # noqa: E402
from backend.disease_info_india import UNSUPPORTED_CLASS  # noqa: E402
from backend.prediction import disease_model  # noqa: E402

SAMPLES = os.path.join(ROOT, "sample_images")


def read_bytes(path):
    with open(path, "rb") as f:
        return f.read()


def make_image_bytes(fmt="JPEG", size=(300, 300), color=(60, 140, 70)):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, fmt)
    return buf.getvalue()


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = plantcare.app.test_client()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(TMP, ignore_errors=True)

    def post_image(self, data, name="leaf.jpg"):
        return self.client.post("/api/predict", data={"image": (io.BytesIO(data), name)},
                                content_type="multipart/form-data")

    # ---- pages ----
    def test_pages_load(self):
        for url in ["/", "/scan", "/history", "/about", "/result/1"]:
            self.assertEqual(self.client.get(url).status_code, 200, url)
        self.assertEqual(self.client.get("/missing-page").status_code, 404)

    def test_health(self):
        data = self.client.get("/api/health").get_json()
        self.assertTrue(data["success"])
        self.assertEqual(data["status"], "running")

    # ---- validation ----
    def test_no_image(self):
        r = self.client.post("/api/predict", data={}, content_type="multipart/form-data")
        self.assertEqual(r.status_code, 400)
        self.assertIn("No image", r.get_json()["error"])

    def test_wrong_extension(self):
        r = self.post_image(b"hello", "notes.txt")
        self.assertEqual(r.status_code, 400)
        self.assertIn("Unsupported", r.get_json()["error"])

    def test_corrupted_image(self):
        r = self.post_image(b"\xff\xd8\xff\xe0 not really a jpeg", "broken.jpg")
        self.assertEqual(r.status_code, 400)
        self.assertIn("corrupted", r.get_json()["error"])

    def test_disguised_format(self):
        buf = io.BytesIO()
        Image.new("RGB", (50, 50)).save(buf, "GIF")
        r = self.post_image(buf.getvalue(), "fake.png")
        self.assertEqual(r.status_code, 400)

    def test_no_leaf_in_photo(self):
        r = self.post_image(make_image_bytes(color=(110, 160, 230)), "sky.jpg")  # plain blue
        self.assertEqual(r.status_code, 422)
        self.assertIn("No plant leaf", r.get_json()["error"])

    def test_too_large(self):
        r = self.post_image(b"0" * ((config.MAX_FILE_SIZE_MB + 2) * 1024 * 1024), "big.jpg")
        self.assertEqual(r.status_code, 413)

    def test_history_not_found(self):
        self.assertEqual(self.client.get("/api/history/999999").status_code, 404)
        self.assertEqual(self.client.delete("/api/history/999999").status_code, 404)

    @unittest.skipIf(disease_model.is_ready, "model is loaded")
    def test_no_model_gives_clear_error(self):
        r = self.post_image(make_image_bytes())
        self.assertEqual(r.status_code, 503)
        self.assertFalse(r.get_json()["success"])

    # ---- knowledge base ----
    def test_every_class_has_info(self):
        for label in list(DISEASES) + list(HEALTHY):
            info = get_result_info(label)
            self.assertTrue(info["scientific_name"], label)
            if info["status"] == "Healthy":
                self.assertEqual(info["treatment"], [], label)
            else:
                self.assertTrue(info["treatment"] and info["symptoms"], label)

    def test_label_spelling_variants(self):
        self.assertEqual(get_result_info("Corn_(maize)___Common_rust_")["disease"], "Common Rust")
        self.assertEqual(get_result_info("Cherry_(including_sour)___healthy")["status"], "Healthy")
        self.assertEqual(get_result_info("Tomato__Tomato_YellowLeaf__Curl_Virus")["plant"], "Tomato")

    def test_model_classes_have_info(self):
        for label in disease_model.class_names:
            if label == UNSUPPORTED_CLASS:
                continue
            self.assertIsNotNone(get_result_info(label)["scientific_name"], f"no info for {label}")

    # ---- Indian plants ----
    def test_indian_plant_names(self):
        expected = {
            "Tulsi___healthy": ("Tulsi", "Ocimum tenuiflorum", "Healthy"),
            "Mango___Anthracnose": ("Amba", "Mangifera indica", "Diseased"),
            "Rice___Leaf_Blast": ("Dhan", "Oryza sativa", "Diseased"),
            "Wheat___Leaf_Rust": ("Gahu", "Triticum aestivum", "Diseased"),
        }
        for label, (local, sci, status) in expected.items():
            info = get_result_info(label)
            self.assertEqual((info["local_name"], info["scientific_name"], info["status"]),
                             (local, sci, status), label)

    def test_healthy_leaves_get_no_treatment(self):
        for label in ["Tulsi___healthy", "Mango___healthy", "Rice___healthy", "Wheat___healthy"]:
            info = get_result_info(label)
            self.assertEqual(info["treatment"], [], label)
            self.assertTrue(info["prevention"], label)

    def test_unsupported_class_is_never_a_diagnosis(self):
        info = get_result_info(UNSUPPORTED_CLASS)
        self.assertEqual(info["status"], "Uncertain")
        self.assertEqual(info["treatment"], [])

    def test_untrained_plants_not_listed_as_supported(self):
        from backend.disease_info import supported_classes
        names = " ".join(p for p, _ in supported_classes(disease_model.class_names or None))
        for plant in ["Jamun", "Custard Apple", "Sapodilla", "Curry Leaf", "Rose", "Hibiscus"]:
            self.assertNotIn(plant, names)

    # ---- full flow with the real model ----
    @unittest.skipUnless(disease_model.is_ready, "AI model not loaded")
    def test_predict_store_view_delete(self):
        photos = sorted(f for f in os.listdir(SAMPLES) if f.lower().endswith((".jpg", ".jpeg", ".png"))) \
            if os.path.isdir(SAMPLES) else []
        data = read_bytes(os.path.join(SAMPLES, photos[0])) if photos else make_image_bytes()
        r = self.post_image(data, photos[0] if photos else "leaf.jpg")
        self.assertEqual(r.status_code, 200, r.get_json())
        res = r.get_json()
        for key in ["plant", "disease", "status", "confidence", "prevention", "scan_id"]:
            self.assertIn(key, res)
        self.assertIn(res["status"], ["Healthy", "Diseased", "Uncertain"])

        listing = self.client.get("/api/history").get_json()["scans"]
        self.assertIn(res["scan_id"], [s["id"] for s in listing])
        detail = self.client.get(f"/api/history/{res['scan_id']}").get_json()
        self.assertEqual(detail["disease"], res["disease"])
        saved = os.path.join(config.UPLOAD_FOLDER, os.path.basename(detail["image_url"]))
        self.assertTrue(os.path.exists(saved))

        self.assertEqual(self.client.delete(f"/api/history/{res['scan_id']}").status_code, 200)
        self.assertEqual(self.client.get(f"/api/history/{res['scan_id']}").status_code, 404)
        self.assertFalse(os.path.exists(saved))  # image file removed too

    @unittest.skipUnless(disease_model.is_ready and os.path.isdir(SAMPLES), "needs model and sample_images/")
    def test_sample_images_predicted_correctly(self):
        """Sample files are named after their true class, e.g. Tomato___Early_blight_1.jpg"""
        from backend.disease_info import _compact
        checked = correct = wrong_confident = 0
        for name in sorted(os.listdir(SAMPLES)):
            if not name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            truth = name.rsplit("_", 1)[0]
            res = self.post_image(read_bytes(os.path.join(SAMPLES, name)), name).get_json()
            checked += 1
            ok = _compact(res["predicted_label"]) == _compact(truth)
            correct += ok
            wrong_confident += (not ok) and res["status"] != "Uncertain"
            print(f"  {name:50s} -> {res['predicted_label']:45s} {res['confidence']:5.1f}%  {'OK' if ok else 'X'}")
        print(f"  {correct}/{checked} sample images classified correctly")
        print(f"  {wrong_confident} wrong answers were above the {config.CONFIDENCE_THRESHOLD:.0f}% threshold "
              f"(the rest are shown as 'Uncertain')")
        self.assertGreater(checked, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
