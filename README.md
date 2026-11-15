# 🌿 PlantCare AI — Plant Leaf Disease Detection

A web application that identifies plant leaf diseases from a photo using a deep-learning
image classifier (MobileNetV2, TensorFlow/Keras) trained on the **PlantVillage** dataset and on
Indian crop datasets: **Tulsi, Amba (Mango), Dhan (Rice) and Gahu (Wheat)**.

Take a photo with your phone camera or upload one from your gallery, and PlantCare AI shows:

- the plant species, with its local name (e.g. *Amba*) and scientific name
- whether the leaf is **Healthy**, **Diseased**, or the result is **Uncertain**
- the disease name and the AI confidence score
- description, symptoms, visible signs and causes
- treatment steps and prevention / care tips

Every scan is saved in a SQLite database and can be reviewed or deleted on the History page.

---

## 1. Project structure

```
PlantCareAI/
├── app.py                  Flask app: pages + REST API
├── config.py               Paths and settings (threshold, upload size, ...)
├── get_datasets.sh         Downloads every training dataset into sources/
├── prepare_dataset.py      Builds data/train + data/test (resize, de-duplicate, split)
├── dataset_sources.json    Which image folders belong to which model class
├── train_model.py          Trains the AI model (MobileNetV2), can warm-start from an old model
├── evaluate_model.py       Scores the model on the independent test images
├── requirements.txt
├── README.md
├── models/
│   ├── plant_disease_model.keras   trained model            (created by train_model.py)
│   ├── model_info.json             class labels + accuracy  (created by train_model.py)
│   ├── test_report.json            per-plant test results    (created by evaluate_model.py)
│   └── dataset_manifest.json       image counts and sources  (created by prepare_dataset.py)
├── backend/
│   ├── preprocessing.py    image validation, resizing, saving uploads
│   ├── prediction.py       loads the model once, runs predictions
│   ├── disease_info.py     knowledge base (PlantVillage classes + lookup functions)
│   ├── disease_info_india.py  Indian plants: local names, diseases, treatment, prevention
│   └── database.py         SQLite scan_history table
├── templates/              index, scan, result, history, about, 404 (Jinja2)
├── static/
│   ├── css/style.css
│   ├── js/script.js
│   ├── images/             logo and illustration
│   └── uploads/            saved leaf photos (created automatically)
├── database/plantcare.db   created automatically on first run
├── sample_images/          55 test leaves (one per class, NOT used in training)
│   └── not_supported/      18 photos of plants the model was never trained on
└── tests/test_app.py       automated tests
```

---

## 2. Installation

You need **Python 3.10 – 3.13** (TensorFlow does not always support the very newest Python).
Check with `python --version`.

**Windows (Command Prompt / PowerShell)**

```bat
cd PlantCareAI
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**macOS / Linux**

```bash
cd PlantCareAI
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

TensorFlow is a large download (≈ 500 MB); the first install takes a few minutes.

---

## 3. The AI model

The app needs two files in `models/`: `plant_disease_model.keras` and `model_info.json`.

### Option A — use the included model (fastest)

This package includes a trained model in `models/`. It was trained in a CPU-only environment
where ImageNet weights could not be downloaded, so it learned **from scratch**, first on
PlantVillage and then warm-started and extended with the Indian plant datasets.

| | |
|---|---|
| Architecture | MobileNetV2 (width 0.35), 128 × 128 input |
| Classes | 55 = 54 plant/disease classes (18 plants) + 1 "other / unsupported plant" class |
| Training data | 17,144 images (14,579 training + 2,565 validation) |
| Independent test set | **3,570 images never used in training** (`python evaluate_model.py`) |
| Exact class accuracy (test) | **86.6 %** overall |
| Sample images | 48 / 55 correct; of the 7 wrong answers, 4 are shown as "Uncertain" |

**Results for the Indian plants on the independent test images** (from `models/test_report.json`):

| Plant | Test photos | Plant identified | Healthy / diseased correct | Exact disease | Given a diagnosis (≥ 75 %) | Correct when diagnosed |
|---|---|---|---|---|---|---|
| Amba (Mango) | 600 | 99.7 % | 99.2 % | 96.8 % | 95.2 % | 99.1 % |
| Tulsi | 135 | 94.1 % | 88.1 % | 88.1 % | 85.2 % | 97.4 % |
| Gahu (Wheat) | 305 | 78.7 % | 72.8 % | 72.8 % | 60.3 % | 83.7 % |
| Dhan (Rice) | 312 | 96.8 % | 64.1 % | 44.2 % | 17.9 % | 75.0 % |

What these numbers mean:

- **Mango** works very well (MangoLeafBD has clean single-leaf photos). The test photos come from
  the same orchards as the training photos, so real-world accuracy will be somewhat lower.
- **Tulsi**: the dataset only labels plants *healthy* or *unhealthy*, so the app says
  "Unhealthy leaves (cause not identified)" and lists the possible causes instead of naming a disease.
- **Rice**: the plant is recognised, but the disease spots on thin rice blades are tiny and get lost at
  128 × 128 pixels. Most rice photos are therefore shown as **Uncertain** rather than given a wrong
  disease. A 224 × 224 model trained on a GPU (Option B) should do much better.
- **Wheat**: the only wheat dataset reachable was web-collected; its *healthy* photos are mostly
  whole fields. Only *Healthy* and *Leaf Rust* were used. Treat wheat results as experimental.

The included model was produced with:

```bash
bash get_datasets.sh                               # downloads ~4 GB into sources/
export PLANTCARE_SOURCES="$PWD/sources"            # Windows: set PLANTCARE_SOURCES=%CD%\sources
python prepare_dataset.py                          # -> data/train, data/test
python train_model.py --data-dir data/train --init-from models_v1/plant_disease_model.keras \
    --alpha 0.35 --image-size 128 --weights none --epochs 25 --fine-tune-epochs 0 --lr 5e-4 --val-split 0.15
python evaluate_model.py                           # -> models/test_report.json
```

(`models_v1/` was the earlier 38-class PlantVillage model, trained with
`python train_model.py --data-dir sources/plantvillage --max-per-class 300 --alpha 0.35 --image-size 128 --epochs 40 --fine-tune-epochs 0 --weights none`.
`--init-from` copies everything it learned, so only the new classes have to be learned.)

### The 10 Indian plants: what is supported

| Plant | Supported by the AI? | Classes | Dataset |
|---|---|---|---|
| Tulsi – *Ocimum tenuiflorum* | **Yes** | Healthy, Unhealthy | Tulsi dataset, Nagpur/Amravati (Maharashtra) |
| Amba (Mango) – *Mangifera indica* | **Yes** | Healthy, Anthracnose, Bacterial Canker, Cutting Weevil, Die Back, Gall Midge, Powdery Mildew, Sooty Mould | MangoLeafBD |
| Dhan (Rice) – *Oryza sativa* | **Yes** (disease: low accuracy) | Healthy, Brown Spot, Hispa, Leaf Blast | Rice Diseases Image Dataset |
| Gahu (Wheat) – *Triticum aestivum* | **Yes** (experimental) | Healthy, Leaf Rust | Wheat Disease Detection images |
| Jambh (Jamun) – *Syzygium cumini* | No, dataset needed | – | [Jamun Leaf Disease Detection](https://data.mendeley.com/datasets/43d75vptz4/2) |
| Sitafal (Custard Apple) – *Annona squamosa* | No, dataset needed | – | [Custard Apple Disease Image Dataset](https://data.mendeley.com/datasets/jtgh2885yf/2) |
| Chiku (Sapodilla) – *Manilkara zapota* | No, dataset needed | – | no public leaf-disease dataset found |
| Kadi Pata (Curry Leaf) – *Murraya koenigii* | No, dataset needed | – | no public leaf-disease dataset found |
| Gulab Ful (Rose) – *Rosa* spp. | No, dataset needed | – | [Rose Leaf Disease Detection](https://data.mendeley.com/datasets/sfxcjwf6cb) |
| Jaswand (Hibiscus) – *Hibiscus rosa-sinensis* | No, dataset needed | – | [Hibiscus & Tea Leaf Disease](https://data.mendeley.com/datasets/5bzy89brkv/4) |

The six unsupported plants have their names in `backend/disease_info_india.py`, but **the app
does not claim to recognise them**: the About page lists only the classes inside the loaded model.
Their datasets are on Mendeley Data/Kaggle, which could not be downloaded in the environment used to
build this model. Those for Chiku and Kadi Pata only contain healthy leaves (species identification,
no diseases), for example the Kaggle *Indian Medicinal Leaves* dataset.

### Adding one of the remaining plants (e.g. Jamun)

1. Download the dataset, unzip it into `sources/jamun/` (one folder per class).
2. Add one entry per class to `dataset_sources.json`:
   ```json
   "Jamun___Anthracnose": {"folders": ["${PLANTCARE_SOURCES}/jamun/Anthracnose"], "source": "jamun"},
   "Jamun___healthy":     {"folders": ["${PLANTCARE_SOURCES}/jamun/Healthy"],     "source": "jamun"}
   ```
   Class names are `<Plant>___<Disease>`; healthy classes end in `___healthy`.
3. Add the disease information to `INDIAN_DISEASES` and `INDIAN_HEALTHY` in
   `backend/disease_info_india.py`, and set `"trained": True` and `care` tips for the plant.
4. Rebuild and retrain, starting from the current model so nothing is lost:
   ```bash
   python prepare_dataset.py
   python train_model.py --data-dir data/train --output-dir models_new --init-from models/plant_disease_model.keras \
       --alpha 0.35 --image-size 128 --weights none --epochs 25 --fine-tune-epochs 0 --lr 5e-4 --val-split 0.15
   python evaluate_model.py --model-dir models_new
   ```
5. If the test results look good, copy `models_new/*` into `models/` and restart the app.

### Option B — retrain everything on Google Colab (recommended, free GPU, ~1 hour)

With a GPU you can use ImageNet weights and full-size 224 × 224 images, which helps most with the
small rice spots.

1. Open <https://colab.research.google.com>, create a notebook, and choose
   **Runtime → Change runtime type → GPU**.
2. Upload these files from the project (folder icon on the left → upload, keeping the `backend`
   folder): `get_datasets.sh`, `prepare_dataset.py`, `dataset_sources.json`, `train_model.py`,
   `evaluate_model.py`, `config.py` and the `backend/` folder.
3. Run these cells:

```python
!bash get_datasets.sh
```
```python
import os; os.environ["PLANTCARE_SOURCES"] = "/content/sources"
!python prepare_dataset.py --max-train 1000
```
```python
!python train_model.py --data-dir data/train --epochs 6 --fine-tune-epochs 4 --val-split 0.15
!python evaluate_model.py
```
```python
from google.colab import files
for f in ["plant_disease_model.keras", "model_info.json", "test_report.json"]:
    files.download("models/" + f)
```

4. Copy the downloaded files into your project's `models/` folder (replace the old ones)
   and restart `python app.py`.

### Option C — train on your own computer

Same commands as Option A. On a laptop CPU, training takes about 1–2 hours with
`--alpha 0.35 --image-size 128`, or several hours at full size.

### Using another pre-trained model

Any Keras model works if you also write `models/model_info.json`:

```json
{
  "class_names": ["Apple___Apple_scab", "Apple___Black_rot", "..."],
  "image_size": 224,
  "input_scale": "0-1"
}
```

- `class_names` must be in the **exact order of the model's outputs**.
- `input_scale` is `"0-255"` if the model rescales internally (models from `train_model.py`)
  or `"0-1"` if it expects pixels divided by 255.
- The app checks that the number of outputs matches the number of labels and refuses to
  predict if they differ.

**If the model is missing or broken, the app never shows fake results.** The pages still open,
but Detect Disease shows "AI model not available", and `/api/health` reports the reason.

---

## 4. Run the application

```bash
python app.py
```

Then open **<http://localhost:5000>** in your browser.

The terminal also prints an address such as `http://192.168.1.23:5000` for your phone.

### Test on a mobile phone (same Wi-Fi)

1. Connect the phone and computer to the **same Wi-Fi network**.
2. Run `python app.py` and note the "On your phone" address.
3. Open that address in Chrome on the phone.
4. If it does not load, allow Python through the computer's firewall
   (Windows: allow access for "Private networks" when prompted).

**About the camera:** browsers only allow the *live in-page camera* on HTTPS pages.
Over plain HTTP, "Capture using camera" automatically opens the phone's own camera app
instead, which works just as well. For the live camera preview on the phone, run:

```bash
pip install cryptography
python app.py --https
```

and open `https://<your-ip>:5000` (accept the "not secure" warning, since the certificate is self-signed).

---

## 5. How to test

### Automated tests

```bash
python -m unittest tests/test_app.py -v
```

These check every page, the health endpoint, all upload errors (no file, wrong format,
corrupted, disguised GIF, too large, no leaf in the photo), the knowledge base for every class (including local names for the Indian plants), and a full
predict → history → view → delete cycle. With the model present, they also classify all
38 images in `sample_images/` and print the results. A temporary database is used, so your
real history is not changed.

### Manual test checklist

| Feature | How to test |
|---|---|
| Gallery upload | Scan → Upload from gallery → choose a file from `sample_images/` |
| Camera capture | On a phone: Scan → Capture using camera |
| Preview / change / remove | After choosing, use Change and Remove |
| File-type check | Choose a `.txt` or `.gif` file → error message |
| Detection | Press Detect Disease → result page |
| Healthy leaf | Use `sample_images/Tulsi___healthy_1.jpg` → local name *Tulsi*, no treatment section |
| Indian plant disease | Use `sample_images/Mango___Sooty_Mould_1.jpg` → *Amba (Mango)*, Sooty Mould, treatment |
| Unsupported plant | Use any photo in `sample_images/not_supported/` → usually "Result uncertain" |
| No leaf in photo | Upload a photo of a wall or the sky → "No plant leaf was detected" |
| Uncertain result | `sample_images/Rice___Hispa_1.jpg` scores below 75 % → "Result uncertain" |
| History | History page → View Results / Delete |
| Server down | Stop `app.py`, press Detect Disease → "Cannot reach the server" |
| Model missing | Rename `models/model_info.json`, restart → clear model error |

### Sample images

`sample_images/` contains one real leaf for each of the 55 classes, taken from the independent
test set (never used in training). The file name is the true class, e.g.
`Mango___Sooty_Mould_1.jpg` or `Tulsi___healthy_1.jpg`.

`sample_images/not_supported/` holds 18 photos of plants the model was **never** trained on
(Jamun, Guava, Pomegranate, Pongamia and others, from the Chouhan et al. 2019 leaf database).
They should come out as "Uncertain"; at the 75 % threshold 14 of 18 do and 4 are wrongly given a
diagnosis (`python evaluate_model.py` lists them).

---

## 6. API documentation

All responses are JSON. Errors always look like `{"success": false, "error": "message"}`.

### `POST /api/predict`

Analyse a leaf image. Body: `multipart/form-data` with field **`image`** (JPG/JPEG/PNG, max 10 MB).

```bash
curl -F "image=@sample_images/Mango___Sooty_Mould_1.jpg" http://localhost:5000/api/predict
```

Success response (200), shortened:

```json
{
  "success": true,
  "scan_id": 7,
  "plant": "Mango",
  "local_name": "Amba",
  "scientific_name": "Mangifera indica",
  "disease": "Sooty Mould",
  "status": "Diseased",
  "confidence": 90.7,
  "pathogen": "Fungus (Capnodium spp.) growing on insect honeydew",
  "description": "Sooty mould is a black coating that grows on the sticky honeydew left  ...",
  "symptoms": [
    "Black, soot-like coating on the upper leaf surface",
    "..."
  ],
  "signs": [
    "Black layer that can be peeled or wiped off",
    "..."
  ],
  "causes": [
    "Honeydew from mango hoppers, scale insects or mealybugs",
    "..."
  ],
  "treatment": [
    "Control the sap-sucking insects first (hoppers, scales, mealybugs) with a labelled insecticide or neem oil.",
    "..."
  ],
  "prevention": [
    "Monitor and control hoppers during flowering.",
    "..."
  ],
  "predicted_label": "Mango___Sooty_Mould",
  "top_predictions": [
    {
      "confidence": 90.7,
      "label": "Mango___Sooty_Mould"
    },
    "..."
  ],
  "threshold": 75.0,
  "image_url": "/static/uploads/018bda40656540eaa6cf29aac92fa118.jpg"
}
```

- `local_name` is the local (Marathi) name for the Indian plants, `null` otherwise.
- `status` is `Healthy`, `Diseased` or `Uncertain`.
- Healthy leaves have empty `symptoms`, `causes` and `treatment`; `prevention` holds care tips.
- If confidence is below 75 % (`CONFIDENCE_THRESHOLD` in `config.py`), `status` is
  `Uncertain` and no disease name is given.

Error codes: `400` no image / wrong format / corrupted image, `413` file too large,
`422` no leaf detected in the photo,
`503` AI model not available, `500` prediction or database error.

### `GET /api/history`

```json
{ "success": true, "scans": [
  { "id": 12, "plant": "Tomato", "disease": "Early Blight", "status": "Diseased",
    "confidence": 94.5, "image_url": "/static/uploads/....jpg", "scan_date": "2026-11-10T21:40:05" }
]}
```

### `GET /api/history/<id>`

The complete stored result of one scan (same fields as `/api/predict`). `404` if not found.

### `DELETE /api/history/<id>`

Deletes the record and its image. Returns `{"success": true, "deleted_id": 12}`; `404` if not found.

### `GET /api/health`

```json
{ "success": true, "status": "running", "model_loaded": true, "model_classes": 55, "model_error": null }
```

---

## 7. Database

SQLite file `database/plantcare.db`, table `scan_history`:

| Column | Type | Notes |
|---|---|---|
| id | INTEGER | primary key |
| plant_name | TEXT | |
| disease_name | TEXT | |
| disease_status | TEXT | Healthy / Diseased / Uncertain |
| confidence | REAL | 0–100 |
| image_path | TEXT | file name in `static/uploads/` |
| scan_date | TEXT | ISO date-time |
| result_data | TEXT | full JSON result |

To clear all history, stop the app and delete `database/plantcare.db` and the files in `static/uploads/`.

---

## 8. How the AI works

1. A colour check confirms the photo contains leaf-like pixels (green, yellow, brown).
2. The image is converted to RGB and resized to the model's input size (from `model_info.json`).
3. Pixels go into the network as 0–255 values; a `Rescaling` layer **inside** the saved model maps
   them to −1…1 exactly as in training, so training and prediction preprocessing can never differ.
4. MobileNetV2 extracts visual features (spots, rings, colour, texture), and a softmax layer gives a
   probability for each of the 55 classes.
5. The top class is looked up in `backend/disease_info.py` for plant, disease and advice.

### Supported plants (18 plants, 54 conditions)

Tulsi, Amba (Mango), Dhan (Rice), Gahu (Wheat), Apple, Blueberry, Cherry, Corn (maize), Grape,
Orange, Peach, Bell Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato. The About page
lists every condition, generated from the loaded model's own class list.

An extra class, `Other___unsupported_plant`, was trained on leaves of other species (sweet basil,
mint, coriander, kale, lettuce, parsley). When the model picks it, the app answers "not a supported
plant" instead of guessing a disease.

### Limitations (worth mentioning in your report)

- Before the model runs, a simple colour check rejects photos with almost no leaf-coloured
  pixels (walls, sky, screens). It cannot catch everything: skin, wood or other green objects
  still reach the model, which then picks the most similar class.
- The model is a **classifier, not a general plant identifier**. Leaves of species it was never
  trained on are usually reported as uncertain, but not always: in testing, 4 of 18 such photos
  (e.g. a guava leaf) were confidently matched to a supported plant such as tomato.
- Mango and rice test photos come from the same collections as their training photos (same farms,
  cameras and backgrounds). Accuracy on your own photos will be lower than the test numbers.
- The Tulsi dataset photographs whole plants in pots. Several "unhealthy" photos have similar
  backgrounds, so the model may partly rely on the background; photograph the plant in good light.
- PlantVillage photos show a single leaf on a plain background. Field photos with soil, several
  leaves or strong shadows are harder; accuracy on such photos is lower than the validation accuracy.
- Treatment advice is general guidance. Always follow product labels and local agricultural advice.

---

## 9. Configuration (`config.py`)

| Setting | Default | Meaning |
|---|---|---|
| `CONFIDENCE_THRESHOLD` | 75.0 | below this, the result is "Uncertain" (chosen from the test results, see section 3) |
| `MIN_LEAF_PIXEL_RATIO` | 0.015 | minimum share of leaf-coloured pixels |
| `MAX_FILE_SIZE_MB` | 10 | upload limit |
| `PORT` | 5000 | server port (`python app.py --port 8000` also works) |
| `HOST` | 0.0.0.0 | lets phones on your network connect |

---

## 10. Troubleshooting

| Problem | Fix |
|---|---|
| `No module named tensorflow` | Activate the venv and `pip install -r requirements.txt`. Use Python 3.10–3.13. |
| "AI model not available" | Make sure both files are in `models/`, then restart the app. |
| Phone can't open the site | Same Wi-Fi? Allow Python in the firewall. Use the IP printed in the terminal. |
| Camera button opens the file picker | Normal over HTTP. Use `python app.py --https` for the live camera. |
| Port already in use | `python app.py --port 5001` |

### Dataset credits

| Dataset | Used for | Source |
|---|---|---|
| PlantVillage (Hughes & Salathé 2015), CC BY-SA 3.0 | 14 PlantVillage plants | github.com/spMohanty/PlantVillage-Dataset |
| MangoLeafBD (Ahmed et al. 2023, *Data in Brief*), CC BY 4.0 | Amba (Mango) | Mendeley 10.17632/hxsnvwty3r.1, copy in github.com/DevyanshMalhotra/MangoLeafDiseaseDetection |
| Rice Diseases Image Dataset (Huy Minh Do) | Dhan (Rice) | Kaggle, copy in github.com/jonathanrjpereira/Rice-Disease-Classification |
| Wheat Disease Detection images | Gahu (Wheat) | github.com/suhasmaddali/Wheat-Disease-Detection- |
| India's Most Famous Basil Plant Leaves Quality Dataset | Tulsi | copy in github.com/yassa-life/Basil_Leaf_health_check_dataset-_trained_model-evidence |
| Herb leaves (basil, mint, coriander, kale, lettuce, parsley) | "Unsupported plant" class | github.com/waqas-ali/tf_plant_disease_classification |
| A Database of Leaf Images (Chouhan et al. 2019), CC BY 4.0 | 18 test photos of unsupported plants | Mendeley 10.17632/hb74ynkjcn.1 |

Image counts per class, after de-duplication, are in `models/dataset_manifest.json`.
