# models/

Put the trained model files here:

| File | Created by | Purpose |
|------|-----------|---------|
| `plant_disease_model.keras` | `train_model.py` | The trained MobileNetV2 classifier |
| `model_info.json` | `train_model.py` | Class labels in output order, input size, accuracy |

Both files are required. Without them the website still opens, but
`/api/predict` returns "AI model not available" instead of a result.

See **Step 3** of the main `README.md` for how to train or obtain the model.
