"""
Flask serving app for the Fake Currency Detection model.

Accuracy depends on matching the TRAINING preprocessing exactly:
the optimized notebook used MobileNetV2 `preprocess_input` (scales pixels to [-1, 1]),
NOT a simple /255 rescale. This app uses the same function, so predictions are valid.
"""

import os
import io
import json
import base64

import numpy as np
from flask import Flask, request, render_template, jsonify
from PIL import Image
from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ----------------------------------------------------------------------
# Config — MUST match how the model was trained
# ----------------------------------------------------------------------
IMG_SIZE = (224, 224)
MODEL_CANDIDATES = ["model.keras", "currency_classifier_best.keras", "model.h5"]
DEFAULT_CLASS_NAMES = ["fake", "real"]   # alphabetical -> matches training class_indices

app = Flask(__name__)


def _load_class_names():
    if os.path.exists("class_names.json"):
        with open("class_names.json") as f:
            return json.load(f)
    return DEFAULT_CLASS_NAMES


def _find_model_path():
    for name in MODEL_CANDIDATES:
        if os.path.exists(name):
            return name
    raise FileNotFoundError(
        "No model file found. Place 'model.keras' (or 'model.h5') in the app root. "
        f"Looked for: {MODEL_CANDIDATES}. "
        "Export it from the optimized notebook's 'Export for deployment' cell."
    )


CLASS_NAMES = _load_class_names()
MODEL_PATH = _find_model_path()
print(f"Loading model from {MODEL_PATH} ...", flush=True)
model = keras.models.load_model(MODEL_PATH)
print(f"Model loaded. Classes: {CLASS_NAMES}", flush=True)


def preprocess(pil_img):
    img = pil_img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype="float32")
    arr = preprocess_input(arr)          # same preprocessing as training -> range [-1, 1]
    return np.expand_dims(arr, axis=0)


def predict(pil_img):
    x = preprocess(pil_img)
    preds = np.array(model.predict(x, verbose=0)).ravel()
    if preds.size == 1:                  # sigmoid binary head (optimized notebook)
        p_pos = float(preds[0])          # P(class == index 1)
        idx = int(p_pos >= 0.5)
        confidence = p_pos if idx == 1 else 1.0 - p_pos
    else:                                # softmax head (original notebook)
        idx = int(np.argmax(preds))
        confidence = float(preds[idx])
    label = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else str(idx)
    return label, round(confidence * 100, 2)


@app.route("/health")
def health():
    return jsonify(status="ok", model=os.path.basename(MODEL_PATH), classes=CLASS_NAMES)


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        file = request.files.get("image")
        if not file or file.filename == "":
            return render_template("index.html", error="Please choose an image file first.")
        try:
            raw = file.read()
            pil = Image.open(io.BytesIO(raw))
            label, confidence = predict(pil)
            b64 = base64.b64encode(raw).decode("utf-8")
            return render_template("index.html", label=label,
                                   confidence=confidence, image_data=b64)
        except Exception as exc:
            return render_template("index.html", error=f"Could not process image: {exc}")
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_api():
    """JSON API: curl -F image=@note.jpg http://localhost:5000/predict"""
    file = request.files.get("image")
    if not file:
        return jsonify(error="no image provided (field name must be 'image')"), 400
    try:
        pil = Image.open(io.BytesIO(file.read()))
        label, confidence = predict(pil)
        return jsonify(label=label, confidence=confidence)
    except Exception as exc:
        return jsonify(error=str(exc)), 400


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
