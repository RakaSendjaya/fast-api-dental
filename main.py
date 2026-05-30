from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import tensorflow as tf
from PIL import Image
import numpy as np
import io
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dentiscan-fastapi")

app = FastAPI(title="DentiScan AI Proxy")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model will be loaded from the repository root.
MODEL_PATH = "./model.keras"
MODEL = None
INPUT_SIZE = (224, 224)
CLASS_NAMES = ["gigi caries", "gigi sehat", "gusi sehat", "karang gigi"]


@app.on_event("startup")
def load_model():
    global MODEL
    try:
        logger.info("Loading model from %s", MODEL_PATH)
        MODEL = tf.keras.models.load_model(MODEL_PATH)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.warning("Could not load model: %s", e)
        MODEL = None


def preprocess_image(contents: bytes):
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img = img.resize(INPUT_SIZE)
        arr = np.asarray(img).astype("float32") / 255.0
        arr = np.expand_dims(arr, axis=0)
        return arr
    except Exception as e:
        raise ValueError(f"Invalid image data: {e}")


def build_prediction_payload(probabilities: np.ndarray):
    flat = np.asarray(probabilities).squeeze()

    if flat.ndim == 0:
        raise ValueError("Model output must contain 4 class probabilities")

    if flat.shape[-1] != 4:
        raise ValueError(f"Expected 4 output classes, got {flat.shape[-1]}")

    probs = flat.astype(float)
    total = float(np.sum(probs))

    if not np.isfinite(total) or total <= 0:
        raise ValueError("Invalid prediction probabilities")

    # If the model already outputs softmax probabilities, keep them.
    # Otherwise normalize to keep the response stable.
    if not (np.all(probs >= 0) and abs(total - 1.0) < 1e-3):
        probs = np.exp(probs - np.max(probs))
        probs = probs / np.sum(probs)

    top_idx = int(np.argmax(probs))
    confidence = float(probs[top_idx])

    all_predictions = {
        CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))
    }

    return {
        "label": CLASS_NAMES[top_idx],
        "confidence": confidence,
        "all_predictions": all_predictions,
    }


@app.post("/predict")
async def predict(image: UploadFile = File(...), debug: Optional[str] = None):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not available on server")

    contents = await image.read()
    try:
        input_arr = preprocess_image(contents)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        preds = MODEL.predict(input_arr)
        payload = build_prediction_payload(np.asarray(preds))

        return {"success": True, "data": payload}

    except Exception as e:
        logger.exception("Prediction failed: %s", e)
        raise HTTPException(status_code=500, detail="Prediction failed")
