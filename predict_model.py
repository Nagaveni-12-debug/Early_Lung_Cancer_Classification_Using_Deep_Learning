# predict_model.py
"""
Prediction module for Lung Cancer Detection.
Fixes:
- Correct class order based on folder names
- Correct cancer vs normal index mapping
- Correct probability voting
"""

import os
import logging
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("predict_model")

# -------------------------
# Model path
# -------------------------
MODEL_PATH = os.path.join("models", "best_model.h5")

model = None
try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.exception("Model loading failed: %s", e)
    model = None

IMG_SIZE = (224, 224)

# ---------------------------------
# CORRECT CLASS ORDER (alphabetical)
# ---------------------------------
# train / test / valid folder names:
# adenocarcinoma, large.cell.carcinoma, normal, squamous.cell.carcinoma
CLASS_NAMES = [
    "adenocarcinoma",          # index 0
    "large.cell.carcinoma",    # index 1
    "normal",                  # index 2  <-- NORMAL HERE
    "squamous.cell.carcinoma"  # index 3
]


def default_response(message="Invalid image"):
    return {
        "result": "Error",
        "subtype": None,
        "message": message,
        "confidence": 0.0,
        "probs": {c: 0.0 for c in CLASS_NAMES},
        "tumor_location": None,
        "tumor_size": 0.0
    }


# -----------------------------------------
# READ IMAGE
# -----------------------------------------
def read_image_any(path):
    try:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            return None
        return img
    except:
        return None


# -----------------------------------------
# VALIDATION CHECK
# -----------------------------------------
def is_valid_scan(img):
    if img is None:
        return False
    h, w = img.shape[:2]
    if h < 64 or w < 64:
        return False
    return True


# -----------------------------------------
# PREPROCESS
# -----------------------------------------
def preprocess_image_for_model(img):
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype(np.float32)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return img


# -----------------------------------------
# SOFTMAX
# -----------------------------------------
def _apply_softmax(arr):
    arr = np.array(arr, dtype=np.float64)
    arr -= np.max(arr)
    exp = np.exp(arr)
    return exp / np.sum(exp)


# -----------------------------------------
# MAIN PREDICTION FUNCTION
# -----------------------------------------
def predict_image(path):
    try:
        if model is None:
            return default_response("Model not loaded.")

        img = read_image_any(path)
        if not is_valid_scan(img):
            return default_response("Invalid or unreadable scan.")

        x = preprocess_image_for_model(img)

        preds = model.predict(x)
        p = np.array(preds).squeeze()

        # Convert logits to softmax probabilities
        if p.ndim == 1 and p.size == 4:
            if not np.isclose(np.sum(p), 1.0, atol=1e-3):
                per_class = _apply_softmax(p)
            else:
                per_class = p
        else:
            return default_response("Model output shape incorrect.")

        # Normalize
        per_class = per_class / per_class.sum()

        # Build probability dictionary
        probs = {
            CLASS_NAMES[i]: round(float(per_class[i] * 100), 2)
            for i in range(len(CLASS_NAMES))
        }

        # -------------------------------
        # FIXED CANCER VS NORMAL LOGIC
        # -------------------------------
        cancer_prob = float(per_class[0] + per_class[1] + per_class[3])  # 0,1,3 = cancer classes
        normal_prob = float(per_class[2])                                # index 2 = NORMAL

        is_cancer = cancer_prob > normal_prob
        confidence = round(max(cancer_prob, normal_prob) * 100, 2)

        if is_cancer:
            # among cancer classes pick subtype
            subtype_idx = np.argmax([per_class[0], per_class[1], per_class[3]])
            subtype = [CLASS_NAMES[0], CLASS_NAMES[1], CLASS_NAMES[3]][subtype_idx]
            result_text = "Cancer"
        else:
            subtype = None
            result_text = "Healthy"

        return {
            "result": result_text,
            "subtype": subtype,
            "message": "OK",
            "confidence": confidence,
            "probs": probs,
            "tumor_location": None,
            "tumor_size": 0.0
        }

    except Exception as e:
        logger.exception("Prediction error: %s", e)
        return default_response(str(e))
