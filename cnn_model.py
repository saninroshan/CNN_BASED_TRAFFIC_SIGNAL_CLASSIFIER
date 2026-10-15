import tensorflow as tf
import numpy as np

# ── class names from your trained model ──────────────────────────────────────
CLASS_NAMES = [
    'No_Entry',
    'No_Horn',
    'No_Uturn',
    'Speedlimit15kmh',
    'Stop_Sign',
    'Uturn',
    'Zebra_Crossing'
]

# ── load model once when this file is imported ───────────────────────────────
MODEL_PATH = r"C:\SANIN\Internship\Project\traffic_classifier.keras"
model = tf.keras.models.load_model(MODEL_PATH)
print(f"[cnn_model] Model loaded from: {MODEL_PATH}")


def predict(img_array):
    """
    Takes a raw img_array (H, W, 3) — no normalization needed,
    model's Rescaling layer handles it.
    Returns: (class_name, confidence_percent)
    """
    img_batch = np.expand_dims(img_array, axis=0)       # (1, 128, 128, 3)
    predictions = model.predict(img_batch, verbose=0)

    class_index = np.argmax(predictions)
    class_name  = CLASS_NAMES[class_index]
    confidence  = round(float(np.max(predictions)) * 100, 2)

    return class_name, confidence
