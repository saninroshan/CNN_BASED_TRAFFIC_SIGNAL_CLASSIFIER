from fastapi import FastAPI, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import sys, os, io

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from image_loader import load_image
from cnn_model import predict, model, CLASS_NAMES
from PIL import Image
import numpy as np
import tensorflow as tf

app = FastAPI(title="Traffic Sign Classifier API")

# ── CORS ─────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Map backend class names → frontend display names ─────────────────────────
# The frontend's CLASSES array uses these exact names for matching emoji/color
CLASS_DISPLAY = {
    "No_Entry":        "No Entry",
    "No_Horn":         "No Horn",
    "No_Uturn":        "No U-Turn",
    "Speedlimit15kmh": "Speed Limit",
    "Stop_Sign":       "Stop Sign",
    "Uturn":           "U-Turn",
    "Zebra_Crossing":  "Pedestrian Crossing",
}

def _run_predict(img_array):
    """Run model inference and return (display_name, confidence, scores_dict)."""
    img_batch = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_batch, verbose=0)[0]          # shape (7,)

    class_index = int(np.argmax(predictions))
    raw_name    = CLASS_NAMES[class_index]
    confidence  = round(float(np.max(predictions)) * 100, 2)

    # Build full scores dict with display names
    scores = {
        CLASS_DISPLAY.get(cls, cls): round(float(predictions[i]) * 100, 2)
        for i, cls in enumerate(CLASS_NAMES)
    }

    display_name = CLASS_DISPLAY.get(raw_name, raw_name)
    return display_name, confidence, scores


# ── Serve Frontend ────────────────────────────────────────────────────────────
@app.get("/")
def root():
    index_path = os.path.join(os.path.dirname(__file__), "../frontend/index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"status": "Frontend not found"}


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "Traffic Sign Classifier API is running"}


# ── Unified /predict endpoint (handles both file upload and JSON URL) ─────────
@app.post("/predict")
async def predict_unified(request: Request, file: UploadFile = File(None)):
    """
    Accepts:
      - multipart/form-data with a 'file' field  → file upload
      - application/json with {"url": "..."}     → image URL
      - application/json with {"path": "..."}    → local path (server-side)

    Returns:
      { "prediction": str, "confidence": float, "scores": {class: float, ...} }
    """
    content_type = request.headers.get("content-type", "")

    # ── File upload ──
    if file is not None:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB").resize((128, 128))
        img_array = tf.keras.utils.img_to_array(img)
        display_name, confidence, scores = _run_predict(img_array)

    # ── JSON body (URL or path) ──
    elif "application/json" in content_type:
        body = await request.json()
        url  = body.get("url", "").strip()
        path = body.get("path", "").strip()

        if url:
            img_array = load_image(url=url)
        elif path:
            if not os.path.exists(path):
                return {"error": f"File not found: {path}"}
            img_array = load_image(local_path=path)
        else:
            return {"error": "Provide either 'url' or 'path' in the JSON body."}

        display_name, confidence, scores = _run_predict(img_array)

    else:
        return {"error": "Send a file upload or JSON body with 'url'/'path'."}

    return {
        "prediction": display_name,
        "confidence": round(confidence / 100, 4),   # frontend expects 0–1 float
        "scores":     {k: round(v / 100, 4) for k, v in scores.items()},
        # Also include raw fields for backward-compat with old frontend
        "predicted_class": display_name,
    }


# ── Legacy routes (kept for backward compatibility) ───────────────────────────
@app.post("/predict/file")
async def predict_from_file(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB").resize((128, 128))
    img_array = tf.keras.utils.img_to_array(img)
    display_name, confidence, scores = _run_predict(img_array)
    return {"predicted_class": display_name, "confidence": confidence}


@app.post("/predict/url")
async def predict_from_url(data: dict):
    url = data.get("url", "")
    if not url:
        return {"error": "No URL provided"}
    img_array = load_image(url=url)
    display_name, confidence, scores = _run_predict(img_array)
    return {"predicted_class": display_name, "confidence": confidence}


# ── Static files (CSS, JS for our custom frontend) ────────────────────────────
frontend_dir = os.path.join(os.path.dirname(__file__), "../frontend")
if os.path.isdir(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")