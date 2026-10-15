import tensorflow as tf
import requests
from PIL import Image
from io import BytesIO


def load_image(url="", local_path="", target_size=(128, 128)):
    """
    Load an image from a URL or local path.
    Returns a preprocessed numpy array ready for model prediction.
    """
    if url:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        img = Image.open(local_path).convert("RGB")

    img = img.resize(target_size)
    return tf.keras.utils.img_to_array(img)
