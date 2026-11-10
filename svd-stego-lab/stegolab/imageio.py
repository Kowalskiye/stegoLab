"""Image loading/saving helpers (always RGB uint8, always lossless PNG output)."""
import numpy as np
from PIL import Image


def load_image(path) -> np.ndarray:
    img = Image.open(path)
    return np.asarray(img.convert("RGB"), dtype=np.uint8).copy()


def save_png(arr, path):
    path = str(path)
    if not path.lower().endswith(".png"):
        raise ValueError("Output must be a .png file (JPEG is lossy and destroys the hidden data)")
    Image.fromarray(arr).save(path, format="PNG")


def load_pil(path) -> Image.Image:
    return Image.open(path)
