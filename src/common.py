import numpy as np
import pickle

from PIL import Image
from pathlib import Path
from typing import Tuple

IMAGE_DATA_SCALING_FACTOR = 1./255
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
COLOR_CHANNELS = 3


def bool_or_fail(arg):
    ua = str(arg).upper()
    if 'TRUE'.startswith(ua):
        return True
    elif 'FALSE'.startswith(ua):
        return False
    else:
        raise IOError(f"argument is not a boolean type: {arg}")


def crop_img(box: Tuple[int, int, int, int], img, required_size):
    x, y, width, height = box
    y_right = min(y + height, img.shape[0])
    x_bottom = min(x + width, img.shape[1])

    cropped_img = img[max(y, 0):y_right,
                      max(x, 0):x_bottom]

    image = Image.fromarray(cropped_img)
    resized = image.resize(required_size)
    as_array = np.asarray(resized)

    return as_array, cropped_img


def load_pickle(filename: Path):
    with open(filename, "rb") as file:
        return pickle.load(file)


def save_pickle(filename: Path, data):
    filename.parent.mkdir(exist_ok=True, parents=True)
    with open(filename, "wb") as file:
        pickle.dump(data, file)
