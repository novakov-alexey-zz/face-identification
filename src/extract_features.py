import numpy as np
import os
import glob
import pathlib as pl
import tensorflow as tf

from keras_vggface.vggface import VGGFace
from keras_vggface import utils
from keras.preprocessing import image
from pathlib import Path
from common import IMAGE_HEIGHT, IMAGE_WIDTH, save_pickle, COLOR_CHANNELS
from typing import List
from functools import reduce
from keras.engine.training import Model


def image2x(image_path):
    img = image.load_img(image_path, target_size=(IMAGE_HEIGHT, IMAGE_WIDTH))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = utils.preprocess_input(x, version=1)  # or version=2
    return x


def cal_mean_feature(image_folder: Path, model: Model):
    face_images = list(image_folder.glob('*.jpg'))

    def chunks(l, n):
        """Yield successive n-sized chunks from l."""
        for i in range(0, len(l), n):
            yield l[i:i + n]

    batch_size = 32
    face_images_chunks = chunks(face_images, batch_size)
    fvecs: List[np.ndarray] = list()
    for face_images_chunk in face_images_chunks:
        images = np.concatenate([image2x(face_image)
                                for face_image in face_images_chunk])
        batch_fvecs = model.predict(images)
        fvecs.append(batch_fvecs)

    fvecs = reduce(lambda a, b:  np.append(a, b, axis=0), fvecs)
    return np.array(fvecs).sum(axis=0) / len(fvecs)


def extract():
    FACE_IMAGES_FOLDER = Path("./dataset-family")
    model = VGGFace(model='vgg16', include_top=False, input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, COLOR_CHANNELS),
                    pooling='avg')
    model.summary()

    precompute_features = []
    labels = [x for x in FACE_IMAGES_FOLDER.iterdir() if x.is_dir()]
    for label in labels:
        print(
            f"Calculating features for '{label.name}' label in [{label}] folder")
        mean_features = cal_mean_feature(label, model)
        precompute_features.append(
            {"name": label.name, "features": mean_features})

    save_pickle(Path("./data/precompute_features.pickle"),
                precompute_features)


if __name__ == '__main__':
    extract()
