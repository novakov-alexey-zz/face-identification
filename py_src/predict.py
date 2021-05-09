import os
from functools import reduce
from typing import Dict
from typing import List
from typing import Tuple
from pathlib import Path

import numpy as np
from tensorflow.keras.models import load_model
from keras.engine.training import Model

import face_detection_operation as fdo
from common import IMAGE_DATA_SCALING_FACTOR
from face_detection_operation import DetectedFace


def predict_by_img(img, model_path: Path, grey_img: bool = False) -> Tuple[List[Tuple[str, DetectedFace]], List[DetectedFace]]:
    detected = fdo.detect_faces_in_img(img)
    return predict(detected, model_path, grey_img), detected


def predict_at_path(image_path: Path, model_path: Path, grey_img: bool = False) -> List[Tuple[str, DetectedFace]]:
    detected = fdo.detect_faces(image_path, grey_img)
    return predict(detected, model_path, grey_img)


def load_classes(model_path: Path) -> Dict[int, str]:
    return np.load(os.path.join(
        model_path, "class_names.npy"), allow_pickle=True).item()


def predict(detected_face: List[DetectedFace], model_path: Path, grey_img: bool) -> List[Tuple[str, DetectedFace]]:
    model = load_model(model_path)
    classes = load_classes(model_path)

    return predict_by_model(detected_face, model, classes, grey_img)


def predict_by_model(detected_faces: List[DetectedFace], model: Model, classes: Dict[int, str], grey_img: bool) -> List[Tuple[str, DetectedFace]]:
    threshold = 0.5

    def expand_dims(img):
        return np.expand_dims(img, ((0, 3) if grey_img else 0))

    if not detected_faces:
        return []
    else:
        input_samples = [expand_dims(detected.face.astype(
            'float32')) * IMAGE_DATA_SCALING_FACTOR for detected in detected_faces]
        input_samples = reduce(
            lambda a, b: np.append(a, b, axis=0), input_samples)

        results = model.predict(input_samples)
        labels: List[str] = []

        for logits in results:
            index = np.argmax(logits, axis=0)
            index = -1 if logits[index] < threshold else index
            labels.append(classes.get(index, "unknown"))

        predictions = list(zip(labels, detected_faces))
        return predictions


class Predictor:

    def __init__(self, model_path: Path):
        self.model = load_model(model_path)
        self.classes = load_classes(model_path)

    def __call__(self, img, grey_img: bool) -> List[Tuple[str, DetectedFace]]:
        detected_faces = fdo.detect_faces_in_img(img)
        return predict_by_model(detected_faces, self.model, self.classes, grey_img)
