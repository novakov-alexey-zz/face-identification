import sys
import glob
import cv2
import numpy as np
import typing

from pathlib import Path
from os.path import join, exists
from os import mkdir, listdir
from dataclasses import dataclass
from mtcnn import MTCNN
from PIL import Image
from typing import List, Tuple
from common import IMAGE_HEIGHT, IMAGE_WIDTH, crop_img

FACE_DETECTION_CONFIDENCE = 0.95
detector: MTCNN = MTCNN()
frontal_cascade = cv2.CascadeClassifier(
    "./cv2_cascades/haarcascades/haarcascade_frontalface_alt_tree.xml")
frontal_alt_cascade = cv2.CascadeClassifier(
    "./cv2_cascades/haarcascades/haarcascade_frontalface_alt.xml")
frontal_alt2_cascade = cv2.CascadeClassifier(
    "./cv2_cascades/haarcascades/haarcascade_frontalface_alt2.xml")
profile_cascade = cv2.CascadeClassifier(
    "./cv2_cascades/haarcascades/haarcascade_profileface.xml")


@dataclass
class DetectedFace:
    face: np.ndarray
    img: typing.Any
    x: int
    y: int
    width: int
    height: int


def save_cropped_face(images_root_folder: Path,
                      required_size: Tuple[int, int] = (
                          IMAGE_HEIGHT, IMAGE_WIDTH),
                      cropped_folder: Path = Path('dataset'),
                      confidence: float = FACE_DETECTION_CONFIDENCE):

    if not exists(images_root_folder):
        return Exception(f"Input Images {images_root_folder} folder does not exist.")

    file_types = ["*.png", "*.PNG", "*.JPEG", "*.jpeg", "*.jpg", "*.JPG"]
    labels = listdir(images_root_folder)
    print(f"found labels: {labels}")

    if not exists(cropped_folder):
        mkdir(cropped_folder)

    for file_type in file_types:
        for label in labels:
            if not exists(join(cropped_folder, label)):
                mkdir(join(cropped_folder, label))

            for i, image_file in enumerate(glob.glob(
                    join(images_root_folder, label, file_type)
            )
            ):
                print(f"processing {image_file}")
                img = cv2.imread(image_file)
                results = filter_faces(detector, img, confidence)
                print(f"Found {len(results)} possible faces")

                for j, result in enumerate(results):
                    output_file_name = f"{label}_{i}_{j}{image_file[-4:]}"
                    file_path = join(cropped_folder, label, output_file_name)
                    save_as_image(result, img, required_size,
                                  file_path)


def filter_faces(detector: MTCNN, img, confidence: float) -> list:
    faces = detector.detect_faces(img)
    return [r for r in faces if r['confidence'] > confidence]


def save_as_image(detected_face, img, required_size, file_path):
    x, y, width, height = detected_face['box']
    face = img[y:y + height, x:x + width]
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = np.asarray(image)

    cv2.imwrite(
        file_path,
        face_array)


def detect_faces(filename: Path, gray_mode: bool = True) -> List[DetectedFace]:
    img = cv2.imread(f"{filename}")
    if gray_mode:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return detect_faces_in_img(img)


def detect_faces_in_img(img, required_size=(IMAGE_HEIGHT, IMAGE_WIDTH)) -> List[DetectedFace]:
    faces = [face['box'] for face in detector.detect_faces(img)]

    # faces = list(frontal_alt2_cascade.detectMultiScale(
    #     img, scaleFactor=1.5, minNeighbors=3))

    detected_faces: List[DetectedFace] = []

    for (x, y, width, height) in faces:
        face_array, face_img = crop_img(
            (x, y, width, height), img, required_size)
        detected_faces.append(DetectedFace(
            face_array, face_img, x, y, width, height))

    return detected_faces


if __name__ == "__main__":
    images_root_folder = Path(sys.argv[1])
    cropped_folder = Path(sys.argv[2])
    save_cropped_face(images_root_folder=images_root_folder,
                      cropped_folder=cropped_folder)
