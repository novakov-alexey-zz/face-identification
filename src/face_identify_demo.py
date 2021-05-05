from keras.engine import Model
from keras import models
from keras import layers
from keras.layers import Input
from keras.preprocessing import image
from keras_vggface.vggface import VGGFace
from common import IMAGE_HEIGHT, IMAGE_WIDTH, crop_img, load_pickle
from keras_vggface import utils
from pathlib import Path

import numpy as np
import scipy.spatial as spatial
import cv2
import os
import glob
import pickle
import time


class FaceIdentify(object):
    """
    Singleton class for real time face identification
    """
    CV2_CASCADE_FRONTAL = "./cv2_cascades/haarcascades/haarcascade_frontalface_alt.xml"
    CV2_CASCADE_PROFILE = "./cv2_cascades/haarcascades/haarcascade_profileface.xml"

    def __init__(self, precompute_features_file: Path):
        self.face_height = IMAGE_HEIGHT
        self.face_width = IMAGE_WIDTH
        self.color_channels = 3
        self.precompute_features_map = load_pickle(precompute_features_file)
        print(f"shape: {self.precompute_features_map[0]['features'].shape}")
        print("[_] Loading VGG Face model...")
        self.model = VGGFace(model='vgg16',
                             include_top=False,
                             input_shape=(self.face_height,
                                          self.face_width, self.color_channels),
                             pooling='avg')
        print("[x] Loading VGG Face model done")

    @classmethod
    def draw_label(cls, image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
                   font_scale=1, thickness=2):
        size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        x, y = point
        cv2.rectangle(
            image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
        cv2.putText(image, label, point, font, font_scale,
                    (255, 255, 255), thickness)

    def identify_face(self, features, threshold=100):
        distances = []
        for person in self.precompute_features_map:
            person_features = person.get("features")
            distance = spatial.distance.euclidean(person_features, features)
            distances.append(distance)
        min_distance_value = min(distances)
        min_distance_index = distances.index(min_distance_value)
        if min_distance_value < threshold:
            return self.precompute_features_map[min_distance_index].get("name")
        else:
            return "?"

    def detect_face(self):
        frontal_face_cascade = cv2.CascadeClassifier(self.CV2_CASCADE_FRONTAL)
        profile_face_cascade = cv2.CascadeClassifier(self.CV2_CASCADE_PROFILE)
        video_capture = cv2.VideoCapture(0)  # defaul video device
        count = 0
        duration = 0

        while True:
            _, img = video_capture.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            start_time = time.time()
            faces = list(frontal_face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5
            ))
            # faces = faces + list(profile_face_cascade.detectMultiScale(
            #     gray,
            #     scaleFactor=1.2,
            #     minNeighbors=5
            # ))
            duration = duration + time.time() - start_time
            count = count + 1
            face_imgs = np.empty(
                (len(faces), self.face_height, self.face_width, self.color_channels))

            for i, face in enumerate(faces):
                face_array, _ = crop_img(
                    face, img, (self.face_height, self.face_width))
                (x1, y1, x2, y2) = box_with_margin(img, face)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 200, 0), 2)
                face_imgs[i, :, :, :] = face_array

            if len(face_imgs) > 0:
                features_faces = self.model.predict(face_imgs)
                predicted_names = [self.identify_face(
                    features_face) for features_face in features_faces]

            # draw results
            for i, (x, y, _, _) in enumerate(faces):
                label = str(predicted_names[i])
                self.draw_label(img, (x, y), label)

            cv2.imshow('Faces', img)

            if cv2.waitKey(5) == 27:  # ESC key press
                break

        video_capture.release()
        cv2.destroyAllWindows()
        print(f"{duration}")
        print(f"{count}")
        print(f"{duration / count}")


def box_with_margin(img, box, margin: int = 20):
    (x, y, w, h) = box
    x_left = max(x - margin, 0)
    x_right = min(img.shape[1], x + w + margin)
    y_top = max(y - margin, 0)
    y_bottom = min(img.shape[0], y + h + margin)

    return (x_left, y_top, x_right, y_bottom)


def identity_faces():
    face = FaceIdentify(
        precompute_features_file="./data/precompute_features.pickle")
    face.detect_face()


if __name__ == "__main__":
    identity_faces()
