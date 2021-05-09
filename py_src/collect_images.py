import cv2
import os
from os.path import exists

import time

directory = "./dataset-family"

collected_dir = f"{directory}-collected"
if not exists(collected_dir):
    os.mkdir(collected_dir)

labels = os.listdir(directory)
image_count = 20
cap = cv2.VideoCapture(2)

time.sleep(2.5)

for label in labels:
    label_dir = os.path.join(collected_dir, label)

    if not exists(label_dir):
        os.mkdir(label_dir)

    for i in range(image_count):
        print(f"Collecting image {label}-{i}")
        cv2.waitKey(5000)
        _, frame = cap.read()
        image_path = os.path.join(
            label_dir, f"{label}-{i}.jpg")
        cv2.imwrite(image_path, frame)
        cv2.imshow("frame", frame)

        if cv2.waitKey(20) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
