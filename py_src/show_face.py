import cv2
import dataclasses

from predict import Predictor
from pathlib import Path

cap = cv2.VideoCapture(0)
predictor = Predictor(model_path=Path("model"))

while True:
    _, frame = cap.read()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    predictions = predictor(rgb, grey_img = False)
    if not predictions:
        pass
    else:
        for predicted_class, face in predictions:
            _, _, x, y, w, h = dataclasses.astuple(face)
            color = (10, 255, 0)  # BGR
            stroke = 2
            end_x = x + w
            end_y = y + h
            text_color = (10, 255, 255)
            cv2.rectangle(frame, (x, y), (end_x, end_y), color)
            cv2.putText(frame, predicted_class, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        1, text_color, stroke, cv2.LINE_AA)
    cv2.imshow("Faces", frame)

    if cv2.waitKey(20) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
