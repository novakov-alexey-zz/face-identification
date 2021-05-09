import predict as pr
import cv2
import click

from pathlib import Path


@click.command()
@click.option("--image_path", help="path to image to test prediction for")
def test(image_path: str):
    model_path = Path("model")
    predictions = pr.predict_at_path(
        Path(image_path), model_path)

    for predicted_class, detected_face in predictions:
        print(f"detected class is {predicted_class}")

        if detected_face.img is not None:
            img_path = f"./test_images/found_faces/{predicted_class}.jpg"
            cv2.imwrite(
                img_path, detected_face.img)


if __name__ == '__main__':
    test()
