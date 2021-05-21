from keras_vggface.vggface import VGGFace
from common import IMAGE_HEIGHT, IMAGE_WIDTH, crop_img, load_pickle, COLOR_CHANNELS

import click
import tensorflow as tf


@click.command("tflite")
@click.option("--output-path", help="output path to save model at", default="model_vggface")
def tflite(output_path: str):
    model = VGGFace(model='vgg16',
                    include_top=False,
                    input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, COLOR_CHANNELS),
                    pooling='avg')
    filename = "./data/vgg_face.tflite"

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]
    converter.target_spec.supported_types = [tf.float16]
    converter.inference_input_type = tf.float16
    converter.inference_output_type = tf.float16
    tflite_model = converter.convert()

    with open(filename, "wb") as file:
        file.write(tflite_model)


@click.command("saved-model")
@click.option("--output-path", help="output path to save model at", default="model_vggface")
def saved_model(output_path: str):
    model = VGGFace(model='vgg16',
                    include_top=False,
                    input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, COLOR_CHANNELS),
                    pooling='avg')
    model.save(output_path)
    model.summary()


@click.group()
def cli():
    pass


cli.add_command(tflite)
cli.add_command(saved_model)

if __name__ == '__main__':
    cli()
