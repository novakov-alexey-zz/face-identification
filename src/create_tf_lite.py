from keras_vggface.vggface import VGGFace
from common import IMAGE_HEIGHT, IMAGE_WIDTH, crop_img, load_pickle, COLOR_CHANNELS

import tensorflow as tf


def store_as_tflite(filename, model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]
    converter.target_spec.supported_types = [tf.float16]
    converter.inference_input_type = tf.float16
    converter.inference_output_type = tf.float16
    tflite_model = converter.convert()

    with open(filename, "wb") as file:
        file.write(tflite_model)


if __name__ == '__main__':
    model = VGGFace(model='vgg16',
                    include_top=False,
                    input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, COLOR_CHANNELS),
                    pooling='avg')
    filename = "./data/vgg_face.tflite"
    store_as_tflite(filename, model)
