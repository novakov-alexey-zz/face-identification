from keras_vggface.vggface import VGGFace
from common import IMAGE_HEIGHT, IMAGE_WIDTH, crop_img, load_pickle, COLOR_CHANNELS

if __name__ == '__main__':
    model = VGGFace(model='vgg16',
                    include_top=False,
                    input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, COLOR_CHANNELS),
                    pooling='avg')
    model.save("model_vggface")                    
