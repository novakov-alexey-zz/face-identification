import sys
import face_recognition as fr
import keras
import click

from keras_vggface.vggface import VGGFace
from keras_vggface import utils
from face_recognition import TrainParams, TransferModel
from tensorflow.python.keras.layers.core import Dense, Flatten, Dropout
from tensorflow.keras import regularizers
from common import IMAGE_HEIGHT, IMAGE_WIDTH, bool_or_fail
from pathlib import Path

vgg_model_arch = "resnet50"


def get_model() -> TransferModel:
    base_model = VGGFace(
        include_top=False, model=vgg_model_arch, pooling='avg')
    base_model.trainable = False
    
    inputs = base_model.layers[0].input
    classes = 3

    x = base_model(inputs, training=False)
    x = Flatten(name='flatten')(x)
    x = Dense(256, activation='relu',kernel_regularizer=regularizers.l2(0.),
                    activity_regularizer=regularizers.l2(0.))(x)
    x = Dense(128, activation='relu',kernel_regularizer=regularizers.l2(0.),
                    activity_regularizer=regularizers.l2(0.))(x)
    x = Dropout(0.05)(x)
    outputs = Dense(classes, activation='softmax', name='classifier')(x)
    final_model = keras.Model(inputs, outputs)

    return TransferModel(base_model, final_model)


@click.command()
@click.option("--epochs", help="number of training epochs", default=50)
@click.option("--parallel_folds", help="whether to run cross-validation folds in parallel or not", is_flag=True)
def train(epochs: int, parallel_folds: bool):
    dataset_dir = Path("./dataset-family")

    # Train base model
    model_path = Path("model")
    best_fold = fr.train(epochs=epochs, build_model=get_model, dataset_dir=dataset_dir,
                         save_at_path=model_path, parallel_folds=parallel_folds,
                         preprocessing_func=utils.preprocess_input)
    print(f"model saved at: {model_path}")

    # Train entire model
    model = fr.load_saved_model(model_path)
    model.get_layer(f'vggface_{vgg_model_arch}').trainable = True

    dataset_df = fr.read_to_dataframe(dataset_dir)
    params = TrainParams(epochs, lr=1e-5, dataset_dir=dataset_dir, dataset_df=dataset_df,
                         preprocessing_func=utils.preprocess_input, build_model=lambda: TransferModel(model, model))
    fold_res = fr.train_fold(fold_index=best_fold.index,
                             fold_indicies=best_fold.dataset_indices,
                             params=params)
    print(f"val_accuracy: {fold_res.val_accuracy}")
    print(f"model saved at: {fold_res.model_save_path}")


if __name__ == '__main__':
    train()
