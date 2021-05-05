import os
import shutil
import typing
from dataclasses import dataclass
from typing import Optional, Tuple, Callable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ray

from keras.callbacks import Callback
from keras.engine.training import Model
from sklearn.model_selection import KFold
from tensorflow.keras import optimizers
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from common import IMAGE_DATA_SCALING_FACTOR, IMAGE_WIDTH, IMAGE_HEIGHT
from pathlib import Path

EPOCHS = 50
BATCH_SIZE = 32
COLOR_MODE = 'rgb'
folds_dir = Path("model_folds")


@dataclass
class TransferModel:
    base_model: Model
    final_model: Model


@dataclass
class FoldResult:
    val_accuracy: float
    index: int
    dataset_indices: Tuple[np.ndarray, np.ndarray]
    model_save_path: Path


@dataclass
class TrainParams:
    epochs: int
    lr: float
    dataset_dir: Path
    dataset_df: pd.DataFrame
    preprocessing_func: Optional[Callable[[np.ndarray], np.ndarray]]
    build_model: Callable[..., TransferModel]


def absolute_file_paths(directory: Path):
    for dir_path, _, filenames in os.walk(directory):
        for f in filenames:
            yield os.path.abspath(os.path.join(dir_path, f))


def read_to_dataframe(training_data_dir: Path) -> pd.DataFrame:
    files = map(lambda path: (path, path.split("/")[-2:-1]),
                absolute_file_paths(training_data_dir))
    return pd.DataFrame(files, columns=["path", "label"])


def dataframe_training_gen(df: pd.DataFrame,
                           preprocessing_func: Optional[Callable[[np.ndarray], np.ndarray]] = None):
    return ImageDataGenerator(
        rescale=IMAGE_DATA_SCALING_FACTOR,
        horizontal_flip=True,
        vertical_flip=False,
        fill_mode="nearest",
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        rotation_range=10,
        preprocessing_function=preprocessing_func
    ).flow_from_dataframe(
        dataframe=df,
        x_col="path",
        y_col="label",
        target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        color_mode=COLOR_MODE
    )


def dataframe_validation_gen(df: pd.DataFrame,
                             preprocessing_func: Optional[Callable[[np.ndarray], np.ndarray]] = None):
    return ImageDataGenerator(
        rescale=IMAGE_DATA_SCALING_FACTOR,
        preprocessing_function=preprocessing_func
    ).flow_from_dataframe(
        df,
        x_col="path",
        y_col="label",
        target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
        class_mode='categorical',
        color_mode=COLOR_MODE
    )


def plot_training(history, index: int):
    plot_folder = "plot"
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.1, 1])
    plt.legend(loc='lower right')

    if not os.path.exists(plot_folder):
        os.mkdir(plot_folder)

    plt.savefig(os.path.join(plot_folder, f"model_accuracy_{index}.png"))
    plt.clf()


def get_callbacks(patience_lr: int) -> List[Callback]:
    early_stop = EarlyStopping(monitor='val_loss',
                           patience=50,
                           restore_best_weights=True,
                           mode='min')
    reduce_lr_loss = ReduceLROnPlateau(
        monitor='loss', factor=0.1, patience=patience_lr, verbose=1, min_delta=1e-4, mode='min')
    return [reduce_lr_loss, early_stop]


def fold_path(i: int) -> Path:
    return folds_dir/f"fold_{i}"


@ray.remote
def train_fold_remotely(fold_index: int, fold_indicies, params: TrainParams) -> FoldResult:
    return train_fold(fold_index, fold_indicies, params)


def train_fold(fold_index: int, fold_indicies: Tuple[np.ndarray, np.ndarray], params: TrainParams) -> FoldResult:
    print(f"Fold index: {fold_index}")
    train_index, test_index = fold_indicies
    train_df = params.dataset_df.iloc[train_index]
    validation_df = params.dataset_df.iloc[test_index]

    training_gen = dataframe_training_gen(
        train_df, params.preprocessing_func)
    validation_gen = dataframe_validation_gen(
        validation_df, params.preprocessing_func)

    callbacks = get_callbacks(patience_lr=5)

    new_model = params.build_model()

    model = new_model.final_model
    model.compile(
        loss=categorical_crossentropy,
        optimizer=optimizers.Adam(
            lr=params.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
        metrics=["accuracy"]
    )
    model.summary()

    train_image_count = len(train_index)
    validation_image_count = len(test_index)
    history = model.fit(
        x=training_gen,
        steps_per_epoch=train_image_count // BATCH_SIZE,
        epochs=params.epochs,
        validation_data=validation_gen,
        callbacks=callbacks,
        shuffle=True,
        validation_steps=validation_image_count // BATCH_SIZE
    )
    plot_training(history, fold_index)

    last_val_accuracy: float = history.history['val_accuracy'][-1]
    print(
        f"\nFold [{fold_index}] validation accuracy: {last_val_accuracy}")
    save_path = fold_path(fold_index)
    save_model(model, save_path, params.dataset_dir)

    return FoldResult(last_val_accuracy, fold_index, fold_indicies, save_path)


def train(dataset_dir: Path,
          build_model: Callable[..., TransferModel],
          epochs: int = EPOCHS,
          save_at_path: Path = Path("model"),
          parallel_folds: bool = False,
          lr: float = 0.001,
          preprocessing_func: Optional[Callable[[np.ndarray], np.ndarray]] = None) -> FoldResult:
    dataset_df = read_to_dataframe(dataset_dir)
    params = TrainParams(epochs, lr, dataset_dir,
                         dataset_df, preprocessing_func, build_model)
    # number of splits to be parameterized
    folds_count = 6
    kf = KFold(n_splits=folds_count, shuffle=True)

    if parallel_folds:
        ray.init()
        fut_val_accuracies = [train_fold_remotely.remote(
            i, indices, params) for i, indices in enumerate(kf.split(dataset_df))]
        val_accuracies: List[FoldResult] = ray.get(fut_val_accuracies)
    else:
        val_accuracies = [train_fold(
            i, indices, params) for i, indices in enumerate(kf.split(dataset_df))]

    best_fold = max(
        val_accuracies, key=lambda a: a.val_accuracy)

    print(f"max fold val accuracy: {best_fold.val_accuracy}")
    print(f"max validation is on fold: {best_fold.index}")

    shutil.move(f"{fold_path(best_fold.index)}", f"{save_at_path}")
    shutil.rmtree(folds_dir)

    return best_fold


def save_model(model, model_path: Path, dataset_dir: Path):
    model.save(model_path)

    df = read_to_dataframe(dataset_dir)
    training_gen = dataframe_training_gen(df)

    class_indices = training_gen.class_indices
    class_names_file_reverse = "class_names_reverse.npy"

    np.save(os.path.join(model_path, class_names_file_reverse), class_indices)

    class_names_reversed = np.load(os.path.join(
        model_path, class_names_file_reverse), allow_pickle=True).item()
    class_names = dict([(value, key)
                        for key, value in class_names_reversed.items()])
    print(f"class names: {class_names}")
    np.save(os.path.join(model_path, "class_names.npy"), class_names)


def load_saved_model(model_path) -> Model:
    return load_model(model_path)
