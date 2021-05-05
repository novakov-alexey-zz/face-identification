import sys
import face_recognition as fr
import click

from model import get_model
from common import bool_or_fail
from pathlib import Path
from face_recognition import TransferModel


def build_model() -> TransferModel:
    model = get_model(image_channels=3)
    return TransferModel(model, model)


# @click.version_option(__version__)
@click.command()
@click.option("--epochs", help="number of training epochs", default=50)
@click.option("--parallel_folds", help="whether to run cross-validation folds in parallel or not", is_flag=True)
def train(epochs, parallel_folds):
    path = Path("model")
    dataset_dir = Path("./dataset-family")

    fr.train(epochs=epochs, dataset_dir=dataset_dir, build_model=build_model,
             save_at_path=path, parallel_folds=parallel_folds)
    print(f"Model saved at: {path}")    


if __name__ == '__main__':
    train()
