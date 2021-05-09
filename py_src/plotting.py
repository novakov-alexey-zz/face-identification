import face_recognition as fr
import seaborn as sns
import numpy as np

from sklearn.metrics import confusion_matrix
from pathlib import Path
from matplotlib import pyplot as plt
import os


def plot_confusion_matrix(model_path: Path, dataset_dir: Path):
    model = fr.load_saved_model(model_path)

    df = fr.read_to_dataframe(dataset_dir)
    validation_gen = fr.dataframe_validation_gen(df)
    true_classes = validation_gen.classes
    class_names = validation_gen.class_indices.keys()
    fig, (ax1) = plt.subplots(1, 1, figsize=(20, 10))

    scratch_preds = model.predict(validation_gen)
    scratch_pred_classes = np.argmax(scratch_preds, axis=1)

    plot_heatmap(true_classes, scratch_pred_classes,
                 class_names, ax1, "Custom CNN")


def plot_heatmap(y_true, y_pred, class_names, ax, title):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(
        cm,
        annot=True,
        square=True,
        xticklabels=class_names,
        yticklabels=class_names,
        fmt='d',
        cmap=plt.cm.Blues,
        cbar=False,
        ax=ax
    )
    ax.set_title(title, fontsize=16)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    plot_folder = "plot"
    plt.savefig(os.path.join(plot_folder, f"heatmap.png"))
    plt.clf()


if __name__ == '__main__':
    path = Path("model")
    dataset_dir = Path("./dataset-family")

    plot_confusion_matrix(path, dataset_dir)
