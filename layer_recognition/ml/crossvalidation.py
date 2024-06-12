import argparse
import logging
import os
import pathlib
from typing import Any

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from matplotlib.axes import Axes
from sklearn import metrics
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedKFold
from sklearn.neighbors import KNeighborsClassifier

from utils import (
    clean_predictions,
    get_image_files,
    image_to_df,
    plot_crossval_metrics,
    plot_overall_metrics,
)

logger = logging.getLogger(__name__)


def get_parser() -> argparse.ArgumentParser:
    """Get parser for command line arguments."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "train_dir",
        type=pathlib.Path,
        help="Directory containing training data files.",
    )
    parser.add_argument(
        "save_dir",
        type=pathlib.Path,
        help="Directory where to save the images and the model.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Number of splits for the k-fold cross-validation.",
    )
    parser.add_argument(
        "--k-repeat",
        type=int,
        default=1,
        help="Number of repetition of the k-fold cross-validation.",
    )
    parser.add_argument(
        "--distinguishable-second-layer",
        "-d",
        action="store_true",
        help="Treats layer 2 and 3 as separate layers.",
    )
    parser.add_argument(
        "--extension",
        type=str,
        default="csv",
        choices=["txt", "csv"],
        help="extension of the files containing the data.",
    )
    parser.add_argument(
        "--estimators",
        "-e",
        type=int,
        default=100,
        help="Number of estimators for the random forest model.",
    )
    parser.add_argument(
        "--neighbors",
        "-n",
        type=int,
        help="Number of neighbors for KNN.",
        default=32,
    )
    parser.add_argument(
        "--clean",
        "-c",
        action="store_true",
        help=(
            "Use post-processing to attempt removing wrongly classified cluster. Can"
            " potentially harm the prediction's quality."
        ),
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=0.05,
        help="Radius of the circle used in the DBSCAN algorithm for post-processing.",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=3,
        help=(
            "Minimum number of neighbors required within eps to be treated as a central"
            " point."
        ),
    )
    parser.add_argument(
        "--gt-column",
        type=str,
        default="Expert_layer",
        help="Name of the ground truth column in the CSVs.",
    )
    parser.add_argument(
        "--train-glob",
        type=str,
        default="*.csv",
        help="Glob pattern to match the training files.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Control verbosity.",
    )
    return parser


def cross_validate(
    train_dir: pathlib.Path,
    image_names: list[str],
    features: list[str],
    classes: list[str],
    model: BaseEstimator,
    model_name: str,
    ax_metrics: Axes,
    ax_confusion: Axes,
    subplot_names: list[str],
    k: int = 10,
    k_repeat: int = 1,
    clean: bool = False,
    eps: float = 0.05,
    min_samples: int = 3,
    distinguishable_second_layer: bool = True,
    extension: str = "csv",
    gt_column: str = "Expert_layer",
) -> list[np.ndarray[float, Any]]:
    """Cross validate the random forest model.

    Parameters
    ----------
    train_dir
        Path to the directory containing the training data.
    image_names
        List of image filenames.
    features
        List of data feature kept.
    classes
        List of potential classes the model can classify to.
    get_model
        Callable that returns a new trainable model for cv.
    model_name
        Display name of a trainable model type.
    ax_metrics
        matplotlib axis object for metrics plot.
    ax_confusion
        matplotlib axis object for confusion plot.
    subplot_names
        List of names for each subplot.
    k
        Number of splits to do for cross validation.
    k_repeat
        Number of times to repeat the k splits.
    clean
        Whether to apply post processing cleaning
    eps
        Radius of the circle used in DBSCAN when doing post processing.
    min_samples
        Minimum number of samples to find to define a central point in DBSCAN.
    distinguishable_second_layer
        Whether second and third layer should be distinguishable.
    extension
        Extension of the files containing the data. Either csv or txt.
    gt_column
        Name of the column containing the ground truth.
    Returns
    -------
    List of metrics.
    """
    logger.info(f"Cross-validating with {model_name}.")
    _, _, df = image_to_df(
        image_names=image_names,
        data_path=str(train_dir),
        classes=classes,
        features=features,
        filter=True,
        distinguish_second_layer=distinguishable_second_layer,
        extension=extension,
        gt_column=gt_column,
    )
    kfold = RepeatedKFold(n_splits=k, n_repeats=k_repeat, random_state=42)
    scores = []
    confusion_matrices = []
    image_names = np.array(image_names)  # type: ignore

    for train_images, test_images in tqdm.tqdm(
        kfold.split(image_names[:, np.newaxis]), total=k  # type: ignore
    ):
        # Split df in train and test.
        train_images, test_images = image_names[train_images], image_names[test_images]
        train_df = df[df["image_id"].isin(train_images)]
        test_df = df[df["image_id"].isin(test_images)]

        # Make the corresponding cells ready for training.
        X_train, y_train = train_df[features], train_df[gt_column]
        X_test, y_test = test_df[features], test_df[gt_column]

        model.fit(X_train, y_train)
        if clean:
            y_pred = clean_predictions(
                test_df,
                model.predict(X_test),
                eps=eps,
                min_samples=min_samples,
                gt_column=gt_column,
            )
        else:
            y_pred = model.predict(X_test)
        class_scores = metrics.precision_recall_fscore_support(
            y_test, y_pred, average=None, labels=classes
        )
        scores.append(class_scores)
        confusion_matrices.append(
            metrics.confusion_matrix(y_test, y_pred, labels=classes)
        )

    avg_scores = np.mean(scores, axis=0)
    avg_confusion_matrix = np.mean(confusion_matrices, axis=0)
    per_class_accuracy = avg_confusion_matrix.diagonal() / avg_confusion_matrix.sum(
        axis=1
    )

    plot_crossval_metrics(
        avg_scores,
        per_class_accuracy,
        avg_confusion_matrix,
        classes,
        model_name,
        ax_metrics,
        ax_confusion,
        subplot_names,
    )
    overall = [np.mean(per_class_accuracy)] + list(
        np.stack(avg_scores[:3]).mean(axis=1)
    )
    return overall


if __name__ == "__main__":
    # Parse command line.
    parser = get_parser()
    args = parser.parse_args()

    # Additional parameters.
    logging_level = logging.INFO if args.verbose else logging.WARNING

    # setup logging.
    logging.basicConfig(
        format="[%(levelname)s]  %(asctime)s %(name)s  %(message)s", level=logging_level
    )
    # Features kept for classification.
    if args.distinguishable_second_layer:
        classes = [
            "Layer 1",
            "Layer 2",
            "Layer 3",
            "Layer 4",
            "Layer 5",
            "Layer 6 a",
            "Layer 6 b",
        ]
        features = [
            "Smoothed: 50 µm: Distance to annotation with Outside Pia µm",
            "Distance to annotation with Outside Pia µm",
            "Smoothed: 50 µm: Min diameter µm",
            "Centroid Y µm",
            "Smoothed: 50 µm: Max diameter µm",
            "Centroid X µm",
            "Smoothed: 50 µm: Circularity",
            "Smoothed: 50 µm: Delaunay: Max triangle area",
            "Smoothed: 50 µm: Hematoxylin: Std.Dev.",
            "Smoothed: 50 µm: Solidity",
            "Smoothed: 50 µm: Delaunay: Num neighbors",
            "Smoothed: 50 µm: Delaunay: Min distance",
            "Length µm",
            "Delaunay: Median distance",
            "DAB: Std.Dev.",
            "Max diameter µm",
            "Area µm^2",
            "Min diameter µm",
            "Delaunay: Mean triangle area",
        ]
    else:
        classes = [
            "Layer 1",
            "Layer 2/3",
            "Layer 4",
            "Layer 5",
            "Layer 6 a",
            "Layer 6 b",
        ]
        features = [
            "Distance to annotation with Outside Pia µm",
            "Smoothed: 50 µm: Distance to annotation with Outside Pia µm",
            "Smoothed: 50 µm: Min diameter µm",
            "Centroid Y µm",
            "Smoothed: 50 µm: Max diameter µm",
            "Centroid X µm",
            "Smoothed: 50 µm: Delaunay: Max triangle area",
            "Smoothed: 50 µm: Circularity",
            "Smoothed: 50 µm: Solidity",
            "Smoothed: 50 µm: Delaunay: Min distance",
            "Smoothed: 50 µm: Nearby detection counts",
            "Area µm^2",
            "Smoothed: 50 µm: Hematoxylin: Min",
            "Smoothed: 50 µm: Area µm^2",
            "Smoothed: 50 µm: Length µm",
            "Delaunay: Median distance",
            "Smoothed: 50 µm: Hematoxylin: Std.Dev.",
            "Delaunay: Mean distance",
            "Max diameter µm",
        ]

    fig = plt.figure(figsize=(18, 15))
    gs = gridspec.GridSpec(3, 4, height_ratios=[1, 1, 0.5])

    ax_knn_metrics = plt.subplot(gs[0, :2])
    ax_knn_confusion = plt.subplot(gs[0, 2:])

    ax_rf_metrics = plt.subplot(gs[1, :2])
    ax_rf_confusion = plt.subplot(gs[1, 2:])

    ax_overall = plt.subplot(gs[2, 1:3])

    overall_metrics_list = []

    model_names = ["K Nearest Neighbors", "Random Forest Classifier"]

    # Get the image names and split them in train/test.
    filenames = get_image_files(args.train_dir, args.train_glob)
    knn = KNeighborsClassifier(n_neighbors=args.neighbors)
    rf = RandomForestClassifier(n_estimators=args.estimators, random_state=42)
    overall_knn = cross_validate(
        train_dir=args.train_dir,
        image_names=filenames,
        features=features,
        classes=classes,
        model=knn,
        model_name=model_names[0],
        ax_metrics=ax_knn_metrics,
        ax_confusion=ax_knn_confusion,
        subplot_names=["A", "B"],
        k=args.k,
        k_repeat=args.k_repeat,
        clean=args.clean,
        eps=args.eps,
        min_samples=args.min_samples,
        distinguishable_second_layer=args.distinguishable_second_layer,
        extension=args.extension,
        gt_column=args.gt_column,
    )
    overall_metrics_list.append(overall_knn)

    overall_rf = cross_validate(
        train_dir=args.train_dir,
        image_names=filenames,
        features=features,
        classes=classes,
        model=rf,
        model_name=model_names[1],
        ax_metrics=ax_rf_metrics,
        ax_confusion=ax_rf_confusion,
        subplot_names=["C", "D"],
        k=args.k,
        k_repeat=args.k_repeat,
        clean=args.clean,
        eps=args.eps,
        min_samples=args.min_samples,
        distinguishable_second_layer=args.distinguishable_second_layer,
        extension=args.extension,
        gt_column=args.gt_column,
    )
    overall_metrics_list.append(overall_rf)

    plot_overall_metrics(
        overall_metrics_list, model_names, ax_overall, subplot_name="E"
    )

    fig.subplots_adjust(
        left=0.05, right=0.8, top=0.95, bottom=0.03, wspace=0.25, hspace=0.23
    )

    # Save the combined figure
    plt.savefig(
        os.path.join(args.save_dir, "combined_CV_metrics.svg"), bbox_inches="tight"
    )
