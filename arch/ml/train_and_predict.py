"""Main script to train, evaluate and predict with cell prediction models."""

import argparse
import logging
import os
import pathlib
import pickle
import warnings

import tqdm
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from utils import (
    clean_predictions,
    get_image_files,
    image_to_df,
    plot_eval_metrics,
    plot_results,
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
        "--model-file",
        type=pathlib.Path,
        default=None,
        help=(
            "Path to the file where to save/load the RF model. If the file doesn't"
            " exist, the model is trained and saved at the provided location. If it"
            " exists, loads the model and skips training."
        ),
    )
    parser.add_argument(
        "--train-glob",
        type=str,
        default="*.csv",
        help="Glob pattern to match the training files.",
    )
    parser.add_argument(
        "--gt-column",
        type=str,
        default="Expert_layer",
        help="Name of the ground truth column in the CSVs.",
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
        "--random-split",
        action="store_true",
        help=(
            "Use test images randomly extracted from the train set. If false, defaults"
            " to the predefine hardcoded images."
        ),
    )
    parser.add_argument(
        "--split-ratio",
        "-s",
        type=float,
        default=0.1,
        help=(
            "Fraction of the dataset sent to the test set. Only if using"
            " --random-split."
        ),
    )
    parser.add_argument(
        "--estimators",
        "-e",
        type=int,
        default=100,
        help="Number of estimators for the random forest model.",
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
        "--show",
        action="store_true",
        help="Display the figures.",
    )
    parser.add_argument(
        "--pred-dir",
        default=None,
        type=pathlib.Path,
        help=(
            "Directory containing images to be predicted + glob pattern to match the"
            " corresponding files. No prediction if not specified."
        ),
    )
    parser.add_argument(
        "--pred-save",
        default=None,
        type=pathlib.Path,
        help="Where to save the CSVs containing predictions.",
    )
    parser.add_argument(
        "--pred-glob",
        type=str,
        default="*.csv",
        help="Glob pattern to match the prediction files.",
    )
    parser.add_argument(
        "--pred-extension",
        type=str,
        default="csv",
        choices=["txt", "csv"],
        help="Extension of the files used for prediction.",
    )
    parser.add_argument(
        "--neighbors",
        type=int,
        help="Number of neighbors for KNN.",
        default=32,
    )
    parser.add_argument(
        "--train-knn",
        action="store_true",
        help="Train a K Nearest Neighbor model",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Control verbosity.",
    )
    return parser


def train_and_evaluate_model(
    train_dir: pathlib.Path,
    features: list[str],
    train_image_names: list[str],
    save_dir: pathlib.Path,
    model_file: pathlib.Path | None = None,
    gt_column: str = "Expert_layer",
    extension: str = "csv",
    distinguishable_second_layer: bool = True,
    estimators: int = 100,
    clean_predictions: bool = False,
    eps: float = 0.05,
    min_samples: int = 3,
    test_image_names: list[str] | None = None,
    split_ratio: float = 0.1,
    classes: list[str] | None = None,
    show: bool = False,
    train_knn: bool = False,
    neighbors: int = 32,
) -> RandomForestClassifier:
    """Train the random forest classifier, and optionally compute its accuracy. Optionally train a KNN classifier as well for comparison only.

    Parameters
    ----------
    train_dir
        Directory containing the training files
    features
        List of features to use for training
    train_image_names
        List of filenames used for training
    save_dir
        Path to the directory where to save metrics and images
    model_file
        Optional file of a pre-trained model to skip training
    gt_column
        Name of the column containing the ground truth in the csv
    extension
        File extension of the training files
    distinguishable_second_layer
        Use Layer 2/3 if False, Layer 2 and Layer 3 if True
    estimators
        Number of estimators of the RF model
    clean_predictions
        Apply post-processing algorithm if True
    eps
        Radius of the circle for DBSCAN algorithm
    min_samples
        Number of neighbors that must be contained within eps circle in DBSCAN
    test_image_names
        List of filenames used for testing
    split_ratio
        Ratio of train/test images
    show
        Display figures after testing if True
    train_knn
        Flag to train a KNN or not.
    neighbors
        KNN neighbors parameter.

    Returns
    -------
    The fitted random forest classifier.
    """
    logger.info("Reading training images")
    x_train, y_train, _ = image_to_df(
        image_names=train_image_names,
        data_path=train_dir,
        classes=classes,
        features=features,
        filter=True,
        distinguish_second_layer=distinguishable_second_layer,
        extension=extension,
        gt_column=gt_column,
    )
    logger.info(f"Training dataset shape : {x_train.shape}")
    rf = RandomForestClassifier(n_estimators=estimators, random_state=42)
    logger.info("Training the random forest...")
    rf.fit(x_train, y_train)
    if model_file:
        pickle.dump(rf, open(model_file, "wb"))
    else:
        pickle.dump(rf, open(save_dir / "trained_rf.pkl", "wb"))

    if split_ratio > 0 and test_image_names and classes:
        models = [(rf, "Random Forest Classifier")]

        if train_knn:
            logger.info("Training the knn...")
            knn = KNeighborsClassifier(n_neighbors=neighbors)
            knn.fit(x_train, y_train)
            models.append((knn, "K Nearest Neighbor"))

        test_images = []
        predictions = []
        for image_name in tqdm.tqdm(
            test_image_names,
            desc="Evaluating on test images",
            total=len(test_image_names),
        ):
            ground_truth, prediction = plot_results(
                image_name=image_name,
                data_path=train_dir,
                save_path=save_dir,
                classes=classes,
                models=models,
                clean=clean_predictions,
                eps=eps,
                min_samples=min_samples,
                features=features,
                distinguish_second_layer=distinguishable_second_layer,
                extension=extension,
                gt_column=gt_column,
                show=show,
            )
            test_images.append(ground_truth)
            predictions.append(prediction)
            for pred, (_, model_name) in zip(prediction, models):
                class_scores = metrics.precision_recall_fscore_support(
                    ground_truth, pred
                )
                confusion_matrix = metrics.confusion_matrix(ground_truth, pred)
                save_path = (
                    os.path.join(save_dir, pathlib.Path(image_name).stem + f"_{'_'.join(model_name.split(' '))}") + "_metrics"
                )
                per_class_accuracy = confusion_matrix.diagonal() / confusion_matrix.sum(
                    axis=1
                )
                plot_eval_metrics(
                    class_scores,
                    per_class_accuracy,
                    confusion_matrix,
                    classes,
                    model_name,
                    save_path=save_path,
                    show=show,
                )
        test_images = [label for sub_list in test_images for label in sub_list]
        rf_predictions = [pred[0] for pred in predictions]
        flattened_rf_predictions = [
            label for image in rf_predictions for label in image
        ]
        print(
            "RF Accuracy:",
            metrics.accuracy_score(test_images, flattened_rf_predictions),
        )

        if train_knn:
            knn_predictions = [pred[1] for pred in predictions]
            flattened_knn_predictions = [
                label for image in knn_predictions for label in image
            ]
            print(
                "KNN Accuracy:",
                metrics.accuracy_score(test_images, flattened_knn_predictions),
            )

    return rf


def predict(
    pred_dir: pathlib.Path,
    model: RandomForestClassifier,
    features: list[str],
    classes: list[str],
    pred_save: pathlib.Path | None = None,
    pred_glob: str = "*.csv",
    distinguishable_second_layer: bool = True,
    pred_extension: str = "csv",
    gt_column: str = "Expert_layer",
    clean: bool = False,
    eps: float = 0.05,
    min_samples: int = 3,
) -> None:
    """Predict cell classes on non annotated images.

    Parameters
    ----------
    pred_dir
        Directory where the data to predict is located.
    model
        Trained random forest classifier.
    features
        List of data feature kept.
    classes
        List of classes the model should output.
    pred_save
        Path to the directory where to save the predictions.
    pred_glob
        Glob pattern to select only certain files in pred_dir
    distinguishable_second_layer
        Whether to keep layer 2 and 3 separated.
    pred_extension
        Extension of the pred files.
    gt_column
        Name of the column carrying the ground truth in training data that was used for the model
    clean_predictions
        Whether to apply post-processing.
    eps
        Radius of the circle used in DBSCAN for post-processing.
    min_samples
        Minimum number of samples to find to define a central point in DBSCAN.
    """
    logger.info("Predicting on un-annotated images.")
    image_names = get_image_files(pred_dir, pred_glob)
    for image in tqdm.tqdm(image_names, total=len(image_names)):
        x, _, detection_df = image_to_df(
            [image],
            pred_dir,
            classes=classes,
            features=features,
            filter=True,
            distinguish_second_layer=distinguishable_second_layer,
            extension=pred_extension,
            gt_column=gt_column,
        )
        try:
            if clean:
                predictions = clean_predictions(
                    detection_df.copy(),
                    rf.predict(x),
                    eps=eps,
                    min_samples=min_samples,
                    gt_column=gt_column,
                )
            else:
                predictions = model.predict(x)
        except ValueError:
            continue
        to_remove = [col for col in detection_df.columns if "Unnamed" in col]
        detection_df["RF_prediction"] = predictions
        detection_df.drop(
            columns=to_remove,
            inplace=True,
        )
        if not pred_save:
            warnings.warn("--pred-save not set. Not saving CSVs")
        else:
            if not os.path.exists(pred_save):
                os.makedirs(pred_save, exist_ok=True)
            detection_df.to_csv(
                os.path.join(pred_save, image),
                index=False,
            )


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

    # Get the image names and split them in train/test.
    filenames = get_image_files(args.train_dir, args.train_glob)
    if not args.save_dir.exists():
        args.save_dir.mkdir()
    if args.model_file and not args.model_file.suffix == ".pkl":
        raise ValueError(
            "The model_file argument should end with a filename with the '.pkl'"
            " extension."
        )
    if args.model_file and os.path.exists(args.model_file):
        rf = pickle.load(open(args.model_file, "rb"))
    else:
        if args.split_ratio > 0 and args.random_split:
            train_image_names, test_image_names = train_test_split(
                filenames, test_size=args.split_ratio, random_state=42, shuffle=True
            )
        elif args.split_ratio == 0 and args.random_split:
            train_image_names = filenames
            test_image_names = None
        else:
            test_image_names = [
                "Features_SLD_0000565.vsi-20x_01.csv",
                "Features_SLD_0000540.vsi-20x_01.csv",
                "Features_SLD_0000536.vsi-20x_02.csv",
                "Features_SLD_0000560.vsi-20x_05.csv",
                "Features_SLD_0000533.vsi-20x_03.csv",
                "Features_SLD_0000563.vsi-20x_01.csv",
            ]
            train_image_names = [
                image for image in filenames if image not in test_image_names
            ]
        logger.info(f"The training set contains {len(train_image_names)} images.")
        logger.info(f"The test set contains {len(test_image_names)} images.")

        # Train the model and optionally evaluate it.
        rf = train_and_evaluate_model(
            train_dir=args.train_dir,
            features=features,
            train_image_names=train_image_names,
            save_dir=args.save_dir,
            model_file=args.model_file,
            gt_column=args.gt_column,
            extension=args.extension,
            distinguishable_second_layer=args.distinguishable_second_layer,
            estimators=args.estimators,
            clean_predictions=args.clean,
            eps=args.eps,
            min_samples=args.min_samples,
            test_image_names=test_image_names,
            split_ratio=args.split_ratio,
            classes=classes,
            show=args.show,
            train_knn=args.train_knn,
            neighbors=args.neighbors,
        )
    if args.pred_dir:
        predict(
            pred_dir=args.pred_dir,
            model=rf,
            features=features,
            classes=classes,
            pred_save=args.pred_save,
            pred_glob=args.pred_glob,
            distinguishable_second_layer=args.distinguishable_second_layer,
            pred_extension=args.pred_extension,
            gt_column=args.gt_column,
            clean=args.clean,
            eps=args.eps,
            min_samples=args.min_samples,
        )
    raise SystemExit(0)
