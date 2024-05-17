""" Train a machine learning model that predicts brain layers """

# Copyright (C) 2021  Blue Brain Project, EPFL
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


import pathlib
import click
from sklearn.model_selection import train_test_split
from arch.ml.utils import get_image_files
from arch.ml.train_and_predict import train_and_evaluate_model

@click.command()
@click.option(
    "--train-dir",
    type=pathlib.Path,
    help="Directory containing training data files.",
    required=True,
)
@click.option(
    "--save-dir",
    type=pathlib.Path,
    help="Directory where to save the model.",
    required=True,
)
@click.option(
    "--train-glob",
    type=str,
    default="*.csv",
    help="Glob pattern to match the training files.",
)
@click.option(
    "--extension",
    type=click.Choice(["txt", "csv"]),
    default="csv",
    help="extension of the files containing the data.",
)

@click.option(
    "--gt-column",
    type=str,
    default="Expert_layer",
    help="Name of the ground truth column in the CSVs.",
)
@click.option(
    "--distinguishable-second-layer",
    "-d",
    is_flag=True,
    default=True,
    help="Treats layer 2 and 3 as separate layers.",
)
@click.option(
    "--random-split",
    is_flag=True,
    default=True,
    help=(
        "Use test images randomly extracted from the train set. If false, defaults"
        " to the predefine hardcoded images."
    ),
)
@click.option(
    "--split-ratio",
    "-s",
    type=float,
    default=0.1,
    help=(
        "Fraction of the dataset sent to the test set. Only if using"
        " --random-split."
    ),
)
@click.option(
    "--estimators",
    "-e",
    type=int,
    default=100,
    help="Number of estimators for the random forest model.",
)
@click.option(
    "--train-knn",
    is_flag=True,
    default=False,
    help="Train a K Nearest Neighbor model",
)

def cmd(
    train_dir,
    train_glob,
    extension,
    save_dir,
    gt_column,
    distinguishable_second_layer,
    random_split,
    split_ratio,
    estimators,
    train_knn,
):
    """
    Train a model based on cells feature that predicts brain layers
    """
 # Features kept for classification.
    if distinguishable_second_layer:
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
    filenames = get_image_files(train_dir, train_glob)
    print(f'DEBUG 1 filenames {filenames}')

    if split_ratio > 0 and random_split:
        train_image_names, test_image_names = train_test_split(
            filenames, test_size=split_ratio, random_state=42, shuffle=True
        )
    elif split_ratio == 0 and random_split:
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
    print(f"The training set contains {len(train_image_names)} images. {train_image_names}")
    print(f"The test set contains {len(test_image_names)} images. {test_image_names}")

    # Train the model and optionally evaluate it.
    train_and_evaluate_model(
        train_dir=train_dir,
        save_dir=save_dir,
        features=features,
        train_image_names=train_image_names,
        gt_column=gt_column,
        extension=extension,
        distinguishable_second_layer=distinguishable_second_layer,
        estimators=estimators,
        clean_predictions=False,
        test_image_names=test_image_names,
        split_ratio=split_ratio,
        classes=classes,
        train_knn=train_knn
    )
