""" USe a mahcine learning model to predict brain layers """

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

import os
import pickle
import pathlib
import click


from arch.ml.train_and_predict import predict


@click.command()
@click.option(
    "--model-file",
    type=pathlib.Path,
    help=("Path to the file where to load the RF model."),
    required=True,
)
@click.option(
    "--pred-dir",
    type=pathlib.Path,
    help=(
        "Directory containing images to be predicted + glob pattern to match the"
        " corresponding files. No prediction if not specified."
    ),
    required=True,
)
@click.option(
    "--pred-save",
    type=pathlib.Path,
    help="Where to save the CSVs containing predictions.",
    required=True,
)
@click.option(
    "--pred-glob",
    type=str,
    default="*.csv",
    help="Glob pattern to match the prediction files.",
)
@click.option(
    "--pred-extension",
    type=click.Choice(["txt", "csv"]),
    default="csv",
    help="Extension of the files used for prediction.",
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
    "--extension",
    type=click.Choice(["txt", "csv"]),
    default="csv",
    help="extension of the files containing the data.",
)
@click.option(
    "--clean",
    "-c",
    type=bool,
    default=True,
    help=(
        "Use post-processing to attempt removing wrongly classified cluster. Can"
        " potentially harm the prediction's quality."
    ),
)
@click.option(
    "--eps",
    type=float,
    default=0.05,
    help="Radius of the circle used in the DBSCAN algorithm for post-processing.",
)
@click.option(
    "--min-samples",
    type=int,
    default=3,
    help=(
        "Minimum number of neighbors required within eps to be treated as a central"
        " point."
    ),
)
@click.option(
    "--neighbors",
    type=int,
    help="Number of neighbors for KNN.",
    default=32,
)
def cmd(
    model_file,
    gt_column,
    distinguishable_second_layer,
    extension,
    clean,
    eps,
    min_samples,
    pred_dir,
    pred_save,
    pred_glob,
    pred_extension,
    neighbors,
):
    """
    predict brain layers on images
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
        
    if model_file and not model_file.suffix == ".pkl":
        raise ValueError(
            "The model_file argument should end with a filename with the '.pkl'"
            " extension."
        )
    assert os.path.exists(model_file)
    rf = pickle.load(open(model_file, "rb"))
    
    if pred_dir:
        predict(
            pred_dir=pred_dir,
            model=rf,
            features=features,
            classes=classes,
            pred_save=pred_save,
            pred_glob=pred_glob,
            distinguishable_second_layer=distinguishable_second_layer,
            pred_extension=pred_extension,
            gt_column=gt_column,
            clean=clean,
            eps=eps,
            min_samples=min_samples,
        )



