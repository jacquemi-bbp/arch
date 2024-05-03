""" The convert click command """

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


import configparser
import glob
import os

import click
import pandas as pd

from arch.density import single_image_process
from arch.io import write_dataframe_to_file


@click.command()
@click.option("--config-file-path", required=False, help="Configuration file path")
@click.option("--visualisation-flag", is_flag=True)
@click.option("--save-plot-flag", is_flag=True)
@click.option(
    "--image-to-exlude-path",
    help="exel files that contains the list of image to exclude (xlsx).",
    required=False,
)
def cmd_depth(
    config_file_path,
    visualisation_flag,
    save_plot_flag,
    image_to_exlude_path,
):
    """
    Compute cell densities as function of brain depth 
    """
    config = configparser.ConfigParser()
    config.sections()
    config.read(config_file_path)

    output_path = config["BATCH"]["output_path"]
    cell_position_path = config["BATCH"]["cell_position_path"]
    try:
        cell_position_file_prefix = config["BATCH"]["cell_position_file_prefix"]
    except KeyError:
        cell_position_file_prefix = 'Features_'

    points_annotations_path = config["BATCH"]["points_annotations_path"]
    
    try:
        points_annotations_file_prefix = config["BATCH"]["points_annotations_file_prefix"]
    except KeyError:
        points_annotations_file_prefix = ''

    s1hl_path = config["BATCH"]["s1hl_path"]
    
    try:
        s1hl_file_prefix = config["BATCH"]["s1hl_file_prefix"]
    except KeyError:
        s1hl_file_prefix = ''

    thickness_cut = float(config["BATCH"]["thickness_cut"])
    nb_row = int(config["BATCH"]["nb_row"])
    nb_col = int(config["BATCH"]["nb_col"])

    multiple_image_process(
        cell_position_path,
        cell_position_file_prefix,
        output_path,
        image_to_exlude_path,
        visualisation_flag,
        save_plot_flag,
        compute_per_depth=True,
        compute_per_layer=False,
        points_annotations_path=points_annotations_path,
        points_annotations_file_prefix=points_annotations_file_prefix,
        s1hl_path=s1hl_path,
        s1hl_file_prefix=s1hl_file_prefix,
        thickness_cut=thickness_cut,
        nb_col=nb_col,
        nb_row=nb_row,
    )


@click.command()
@click.option("--config-file-path", required=False, help="Configuration file path")
@click.option("--visualisation-flag", is_flag=True)
@click.option("--save-plot-flag", is_flag=True)
@click.option(
    "--image-to-exlude-path",
    help="exel files that contains the list of image to exclude (xlsx).",
    required=False,
)
def cmd_layer(
    config_file_path,
    visualisation_flag,
    save_plot_flag,
    image_to_exlude_path,
):
    """
    Compute cell densities per brain layer 
    """
    config = configparser.ConfigParser()
    config.sections()
    config.read(config_file_path)

    output_path = config["BATCH"]["output_path"]
    cell_position_path = config["BATCH"]["cell_position_path"]

    try:
        cell_position_file_prefix = config["BATCH"]["cell_position_file_prefix"]
    except KeyError:
        cell_position_file_prefix = 'Features_'
    
    try:
        alpha = int(config["BATCH"]["alpha"])
    except KeyError:
        alpha = 0.05

    multiple_image_process(
        cell_position_path,
        cell_position_file_prefix,
        output_path,
        image_to_exlude_path,
        visualisation_flag,
        save_plot_flag,
        compute_per_depth=False,
        compute_per_layer=True,
        alpha=alpha,
    )


def multiple_image_process(
    cell_position_path,
    cell_position_file_prefix,
    output_path,
    image_to_exlude_path,
    visualisation_flag,
    save_plot_flag,
    points_annotations_path=None,
    points_annotations_file_prefix=None,
    s1hl_path=None,
    s1hl_file_prefix=None,
    compute_per_depth=False,
    compute_per_layer=False,
    nb_col=20,
    nb_row=20,
    thickness_cut=50,
    alpha=0,
):
    """
    loop over image and execute single_image_process for each image
    Args:
        cell_position_path: (str)
        cell_position_file_prefix: (str)
        output_path: (str)
        image_to_exlude_path: (str)
        compute_per_depth: (bool)
        compute_per_layer: (bool)
        points_annotations_path: (str)
        points_annotations_file_prefix: (str)
        s1hl_path: (str)
        s1hl_file_prefix: (str)
        thickness_cut: (float) : The thickness of the slices
        nb_col: (int): Nb of grifd columns. Only used if compute_per_depth is True
        nb_row: (int) : Nb of grid rows. Only used if compute_per_depth is True
        visualisation_flag (bool) If True display plots
        save_plot_flag (bool) If True, save plots
        alpha: (float) alphashape alpha value. Only used if compute_per_layer is True
    """
    # List images to compute
    image_path_list = glob.glob(cell_position_path + "/*.csv")

    image_list = []
    feature_str_length = len(cell_position_file_prefix)
    for path in image_path_list:
        prefix_pos = path.rfind(cell_position_file_prefix)
        if prefix_pos > -1:
            feature_pos = path.rfind(cell_position_file_prefix) + feature_str_length
            image_list.append(path[feature_pos : path.find(".csv")])

    if len(image_list) == 0:
        print("WARNING: No input files to process.")
        return

    print(f'INFO" {len(image_list)} to process {image_list}')

    if not os.path.exists(output_path):
        # if the directory is not present then create it.
        os.makedirs(output_path)
        print(f"INFO: Create output_path {output_path}")

    # Verify that the image is not in the exlude images list
    df_image_to_exclude = None
    if image_to_exlude_path:
        df_image_to_exclude = pd.read_excel(
            image_to_exlude_path, index_col=0, skiprows=[0, 1, 2, 3, 4, 5, 6, 7]
        )
        if compute_per_layer:
            # Some image may be keep if we only compute the density per layer
            df_image_to_exclude = df_image_to_exclude[
                df_image_to_exclude[
                    "Exclusion reason (Cell density calculation)"
                ].str.find("DistanceToMidline_3.05-3.25")
                == -1
            ]

    for image_name in image_list:
        print("INFO: Process single image ", image_name)
        cell_position_file_path = (
            cell_position_path + "/" + cell_position_file_prefix + image_name + ".csv"
        )

        if compute_per_depth:
            points_annotations_file_path = (
                points_annotations_path
                + "/"
                + points_annotations_file_prefix
                + image_name
                + "_points_annotations.csv"
            )
            s1hl_file_path = (
                s1hl_path
                + "/"
                + s1hl_file_prefix
                + image_name
                + "_S1HL_annotations.csv"
            )
        else:
            points_annotations_file_path = None
            s1hl_file_path = None

        densities_dataframe, per_layer_dataframe = single_image_process(
            image_name,
            cell_position_file_path,
            points_annotations_file_path,
            s1hl_file_path,
            output_path,
            df_image_to_exclude=df_image_to_exclude,
            thickness_cut=thickness_cut,
            nb_col=nb_col,
            nb_row=nb_row,
            visualisation_flag=visualisation_flag,
            save_plot_flag=save_plot_flag,
            alpha=alpha,
            compute_per_layer=compute_per_layer,
            compute_per_depth=compute_per_depth,
        )
        if compute_per_depth:
            if densities_dataframe is None:
                print(
                    f"ERROR: {image_name} The computed density is not valid to compute\
the per depth density"
                )
            else:
                densities_dataframe_full_path = output_path + "/" + image_name + ".csv"

                write_dataframe_to_file(
                    densities_dataframe, densities_dataframe_full_path
                )
                print(
                    f"INFO: Write density dataframe =to {densities_dataframe_full_path}"
                )

        if compute_per_layer:
            if per_layer_dataframe is None:
                print(
                    "ERROR: The computed density per layer is not valid to compute the per \
                 depth density"
                )
            else:
                densities_per_layer_dataframe_full_path = (
                    output_path + "/" + image_name + "_per_layer.csv"
                )
                write_dataframe_to_file(
                    per_layer_dataframe, densities_per_layer_dataframe_full_path
                )
                print(
                    f"INFO: Write density per layer dataframe to \
                 {densities_per_layer_dataframe_full_path}"
                )