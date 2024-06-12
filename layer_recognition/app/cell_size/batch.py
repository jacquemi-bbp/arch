""" The cell area click command """

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


def concate_area_dataframes(file_lists, rf_prediction=False):
    """
    conctact dataframe locatated in a directory
    """
    frames = []

    for file_list in file_lists:
        for file in file_list:
            df_image = pd.read_csv(file, index_col=None)
            if rf_prediction:
                df_area = df_image[["Image", "Area µm^2", "RF_prediction"]]
            else:
                df_area = df_image[["Image", "Area µm^2"]]
            frames.append(df_area)

    return pd.concat(frames, ignore_index=True)


@click.command()
@click.option("--config-file-path", required=False, help="Configuration file path")
def cmd(config_file_path):
    """
    Concatenate cells area (um²) dataframe
    """

    config = configparser.ConfigParser()
    config.sections()
    config.read(config_file_path)

    output_path = config["BATCH"]["output_path"]
    cell_features_path = config["BATCH"]["cell_features_path"]
    try:
        cell_features_file_prefix = config["BATCH"]["cell_position_file_prefix"]
    except KeyError:
        cell_features_file_prefix = "Features_"

    file_list = [glob.glob(cell_features_path + cell_features_file_prefix + "*")]

    area_dataframe = concate_area_dataframes(file_list, rf_prediction=True)
    os.makedirs(output_path, exist_ok=True)
    area_dataframe.to_csv(output_path + "/cells_area.csv")
