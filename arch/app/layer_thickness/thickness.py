""" The layer thickness click command """

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

from collections import defaultdict
import configparser

import glob
import pathlib
import os

import click
import numpy as np
import pandas as pd

from arch.utilities import get_image_layers_thickness


@click.command()
@click.option(
    "--feature-file-path",
    type=pathlib.Path,
    required=True,
    help="Path to the directory that contains the cells'features including the RF_predictioh",
)
@click.option(
    "--output-filename",
    type=pathlib.Path,
    required=True,
    help="Path to file that will contain the dataframe with layer thickness results",
)
def cmd(feature_file_path, output_filename):
    """
    Compute layers thickness and saved result to a dataframe2
    """
    regex = feature_file_path / '*.csv'
    features_filelist = glob.glob(regex.as_posix())
    total = len(features_filelist)
    index=0
    rectangle_widths = defaultdict(list)
    for features_path in features_filelist:
        df_feat = pd.read_csv(features_path, index_col=0)
        rectangle_widths = get_image_layers_thickness(df_feat, rectangle_widths)
        print(f'INFO Done {index}/{total}\r', end='')
        index+=1
        
    thickness_std=[]
    thickness_mean=[]
    layers=[]
    for layer, values in rectangle_widths.items():
        thickness_std.append(np.std(values))
        thickness_mean.append(np.mean(values))
        layers.append(layer)
    d = {'layers' : layers, 'thickness_mean': thickness_mean, 'thickness_std': thickness_std}
    df = pd.DataFrame(data=d)
    df.to_csv(output_filename)    
    print(f'INFO DONE: layers thickness information saved to {output_filename} ')