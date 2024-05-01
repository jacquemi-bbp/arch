"""
Utilities module
"""

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

# import math
import random

# import pandas as pd
# import numpy as np


from sklearn.neighbors import NearestNeighbors


def stereology_exclusion(dataframe):
    """
    The classical optical dissector method that adds a virtual -z coordinate
    to cells to be able to exclude from one slice, the cells located on two slices boundaries.

    Args:
        dataframe (pandas.dataframe): dataframe thats contains the cells X/Y coordinates

    Returns:
        The input dataframe wityh the new exlude column
    """

    data = dataframe[["Centroid X µm", "Centroid Y µm"]].values
    nbrs = NearestNeighbors(n_neighbors=5, algorithm="kd_tree").fit(data)
    dataframe["mean_diameter"] = 0.5 * (
        dataframe["Max diameter µm"] + dataframe["Min diameter µm"]
    )

    def exclude(sample, slice_thickness=50):
        sample["neighbors"] = nbrs.kneighbors(data, 6, return_distance=False)[
            sample.name, :
        ]  # sample.name = row index
        neighbor_mean = dataframe.iloc[sample["neighbors"]]["mean_diameter"].mean()
        sample["neighbor_mean"] = neighbor_mean
        sample["exclude_for_density"] = (
            random.uniform(0, slice_thickness) + neighbor_mean / 2 >= slice_thickness
        )
        return sample

    dataframe_with_exclude_flag = dataframe.apply(exclude, axis=1)
    return dataframe_with_exclude_flag


def get_image_to_exlude_list(df_image_to_exclude):
    """
    Get the list of exclude image from the input dataframe
    Args:
        df_image_to_exclude(pandas.Dataframe)
    Returns:
        list of images name (str) to exclude
    """

    df_image_to_exclude = df_image_to_exclude.dropna().reset_index(drop=True)

    # Step 2: Apply a function to each value
    def remove_space(image_name):
        return image_name.replace(" ", "")

    new_image_name_column = df_image_to_exclude["Image ID to exclude"].apply(
        remove_space
    )

    # Step 3: Assign the new values back to the column
    df_image_to_exclude["Image ID to exclude"] = new_image_name_column
    return list(df_image_to_exclude["Image ID to exclude"])
