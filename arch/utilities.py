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


'''
def concat_dataframe(dest, source=None):
    """
    Concatenate the source dataframe to the dest one
    Args:
        source: (pandas.DataFrame)
        dest: (pandas.DataFrame)
    Returns:
         a pandas.DataFrame: The contatenation of source into dest

    Notes: If source == None, return dest Dataframe
    """
    if source is None:
        return dest
    return pd.concat([dest, source])


def get_angle(p1, p2) -> float:
    """
    Get the angle of this line with the horizontal axis.
    Args:
        p1: ()
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    theta = math.atan2(dy, dx)
    if theta < 0:
        theta = math.pi * 2 + theta
    return theta


def get_image_animal(images_metadata):
    """
    Get image animal metadata value
    :param images_metadata: (dictionary) Key -> Image name. Values -> image metadata
    :return:
        str: The image lateral value or np.nan if not existing
    """
    results = {}
    for image in images_metadata:
        if "Animal" in image["metadata"]:
            results[image["imageName"]] = image["metadata"]["Animal"]
        else:
            results[image["imageName"]] = "ND"

    return results


def get_image_immunohistochemistry(images_metadata):
    """
    Get image Immunohistochemistry ID metadata value
    :param images_metadata: (dictionary) Key -> Image name. Values -> image metadata
    :return:
        str: The image lateral value or np.nan if not existing
    """
    results = {}
    for image in images_metadata:
        if "Immunohistochemistry ID" in image["metadata"]:
            results[image["imageName"]] = image["metadata"]["Immunohistochemistry ID"]
        else:
            results[image["imageName"]] = "ND"

    return results


def get_image_lateral(images_metadata):
    """
    Get image lateral metadata value
    :param images_metadata: (dictionary) Key -> Image name. Values -> image metadata
    :return:
        float: The image lateral value or np.nan if not existing
    """
    images_lateral = {}
    for image in images_metadata:
        if "Distance to midline" in image["metadata"]:
            images_lateral[image["imageName"]] = image["metadata"][
                "Distance to midline"
            ]
        else:
            images_lateral[image["imageName"]] = np.nan
    return images_lateral


def get_specific_metadata(images_metadata, meta_name, default=np.nan):
    """
    Get a metadata value
    :param images_metadata: (dictionary) Key -> Image name. Values -> image metadata
    :param meta_name: (str)> The name of the metadata
    :default: the default value to return if metadata name does not exist
    :return:
        float|str: The metadata value oif exist or default if not existing
    """
    result = {}
    for image in images_metadata:
        if meta_name in image["metadata"]:
            result[image["imageName"]] = image["metadata"][meta_name]
        else:
            result[image["imageName"]] = default
    return result

 


def  get_image_to_exlude_list(df_image_to_exclude):
    df_image_to_exclude=df_image_to_exclude.dropna().reset_index(drop=True)
    # Step 2: Apply a function to each value
    def remove_space(image_name):
        return image_name.replace(" ", "")
    new_image_name_column = df_image_to_exclude['Image ID to exclude'].apply(remove_space)

    # Step 3: Assign the new values back to the column
    df_image_to_exclude['Image ID to exclude'] = new_image_name_column
    return list(df_image_to_exclude['Image ID to exclude'])
'''
