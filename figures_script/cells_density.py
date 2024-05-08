#!/usr/bin/env python
# coding: utf-8

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
import glob
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# Customize matplotlib
matplotlib.rcParams.update(
    {
        "text.usetex": False,
        "font.family": "stixgeneral",
        "mathtext.fontset": "stix",
    }
)
import pandas as pd
import seaborn as sns
from shapely import Polygon
from PIL import ImageColor

def get_color(distiguish=True, return_type="dict", return_unit="hex"):

    if distiguish:
        layers_color = {
            "Layer 1": "#ff0000",
            "Layer 2": "#ff0099",
            "Layer 3": "#cc00ff",
            "Layer 4": "#3300ff",
            "Layer 5": "#0066FF",
            "Layer 6 a": "#00ffff",
            "Layer 6 b": "#00ff66",
        }
    else:
        layers_color = {
            "Layer 1": "#ff0000",
            "Layer 2/3": "#751402",
            "Layer 4": "#3300ff",
            "Layer 5": "#0066FF",
            "Layer 6 a": "#00ffff",
            "Layer 6 b": "#00ff66",
        }
    layers_color_int = {}
    layers_color_float = {}

    for key, value in layers_color.items():
        layers_color_int[key] = list(np.array(ImageColor.getcolor(value, "RGB")))
        layers_color_float[key] = list(
            np.array(ImageColor.getcolor(value, "RGB")) / 255
        )

    if return_type == "dict":
        if return_unit == "hex":
            return layers_color
        elif return_unit == "float":
            return layers_color_float
        elif return_unit == "int":
            return layers_color_int

    if return_type == "list":
        if return_unit == "hex":
            return list(layers_color.values())
        elif return_unit == "float":
            return list(layers_color_float.values())
        elif return_unit == "int":
            return list(layers_color_int.values())

    raise (ValueError("return_type or return_unit not valid"))


def get_per_layer_df(path):
    file_list_layer = glob.glob(path + "*.csv")
    layer_df = pd.DataFrame()
    dfs = [layer_df]
    for file in file_list_layer:
        df = pd.read_csv(file, index_col=0)
        dfs.append(df)
    layer_df = pd.concat(dfs)
    return layer_df



def concate_density_dataframes(file_list, std_dev_factor=1):
    """
    conctact dataframe locatated in a directory and filter the density value with std_dev_factor
    """
    df = pd.DataFrame()
    densities = []

    for file in file_list:
        df_image = pd.read_csv(file, index_col=0)
        densities.append(df_image.densities)
    densities_mean = np.mean(densities)
    std_dev = np.std(densities) * 1
    df = pd.DataFrame()
    frames = []
    for file in file_list:
        df_image = pd.read_csv(file, index_col=0)
        if np.mean(df_image.densities) < std_dev:
            print(np.unique(df_image.image))
        else:
            frames.append(df_image)

    return pd.concat(frames, ignore_index=True)


def get_filtered_density_df(
    images_id,
    density_df,
):
    image_id = list(density_df["image"])
    mask = density_df.image.isin(images_id)
    return density_df[mask]


def plot(
    densities_per_depth,
    title,
    plot_median=False,
    plt_detail=False,
    display_legend=False,
    output_path=None,
):
    """
    :param densities: np.array of np.float.32 of shape (nb_images, nb_percentages, 2) 2-> depth_percentage, density
    """

    average = {}
    median = {}
    plt.figure(figsize=(5, 5))

    density_dict = defaultdict(list)
    for density in densities_per_depth:
        densities_value = density[:, 1]
        depthes = density[:, 0]
        if plt_detail:
            plt.plot(densities_value, depthes)
        for density, depth in zip(densities_value, depthes):
            density_dict[depth].append(density)

    for depth, densities in density_dict.items():
        average[depth] = np.average(densities)
        median[depth] = np.median(densities)

    plt.plot(
        list(average.values()),
        list(average.keys()),
        linewidth=3,
        c="black",
        label="average values",
    )
    if plot_median:
        plt.plot(
            list(median.values()),
            list(median.keys()),
            linewidth=3,
            c="khaki",
            label="median values",
        )
    plt.title(f"{title}")
    plt.gca().set_xlabel("Cell density cells/mm3")
    plt.gca().set_ylabel("percentage of depth [%]")
    current_values = plt.gca().get_yticks()
    _ = plt.gca().set_yticklabels(["{:.1e}".format(x) for x in current_values])
    current_values = plt.gca().get_xticks()
    _ = plt.gca().set_xticklabels(["{:.1e}".format(x) for x in current_values])
    plt.legend()
    plt.gca().invert_yaxis()

    if display_legend:
        lgnd = plt.legend(loc="upper left", scatterpoints=1, fontsize=10)
        for i in range(1):
            lgnd.legendHandles[i]._sizes = [5]
            lgnd.legendHandles[i]._alpha = 1

    if output_path is not None:
        plt.savefig(output_path, bbox_inches="tight", pad_inches=0)


def plot_mean_and_std_dev(
    density_dfs, labels="", colors="blue", title=None, output_path=None
):
    """
    param: density_df' list of (or a single) pandas dataframe with columns : image 	depth_percentage densities
    """

    if not isinstance(density_dfs, list):
        density_dfs = [density_dfs]

    if not isinstance(labels, list):
        labels = [labels]

    if not isinstance(colors, list):
        colors = [colors]

    plt.figure(figsize=(5, 5))
    for density_df, label, color in zip(density_dfs, labels, colors):
        percentage = np.unique(density_df.depth_percentage)
        densities = np.array(
            list(density_df.groupby("depth_percentage")["densities"].apply(list).values)
        )

        density_std = np.std(densities, axis=1)
        density_mean = densities.mean(axis=1)

        _ = plt.plot(
            density_mean,
            percentage,
            label=label + " Mean densities",
            linewidth=4,
            c=color,
        )
        _ = plt.fill_betweenx(
            percentage,
            density_mean - density_std,
            density_mean + density_std,
            alpha=0.3,
            color=color,
            label=label + " Standard deviation",
        )
        plt.legend()

        plt.gca().set_xlabel("Cell density [cells/mm$^3$]")
        plt.gca().set_ylabel("Standard cortical depth [%]]")
        current_values = plt.gca().get_xticks()
        _ = plt.gca().set_xticklabels(["{:.1e}".format(x) for x in current_values])
        current_values = plt.gca().get_yticks()
        _ = plt.gca().set_yticklabels(["{:.1e}".format(x) for x in current_values])

    plt.gca().invert_yaxis()
    if title:
        plt.title(title)

    if output_path is not None:
        plt.savefig(output_path, bbox_inches="tight", pad_inches=0)


def dataframe_to_array(dataframe):
    # compute the average cell density
    densities = []
    for image in np.unique(dataframe.image):
        image_data = dataframe[dataframe["image"] == image].to_numpy()[:, [1, 2]]
        densities.append(image_data)
    return np.array(densities, dtype=np.float32)


def plot_density_per_layer(
    _layer_df, output_path=None, title="Cell density per layer", distiguish=True
):
    densities = _layer_df.to_numpy()
    mean = densities.mean(axis=0)
    std = densities.std(axis=0)
    columns = list(_layer_df.columns)
    N = densities.shape[1]
    ind = np.arange(N)  # the x locations for the groups
    width = 0.7  # the width of the bars: can also be len(x) sequence

    bar_colors = get_color(
        distiguish=distiguish, return_type="list", return_unit="float"
    )
    plt.figure(figsize=(5, 5))
    p1 = plt.bar(ind, mean, width, yerr=std, color=bar_colors)
    # plt.errorbar(ind, mean, color="b")
    plt.ylabel("Cell density (cells/mm3)")
    plt.title(title)
    plt.xticks(ind, columns)
    current_values = plt.gca().get_yticks()
    _ = plt.gca().set_yticklabels(["{:.1e}".format(x) for x in current_values])
    # plt.yticks(np.arange(0, 81, 10))
    # plt.legend((p1[0],('Men')))
    if output_path:
        plt.savefig(output_path, bbox_inches="tight", pad_inches=0)


if __name__ == '__main__':

    

    if len(sys.argv) < 5 or len(sys.argv) > 6:
        print('usage: python cell_density.py per_depth_path per_layer_merged_path per_layer_distinguish_path\
        output_figure_path [metadata_file_path]')
        sys.exit()

    if len(sys.argv) == 5 or len(sys.argv) == 6:
        per_depth_path = sys.argv[1] 
        per_layer_merged_path = sys.argv[2] 
        per_layer_distinguish_path = sys.argv[3]
        output_figure_path = sys.argv[4]

    if len(sys.argv) == 6:
        metadata_file_path = sys.argv[5]
    else:
        metadata_file_path = None
    

    print(f'per_depth_path {per_depth_path}')
    print(f'per_layer_merged_path {per_layer_merged_path}')
    print(f'per_layer_distinguish_path {per_layer_distinguish_path}')
    print(f'output_figure_path {output_figure_path}')
    print(f'metadata_file_path {metadata_file_path}')


    file_list = glob.glob(per_depth_path + "/*.csv")
    density_df = concate_density_dataframes(file_list)


    print(f"The dataframe contains {np.unique(density_df.image).size} images")
    try:
        os.makedirs(output_figure_path)
    except OSError as error:
        pass 

    # Cell densities as function of brain depth
    data = dataframe_to_array(density_df)
    plot(
        data,
        "Cell density as a function of SSCX region percentage of depth.",
        plot_median=True,
        plt_detail=False,
        output_path= output_figure_path + "/median_density_percentage.svg",
    )


    print(f"Plot {np.unique(density_df.image).size} images included the data")
    plot(
        data,
        "Cell density as a function of SSCX region percentage of depth.",
        plt_detail=True,
        output_path=output_figure_path + "/full_density_percentage.svg",
    )


    print(f"Plot {np.unique(density_df.image).size} images included the data")
    plot_mean_and_std_dev(
        density_df,
        title="Cell density as a function of percentage of depth of the S1HL brain region",
        output_path=output_figure_path + "/full_std_density_percentage.svg",
    )

    if metadata_file_path is not None:
        meta_df = pd.read_csv(metadata_file_path, index_col=0)

        analyse_df = meta_df[meta_df.Analyze == True]
        left_meta_df = analyse_df[analyse_df["hemisphere(L/R)"] == "left"]
        right_meta_df = analyse_df[analyse_df["hemisphere(L/R)"] == "right"]


        left_image_id = list(left_meta_df["Image_Name"])
        right_image_id = list(right_meta_df["Image_Name"])
        left_density_df = get_filtered_density_df(left_image_id, density_df)
        right_density_df = get_filtered_density_df(right_image_id, density_df)


        data = dataframe_to_array(left_density_df)
        print(
            f"Plot {np.unique(left_density_df.image).size} images included the left_density_df"
        )
        plot(
            data,
            "Left Hemisphere cell density as a function of SSCX region percentage of depth.",
            plt_detail=True,
            output_path=output_figure_path + "/left_density_percentage.svg",
        )


        data = dataframe_to_array(right_density_df)
        print(
            f"Plot {np.unique(right_density_df.image).size} images included the right_density_df"
        )
        plot(
            data,
            "Right Hemisphere cell density as a function of SSCX region percentage of depth",
            plt_detail=True,
            output_path=output_figure_path + "/right_std_density_percentage.svg",
        )


        print(
            f"Plot {np.unique(left_density_df.image).size} images included the left_density_df"
        )
        print(
            f"Plot {np.unique(right_density_df.image).size} images included the right_density_df"
        )
        plot_mean_and_std_dev(
            [left_density_df, right_density_df],
            labels=["left", "right"],
            colors=["blue", "red"],
            title="Cell density as a function of percentage of depth of the S1HL brain region",
            output_path=output_figure_path + "/left_right_std_density_percentage.svg",
        )


        project_ID_list = np.unique(analyse_df["Project_ID"])


        for project_id in project_ID_list:
            animal_meta_df = analyse_df[analyse_df["Project_ID"] == project_id]
            animal_image_id = list(animal_meta_df["Image_Name"])
            animal_density_df = get_filtered_density_df(animal_image_id, density_df)
            data = dataframe_to_array(animal_density_df)
            plot_mean_and_std_dev(
                [animal_density_df],
                title=f"{project_id} Cell density as a function of percentage of depth of the S1HL brain region",
            )
            plot(
                data,
                f"{project_id} cell density as a function of SSCX region percentage of depth",
                plt_detail=True,
                output_path=f"{output_figure_path}/{project_id}_std_density_percentage.svg",
            )


        animal_ID_list = ["ProjectQuPath"]


        animal_df_list = []
        for project_id in animal_ID_list:
            animal_meta_df = analyse_df[analyse_df["Project_ID"] == project_id]
            animal_image_id = list(animal_meta_df["Image_Name"])
            animal_df_list.append(get_filtered_density_df(animal_image_id, density_df))


        animal_df = pd.concat(animal_df_list)


        data = dataframe_to_array(animal_df)
        plot_mean_and_std_dev(
            [animal_df],
            title=f"animal 1413828 Cell density as a function of percentage of depth of the S1HL brain region",
        )
        plot(
            data,
            f"animal 1413828 cell density as a function of SSCX region percentage of depth",
            plt_detail=True,
            output_path=f"{output_figure_path}/animal_01413828_density_df_std_density_percentage.svg",
        )


    ## Cell density mean value per image cells/mm3

    nb_images = len(density_df[density_df.depth_percentage == 0.00])
    print(
        f"mean density on {nb_images} images => {density_df.densities.mean():.2f} cells/mm3"
    )


    # Cell densities per layers

    ## Merged Layers 2/3
    path = per_layer_merged_path
    layer_df = get_per_layer_df(path)
    print(f"Plot {len(layer_df)} images included the layer_df")
    plot_density_per_layer(
        layer_df,
        title="Cell density per layer (Merged L2/L3)",
        output_path=output_figure_path + "/per_layer_merge_23.svg",
        distiguish=False,
    )


    ## Distinguish Layer 2 and 3
    path_d = per_layer_distinguish_path
    d_layer_df = get_per_layer_df(path_d)


    print(f"Plot {len(d_layer_df)} images included the d_layer_df")
    plot_density_per_layer(
        d_layer_df,
        title="Cell density per layer (Distinguishable L2/L3)",
        output_path=output_figure_path + "/per_layer_distinguish_23.svg",
    )
