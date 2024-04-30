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
import click

from arch.io import (
    #write_dataframe_to_file,
    list_images,
    #get_qpproject_images_metadata,
    #save_dataframe_without_space_in_path,
)
'''
from arch.convert import (
    convert,
)
from arch.utilities import (
    get_image_lateral,
    get_image_animal,
    get_image_immunohistochemistry,
)
'''
from arch.convert import single_image_conversion


@click.command()
@click.option("--config-file-path", required=False, help="Configuration file path")
def cmd(config_file_path):

    config = configparser.ConfigParser()
    config.sections()
    config.read(config_file_path)

    try:
        input_detection_directory = config["BATCH"]["input_detection_directory"]
        cell_position_suffix = config["BATCH"]["cell_position_suffix"].replace('"', "")
    except KeyError:
        input_detection_directory = None
        cell_position_suffix = None
    try:
        input_annotation_directory = config["BATCH"]["input_annotation_directory"]
        annotations_geojson_suffix = config["BATCH"]["annotations_geojson_suffix"]
    except KeyError:
        input_annotation_directory = None
        annotations_geojson_suffix = None

    try:
        exclude_flag = config.getboolean('BATCH', 'exclude')
    except configparser.NoOptionError:
        exclude_flag = True


    pixel_size = float(config["BATCH"]["pixel_size"])

    try:
        qupath_project_path = config["BATCH"]["qpproj_path"]
    except KeyError:
        qupath_project_path = None
    output_path = config["BATCH"]["output_directory"]

    images_dictionary = list_images(
        input_detection_directory, cell_position_suffix, 
        input_annotation_directory, annotations_geojson_suffix
    )

    for image_prefix, values in images_dictionary.items():
        print("INFO: Process single image {}".format(image_prefix))
        single_image_conversion(output_path, image_prefix,
                                input_detection_directory, input_annotation_directory,
                                pixel_size, exclude = exclude_flag)
