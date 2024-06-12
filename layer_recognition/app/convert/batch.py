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


import click

from layer_recognition.convert import single_image_conversion
from layer_recognition.io import list_images
from layer_recognition.utilities import get_config


@click.command()
@click.option("--config-file-path", required=False, help="Configuration file path")
def cmd(config_file_path):
    """
    Convert QuPath output files to pandas dataframes
    Args:
        config-file-path (str): The configuration file path
    """
    (
        input_detection_directory,
        cell_position_suffix,
        input_annotation_directory,
        annotations_geojson_suffix,
        exclude_flag,
        pixel_size,
        output_path,
    ) = get_config(config_file_path)
    images_dictionary = list_images(
        input_detection_directory,
        cell_position_suffix,
        input_annotation_directory,
        annotations_geojson_suffix,
    )

    for image_prefix in images_dictionary.keys():
        print(f"INFO: Process single image {image_prefix}")
        single_image_conversion(
            output_path,
            image_prefix,
            input_detection_directory,
            input_annotation_directory,
            pixel_size,
            exclude=exclude_flag,
        )
