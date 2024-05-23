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

import sys

import pandas as pd

from arch.visualisation import plots_layer_thickness

if __name__ == "__main__":
    if len(sys.argv) != 3 :
        print(
            "usage: python layer_thickness.py layer_thickness_dataframe_path output_directory_path"
        )
        sys.exit()

    layer_thickness_dataframe_path = sys.argv[1]
    output_directory_path = sys.argv[2]

    layer_thickness_dataframe_path = pd.read_csv(layer_thickness_dataframe_path)
    plots_layer_thickness(layer_thickness_dataframe_path, output_path=output_directory_path)
