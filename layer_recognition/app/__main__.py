"""
Processing for rat somatosensory cortex QuPath Nissl data
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
#


import logging

import click

from layer_recognition.app.cell_size.batch import cmd as batch_cell_size
from layer_recognition.app.convert.batch import cmd as batch_convert
from layer_recognition.app.convert.project import cmd as project_convert
from layer_recognition.app.density.batch import cmd_depth as batch_density_depth
from layer_recognition.app.density.batch import cmd_layer as batch_density_layer
from layer_recognition.app.density.batch import cmd_animal as batch_density_animal
from layer_recognition.app.train_predict.predict import cmd as predict
from layer_recognition.app.train_predict.train import cmd as train
from layer_recognition.app.layer_thickness.thickness import cmd as layer_thickness
from layer_recognition.app.logger import setup_logging
from layer_recognition.version import VERSION


@click.group("pylayer_recognition", help=__doc__.format(esc="\b"))
@click.option("-v", "--verbose", count=True, help="-v for INFO, -vv for DEBUG")
@click.version_option(VERSION)
def app(verbose=0):
    # pylint: disable=missing-docstring
    setup_logging(
        {
            0: logging.WARNING,
            1: logging.INFO,
            2: logging.DEBUG,
        }[min(verbose, 2)]
    )


app.add_command(name="convert", cmd=batch_convert)
app.add_command(name="convert-qupath-project", cmd=project_convert)
app.add_command(name="density-per-layer", cmd=batch_density_layer)
app.add_command(name="density-per-depth", cmd=batch_density_depth)
app.add_command(name="density-per-animal", cmd=batch_density_animal)
app.add_command(name="cell-size", cmd=batch_cell_size)
app.add_command(name="layer-thickness", cmd=layer_thickness)
app.add_command(name="layers-predict", cmd=predict)
app.add_command(name="train-model", cmd=train)


if __name__ == "__main__":
    app()
