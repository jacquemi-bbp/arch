
[![Build status](https://github.com/jacquemi-bbp/arch/actions/workflows/run-tox.yml/badge.svg?branch=main)](https://github.com/jacquemi-bbp/arch/actions)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License](https://img.shields.io/badge/License-GPLv3-blue)](https://github.com/BlueBrain/NeuroTS/blob/main/LICENSE.txt)
[![Documentation status](https://readthedocs.org/projects/bbparch/badge/?version=latest)](https://arch.readthedocs.io/)
[![DOI](https://img.shields.io/badge)](https://doi.org/)


# ARCH

Automated Recognition and Classification of Histological layers.
Machine learning for histological annotation and quantification of cortical layers pipeline 

## Main usage
Automatically detects brain cells, classithem them by brain layer and compute cells density.
This pipeline has been used for rat somatosensory cortex Nissl microscopy images, provided by the EPFL LNMC laboratory.

From these images imported in QuPath projects, some QuPath annotations made by experts and metadata in cvs files, this package can generate: 
   - The somatosensory cortex S1HL layers boundaries.

<img src="docs/source/images/layer_boundaries.png" alt="Doc/layer_boundaries.png" width="200"/>

 -  Cells densities as a function of the percentage of depth inside the somatosensory cortex S1HL.
 
<img src="docs/source/images/percentage_of_depth.png" alt="Doc/percentage_of_depth.png" width="200"/>

 - The per layer cells densities in the somatosensory cortex S1HL.

<img src="docs/source/images/per_layer_distinguish_23.png" alt="per_layer_distinguish_23.png" width="200"/>
 
## The pipeline consists of two main steps:
1. Within QuPath (Third party application), execute cells detection and export cells features and annotations.
2. Processing the data exported by QuPath during the previous step, in order to compute the cells density.
 
## Lexicon
The following definitions will stay in effect throughout the code.
- S1HL: The rat Hindlimb Somatosensory
- Annotation: QuPath annotation
- ML: Machine learning


## Installation
- QuPath: https://qupath.github.io/
  - QuPath cellpose extension https://github.com/BIOP/qupath-extension-cellpose
  - cellpose extemsion for qupath. Follow the official instructions: https://cellpose.readthedocs.io/en/latest/
 
- Python package and its applications.
```shell
$ git clone https://github.com/jacquemi-bbp/arch.git
$ cd arch
$ pip install .
```

## Third parties 
### Python package
- python third parties libraries are installed during package installation.
see requirements.txt
### QuPath

##  Input data
### Input data for groovy script
- [cellpose model](cellpose_model/cellpose_residual_on_style_on_concatenation_off_train_2022_01_11_16_14_20)

- QuPath project including the images to process and these 5 annotations: S1HL, top_left, top_right, bottom_left and bottom_right 
- cellpose model used in Full_QuPath_script.groovy script to detect cells

### Input data for python single image processing
- The generated data by the Full_QuPath_script.groovy script
  - detected cells features (csv file)
  - annotations file (json file)
    - top_left, top_right, bottom_left and bottom_right annotation points 
    - S1HL polygon annotation
    - outside_pia annotation
- pixel size :  a float number that represents the pixel size of the QuPath input inages


## Export Cells features and QuPath annotations
- Modify the pathes inside ./Layer Classiffier.json to make it corresponding to your environment.
- Execute the following groovy script inside the QuPath application or via a script thanks to the QuPath script command:
qupath_scripts/full_QuPath_script.groovy

## Compute the cells densities as a function of percentage of the S1HL depth processing 
- Read input data from QuPath exported files
- Convert annotations to cartesian point coordinates and shapely polygon.
- Split the S1HL polygon following the S1HL "top and bottom lines" shapes in n polygons (named spitted_polygon)
- Count the number of cells located in each spitted_polygon
- Compute the volume of each spitted_polygon (mm3)
- Compute the cells densities as function of the percentage of the sscx depth
- Export result files

## Compute the densities per S1HL layers
- Read input data from QuPath exported files
- Train a ML model from GroundTruth data produced by some experts
- Use the ML model to predict and affect a layer for each detected cell
- Define a polygon (alphashape) for each layer based on ML prediction
- Count the number of cells located in each layer polygon
- Compute the volume of each layer polygon (mm3)
- Compute the cells densities for each layer
- Export result files
- 
## Compute the densities of one image
- modify ./Config/linux/depth.ini with your configuration
- execute the python script
```shell
$ pyqupath_processing density --config-file-path ./Config/linux/density.ini
```

# Compute the densities of several images in batch
- modify ./Config/linux/batch_density.ini with your configuration
- execute the python script
```shell
$ pyqupath_processing density --config-file-path ./Config/linux/batch_density.ini
```

## Compute the cells density per layers


## Funding & Acknowledgment

The development of this software was supported by funding to the Blue Brain Project, a research center of the École polytechnique fédérale de Lausanne (EPFL), from the Swiss government’s ETH Board of the Swiss Federal Institutes of Technology.

For license and authors, see `LICENSE.txt` and `AUTHORS.md` respectively.

Copyright © 2022 Blue Brain Project/EPFL
