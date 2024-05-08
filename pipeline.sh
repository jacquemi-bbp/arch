#!/bin/bash

QUPATH_PROJECT_PATH="../ProjectQuPath_for_test/ProjectQuPath_for_test.qpproj"
RESULT_PATH="/arch_results"
FIGURE_PATH="/arch_results/Figures"

printf "\nPIPELINE INFO: QuPath: Detect cells, export annotations and cells features\n"
#qupath script ./qupath_scripts/full_QuPath_script.groovy -p $QUPATH_PROJECT_PATH

printf "\nPIPELINE INFO: Convert cells features and annotation to pandas dataframes\n"
pyarch convert --config-file-path ./Config/batch_convert.ini

printf "\nPIPELINE INFO: Convert QuPath project metadata to a pandas dataframes\n"
pyarch convert-qupath-project --qupath-project-path $QUPATH_PROJECT_PATH --output-path $RESULT_PATH

printf "\nPIPELINE INFO: Compute cell densities as function of brain depth\n"
pyarch density-per-depth --config-file-path ./Config/batch_density_depth.ini

printf "\nPIPELINE INFO: Predict images layers\n"
pyarch predict_layer --config-file-path ./Config/batch_predict.ini

printf "\nPIPELINE INFO: Compute cell densities per Layer with L2 and L3 merged\n"
pyarch density-per-layer --config-file-path ./Config/batch_density_layer_merged.ini

printf "\nPIPELINE INFO: Compute cell densities per Layer with L2 and L3 distinguished\n"
pyarch density-per-layer --config-file-path ./Config/batch_density_layer_distinguish.ini

printf "\nPIPELINE INFO: Concatenate Dataframe for Cell area\n"
pyarch cell-size --config-file-path Config/batch_size.ini

printf "\nPIPELINE INFO: Produces cell densities and cell sizes figures\n"
python figures_script/cells_density.py $RESULT_PATH/output_path_batch/PerDepth $RESULT_PATH/output_path_batch/merged/ $RESULT_PATH/output_path_batch/distinguish/ $FIGURE_PATH ./data/metadata.csv
python figures_script/cells_size.py /arch_results/output_path_batch/cell_area/cells_area.csv /tmp/cell_size.png per_layer
printf "\n----- Done ------"
