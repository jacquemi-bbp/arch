#!/usr/bin/env bash

RESULT_PATH="./arch_results"
FIGURE_PATH="./arch_results/Figures"
QUPATH_PROJECT_PATH=../data/ProjectQuPath_for_test.qpproj

rm -rf $RESULT_PATH

printf "\nPIPELINE INFO: Convert cells features and annotation to pandas dataframes\n"
pyarch convert --config-file-path ../Config/batch_convert.ini

printf "\nPIPELINE INFO: Convert QuPath project metadata to a pandas dataframes\n"
pyarch convert-qupath-project --qupath-project-path $QUPATH_PROJECT_PATH --output-path $RESULT_PATH

printf "\nPIPELINE INFO: Compute cell densities as function of brain depth\n"
pyarch density-per-depth --config-file-path ../Config/batch_density_depth.ini

printf "\nPIPELINE INFO: Compute cell densities per Layer with L2 and L3 merged\n"
pyarch density-per-layer --config-file-path ../Config/batch_density_layer_merged.ini

printf "\nPIPELINE INFO: Compute cell densities per Layer with L2 and L3 distinguished\n"
pyarch density-per-layer --config-file-path ../Config/batch_density_layer_distinguish.ini

printf "\nPIPELINE INFO: Produces cell densities figures\n"
python ../../figures_script/cells_density.py $RESULT_PATH/output_path_batch/PerDepth $RESULT_PATH/output_path_batch/merged/ $RESULT_PATH/output_path_batch/distinguish/ $FIGURE_PATH ../data/metadata.csv
printf "\n----- Done ------"

pytest test_csv_results.py
