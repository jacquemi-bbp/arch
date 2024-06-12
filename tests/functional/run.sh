#!/usr/bin/env bash

RESULT_PATH="./arch_results"
FIGURE_PATH="./arch_results/Figures"
QUPATH_PROJECT_PATH=../data/ProjectQuPath_for_test.qpproj
TRAINING_PATH=../data/ml-training
PREDICTION_PATH=../data/ml-pred

rm -rf $RESULT_PATH

printf "\nPIPELINE INFO: Convert cells features and annotation to pandas dataframes\n"
pylayer_recognition -v convert --config-file-path ../Config/batch_convert.ini

printf "\nPIPELINE INFO: Convert QuPath project metadata to a pandas dataframes\n"
pylayer_recognition -v convert-qupath-project --qupath-project-path $QUPATH_PROJECT_PATH --output-path $RESULT_PATH

printf "\nPIPELINE INFO: Train a Machine Learning model\n"
pylayer_recognition -v train-model --train-dir $TRAINING_PATH --train-glob "Feat*" --extension csv  --save-dir $RESULT_PATH --distinguishable-second-layer

printf "\nPIPELINE INFO: Predict the cells layer\n"
pylayer_recognition -v  layers-predict --model-file $RESULT_PATH/trained_rf.pkl  --pred-dir $PREDICTION_PATH --pred-save  $RESULT_PATH  --pred-glob "Feat*" --distinguishable-second-layer

printf "\nPIPELINE INFO: Compute cell densities as function of brain depth\n"
pylayer_recognition density-per-depth --config-file-path ../Config/batch_density_depth.ini

printf "\nPIPELINE INFO: Compute cell densities per Layer with L2 and L3 merged\n"
pylayer_recognition -v density-per-layer --config-file-path ../Config/batch_density_layer_merged.ini

printf "\nPIPELINE INFO: Compute cell densities per Layer with L2 and L3 distinguished\n"
pylayer_recognition -v density-per-layer --config-file-path ../Config/batch_density_layer_distinguish.ini

printf "\nPIPELINE INFO: Produces cell densities figures\n"
python ../../figures_script/cells_density.py $RESULT_PATH/output_path_batch/PerDepth $RESULT_PATH/output_path_batch/merged/ $RESULT_PATH/output_path_batch/distinguish/ $FIGURE_PATH ../data/metadata.csv
printf "\n----- Done ------"

pytest test_csv_results.py

TRAINING_PATH=../data/ml-training