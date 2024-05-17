"""This tests aims at checking the integrity of the circuit built from the snakemake."""

import filecmp
from pathlib import Path

CURRENT_RESULT_DIR = Path(__file__).resolve().parent / "arch_results"
EXPECTED_DIR = Path(__file__).resolve().parent / "expected"


def test_densities():
    assert filecmp.cmp(
        CURRENT_RESULT_DIR/"output_path_batch/distinguish/SLD_0000736.vsi-20x_01_per_layer.csv",
        EXPECTED_DIR/"distinguish/SLD_0000736.vsi-20x_01_per_layer.csv",
    )
    assert filecmp.cmp(
        CURRENT_RESULT_DIR/"output_path_batch/distinguish/SLD_0000736.vsi-20x_02_per_layer.csv",
        EXPECTED_DIR/"distinguish/SLD_0000736.vsi-20x_02_per_layer.csv",
    )

    assert filecmp.cmp(
        CURRENT_RESULT_DIR/"output_path_batch/merged/SLD_0000736.vsi-20x_01_per_layer.csv",
        EXPECTED_DIR/"merged/SLD_0000736.vsi-20x_01_per_layer.csv",
    )
    assert filecmp.cmp(
        CURRENT_RESULT_DIR/"output_path_batch/merged/SLD_0000736.vsi-20x_02_per_layer.csv",
        EXPECTED_DIR/"merged/SLD_0000736.vsi-20x_02_per_layer.csv",
    )

    assert filecmp.cmp(
        CURRENT_RESULT_DIR/"output_path_batch/PerDepth/SLD_0000736.vsi-20x_01.csv",
        EXPECTED_DIR/"PerDepth/SLD_0000736.vsi-20x_01.csv",
    )
    assert filecmp.cmp(
        CURRENT_RESULT_DIR/"output_path_batch/PerDepth/SLD_0000736.vsi-20x_02.csv",
        EXPECTED_DIR/"PerDepth/SLD_0000736.vsi-20x_02.csv",
    )
    
def test_prediction():
    assert filecmp.cmp(
        CURRENT_RESULT_DIR /"Features_SLD_0000733.vsi-20x_01.csv",
        EXPECTED_DIR/"Predictions/Features_SLD_0000733.vsi-20x_01.csv",
    )
