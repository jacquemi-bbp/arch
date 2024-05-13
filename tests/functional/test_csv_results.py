"""This tests aims at checking the integrity of the circuit built from the snakemake."""

import filecmp


def test_densities():
    assert filecmp.cmp(
        "./arch_results/output_path_batch/distinguish/SLD_0000736.vsi-20x_01_per_layer.csv",
        "./expected/distinguish/SLD_0000736.vsi-20x_01_per_layer.csv",
    )
    assert filecmp.cmp(
        "./arch_results/output_path_batch/distinguish/SLD_0000736.vsi-20x_02_per_layer.csv",
        "./expected/distinguish/SLD_0000736.vsi-20x_02_per_layer.csv",
    )

    assert filecmp.cmp(
        "./arch_results/output_path_batch/merged/SLD_0000736.vsi-20x_01_per_layer.csv",
        "./expected/merged/SLD_0000736.vsi-20x_01_per_layer.csv",
    )
    assert filecmp.cmp(
        "./arch_results/output_path_batch/merged/SLD_0000736.vsi-20x_02_per_layer.csv",
        "./expected/merged/SLD_0000736.vsi-20x_02_per_layer.csv",
    )

    assert filecmp.cmp(
        "./arch_results/output_path_batch/PerDepth/SLD_0000736.vsi-20x_01.csv",
        "./expected/PerDepth/SLD_0000736.vsi-20x_01.csv",
    )
    assert filecmp.cmp(
        "./arch_results/output_path_batch/PerDepth/SLD_0000736.vsi-20x_02.csv",
        "./expected/PerDepth/SLD_0000736.vsi-20x_02.csv",
    )
