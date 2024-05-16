""" test the configuration parser"""

from pathlib import Path
import arch.utilities as tested

CONFIG_DIR = Path(__file__).resolve().parent.parent / "Config"


def test_converter_config():
    
    config_file_path = CONFIG_DIR / "batch_convert.ini"
    print(f'DEBUG config_file_path {config_file_path}')
    (
        input_detection_directory,
        cell_position_suffix,
        input_annotation_directory,
        annotations_geojson_suffix,
        exclude_flag,
        pixel_size,
        output_path,
    ) = tested.get_config(config_file_path)

    assert input_detection_directory == "../data"
    assert input_annotation_directory == "../data"
    assert output_path == "./arch_results/converted/"
    assert pixel_size == 0.3460130331522824
    assert cell_position_suffix == "Detections.txt"
    assert annotations_geojson_suffix == "_annotations.json"
    assert exclude_flag==True


