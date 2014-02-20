import json
import collections

from src.file_parsing.csv_parsing import parse_csv_structured
from src.transformations.array_transformations import structured_array_to_ndarray
from src.transformations.text_transformations import transform_text_to_numbers_structured


def load_paths(data_file_path, config_file_path):
    with open(data_file_path, "r") as data_file, \
            open(config_file_path, "r") as config_file:
        return load(data_file, config_file)


def load(data_file, config_file):
    config = json.load(config_file, object_pairs_hook=collections.OrderedDict)

    data = parse_csv_structured(data_file, config)
    data = transform_text_to_numbers_structured(data, config)
    data = structured_array_to_ndarray(data)

    return data





