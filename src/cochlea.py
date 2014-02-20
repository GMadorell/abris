import json
import collections

from src.file_parsing.csv_parsing import parse_csv_structured
from src.transformations.array_transformations import structured_array_to_ndarray
from src.transformations.one_hot_encoding import OneHotEncodingTransformer
from src.transformations.text_transformations import TextToNumberStructuredTransformer


class Cochlea(object):
    def __init__(self, config_file):
        self.__config = json.load(config_file, object_pairs_hook=collections.OrderedDict)
        self.__text_to_number_structured_transformer = None
        self.__one_hot_encoding_transformer = None

    def fit_transform(self, data_file):
        self.__text_to_number_structured_transformer = TextToNumberStructuredTransformer(self.__config)
        self.__one_hot_encoding_transformer = OneHotEncodingTransformer(self.__config)

        data = parse_csv_structured(data_file, self.__config)
        data = self.__text_to_number_structured_transformer.fit_transform(data)
        data = structured_array_to_ndarray(data)
        data = self.__one_hot_encoding_transformer.fit_transform(data)

        return data

    def transform(self, data_file):
        data = parse_csv_structured(data_file, self.__config)
        data = self.__text_to_number_structured_transformer.transform(data)
        data = structured_array_to_ndarray(data)
        data = self.__one_hot_encoding_transformer.transform(data)

        return data





