from abris_transform.configuration.configuration import Configuration
from abris_transform.parsing.csv_parsing import apply_csv_structured, prepare_csv_structured

from abris_transform.transformations.array_transformations import structured_array_to_ndarray
from abris_transform.transformations.boolean_transformations import BooleanToNumberTransformer
from abris_transform.transformations.normalize import NormalizeTransformer
from abris_transform.transformations.one_hot_encoding import OneHotEncodingTransformer
from abris_transform.transformations.text_transformations import TextToNumberStructuredTransformer


class Abris(object):
    """
    Main entry class for the whole preprocessing engine (and probably the only one that needs to be used
    if no more features are needed).
    """
    def __init__(self, config_file):
        self.__config = Configuration(config_file)
        self.__text_to_number_structured_transformer = None
        self.__boolean_to_number_transformer = None
        self.__one_hot_encoding_transformer = None
        self.__normalizer = None

    def prepare(self, data_file):
        """
        Called with the training data.
        """
        self.__text_to_number_structured_transformer = TextToNumberStructuredTransformer(self.__config)
        self.__boolean_to_number_transformer = BooleanToNumberTransformer(self.__config)
        self.__one_hot_encoding_transformer = OneHotEncodingTransformer(self.__config)
        if self.__config.is_option_enabled("normalize"):
            self.__normalizer = NormalizeTransformer(self.__config)

        data = prepare_csv_structured(data_file, self.__config)
        data = self.__text_to_number_structured_transformer.fit_transform(data)
        data = self.__boolean_to_number_transformer.fit_transform(data)
        data = structured_array_to_ndarray(data)
        if self.__config.is_option_enabled("normalize"):
            data = self.__normalizer.fit_transform(data)
        data = self.__one_hot_encoding_transformer.fit_transform(data)

        if self.__config.get_data_model().has_target():
            return self.split_last_column(data)
        else:
            return data

    def split_last_column(self, data):
        return data[:, :-1], data[:, -1]

    def apply(self, data_file):
        """
        Called with the predict data (new information).
        """
        data = apply_csv_structured(data_file, self.__config)
        data = self.__text_to_number_structured_transformer.transform(data)
        data = self.__boolean_to_number_transformer.transform(data)
        data = structured_array_to_ndarray(data)
        if self.__config.is_option_enabled("normalize"):
            data = self.__normalizer.transform(data)
        data = self.__one_hot_encoding_transformer.transform(data)

        return data





