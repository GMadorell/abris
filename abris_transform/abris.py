from sklearn_pandas import DataFrameMapper
from abris_transform.dataframe_manipulation.split_dataframe import split_dataframe_train_test

from abris_transform.parsing.csv_parsing import prepare_csv_to_dataframe
from abris_transform.transformations.mapping import DataFrameMapCreator
from abris_transform.configuration.configuration import Configuration
from abris_transform.transformations.transformer import Transformer
from abris_transform.type_manipulation.translation.data_type_translation import type_name_to_data_type

import numpy as np


class Abris(object):
    """
    Main entry class for the whole preprocessing engine (and probably the only one that needs to be used
    if no more features are needed).
    """

    def __init__(self, config_file):
        """
        @param config_file: File like object containing the configuration.
        """
        self.__config = Configuration(config_file)
        self.__transformer = None

    def prepare(self, data_file):
        """
        Called with the training data.
        """
        data = prepare_csv_to_dataframe(data_file, self.__config)
        self.__transformer = Transformer(self.__config)
        return self.__transformer.prepare(data)

    def apply(self, data_file):
        """
        Called with the predict data (new information).
        """
        data = prepare_csv_to_dataframe(data_file, self.__config, use_target=False)
        data = self.__transformer.apply(data)
        return data





