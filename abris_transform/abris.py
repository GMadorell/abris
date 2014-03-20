from sklearn_pandas import DataFrameMapper
from abris_transform.dataframe_manipulation.split_dataframe import split_dataframe_train_test

from abris_transform.parsing.csv_parsing import prepare_csv_to_dataframe
from abris_transform.transformations.mapping import DataFrameMapCreator
from abris_transform.configuration.configuration import Configuration
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
        self.__mapper = None

    def prepare(self, data_file):
        """
        Called with the training data.
        """
        data = prepare_csv_to_dataframe(data_file, self.__config)
        train, test = split_dataframe_train_test(data, self.__config.get_option_parameter("split", "train_percentage"))

        mapping = DataFrameMapCreator().get_mapping_from_config(self.__config)
        self.__mapper = DataFrameMapper(mapping)

        return self.__get_correct_return_parameters(train, test)

    def __get_correct_return_parameters(self, train, test):
        model = self.__config.get_data_model()

        train_transformed = self.__mapper.fit_transform(train)
        test_transformed = self.__mapper.transform(test)

        if model.has_target():
            return self.__add_target_data(train_transformed, train), \
                   self.__add_target_data(test_transformed, test)
        else:
            return train_transformed, test_transformed

    def __add_target_data(self, transformed_data, original_data):
        """
        Picks up the target data from the original_data and appends it as a
        column to the transformed_data.
        Both arguments are expected to be np.array's.
        """
        model = self.__config.get_data_model()
        name = model.find_target_feature().get_name()

        target = original_data[name].values.astype(type_name_to_data_type("float"))
        target = target[..., None]

        return np.hstack((transformed_data, target))

    def apply(self, data_file):
        """
        Called with the predict data (new information).
        """
        data = prepare_csv_to_dataframe(data_file, self.__config, use_target=False)
        data = self.__mapper.transform(data)
        return data





