from sklearn_pandas import DataFrameMapper
from abris_transform.dataframe_manipulation.split_dataframe import split_dataframe_train_test
from abris_transform.transformations.mapping import DataFrameMapCreator
from abris_transform.type_manipulation.translation.data_type_translation import type_name_to_data_type
import numpy as np


class Transformer(object):
    def __init__(self, config):
        self.__config = config
        mapping = DataFrameMapCreator().get_mapping_from_config(config)
        self.__mapper = DataFrameMapper(mapping)

    def prepare(self, dataframe):
        train, test = split_dataframe_train_test(dataframe, self.__config.get_option_parameter("split", "train_percentage"))

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

    def apply(self, dataframe):
        return self.__mapper.transform(dataframe)


