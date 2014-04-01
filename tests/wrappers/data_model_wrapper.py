from abris_transform.configuration.data_model import DataModel

import pandas as pd


class DataModelWrapper(object):
    def __init__(self):
        self.__model_dict = {}
        self.__dataframe_dict = {}

        self.dataframe = None
        self.data_model = None

        self.clean()

    def clean(self):
        self.__model_dict = {}
        self.__dataframe_dict = {}
        self.rebuild()

    def rebuild(self):
        self.dataframe = pd.DataFrame(self.__dataframe_dict)
        self.data_model = DataModel(self.__model_dict)
        self.data_model.set_features_types_from_dataframe(self.dataframe)

    def add_numerical_feature(self, number=1):
        self.__add(number, "numerical_feature", ["Numerical"], [1, 2])

    def add_boolean_feature(self, number=1):
        self.__add(number, "boolean_feature", ["Boolean"], [True, False])

    def add_ignored_numerical_feature(self, number=1):
        self.__add(number, "ignored_numerical_feature", ["Numerical", "Ignore"], [1, 2])

    def add_categorical_numerical_feature(self, number=1):
        self.__add(number, "categorical_numerical_feature", ["Categorical"], [1, 2])

    def add_categorical_text_feature(self, number=1):
        self.__add(number, "categorical_text_feature", ["Categorical"], ["Hi", "Hello"])

    def add_custom_text_feature(self, values, number=1):
        self.__add(number, "custom_text_feature", ["Categorical"], values)

    def add_numerical_target(self, number=1):
        self.__add(number, "numerical_target", ["Numerical", "Target"], [1, 2])

    def add_categorical_target(self, number=1):
        self.__add(number, "categorical_target", ["Categorical", "Target"], [1, 2])

    def __add(self, number, name, caracteristics, values):
        self.__model_dict.update({
            name + str(number): caracteristics
        })
        self.__dataframe_dict.update({
            name + str(number): pd.Series(values)
        })
        self.rebuild()
