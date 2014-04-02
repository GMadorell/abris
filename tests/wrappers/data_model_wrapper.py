from abris_transform.configuration.data_model import DataModel

import pandas as pd


class DataModelWrapper(object):
    def __init__(self):
        self.__model_dict = {}
        self.__dataframe_dict = {}

        self.dataframe = None
        self.data_model = None

        self.__count = 0

        self.clean()

    def clean(self):
        self.__model_dict = {}
        self.__dataframe_dict = {}
        self.__count = 0
        self.rebuild()

    def rebuild(self):
        self.dataframe = pd.DataFrame(self.__dataframe_dict)
        self.data_model = DataModel(self.__model_dict)
        self.data_model.set_features_types_from_dataframe(self.dataframe)

    def add_numerical_feature(self, values=None):
        if not values: values = [1, 2, 3, 4]
        self.__add("numerical_feature", ["Numerical"], values)
        return self

    def add_boolean_feature(self, values=None):
        if not values: values = [True, False, False, True]
        self.__add("boolean_feature", ["Boolean"], values)
        return self

    def add_ignored_numerical_feature(self, values=None):
        if not values: values = [1, 2, 3, 4]
        self.__add("ignored_numerical_feature", ["Numerical", "Ignore"], values)
        return self

    def add_categorical_numerical_feature(self, values=None):
        if not values: values = [1, 2, 3, 4]
        self.__add("categorical_numerical_feature", ["Categorical"], values)
        return self

    def add_categorical_text_feature(self, values=None):
        if not values: values = ["Hi", "Hello", "Hola", "Salute"]
        self.__add("categorical_text_feature", ["Categorical"], values)
        return self

    def add_numerical_target(self, values=None):
        if not values: values = [1, 2, 3, 4]
        self.__add("numerical_target", ["Numerical", "Target"], values)
        return self

    def add_categorical_target(self, values=None):
        if not values: values = [1, 2, 2, 4]
        self.__add("categorical_target", ["Categorical", "Target"], values)
        return self

    def __add(self, name, caracteristics, values):
        self.__model_dict.update({
            name + str(self.__count): caracteristics
        })
        self.__dataframe_dict.update({
            name + str(self.__count): pd.Series(values)
        })
        self.__count += 1
        self.rebuild()
