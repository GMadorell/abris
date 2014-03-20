from abris_transform import string_aliases
from abris_transform.transformations.base_transformer import BaseTransformer
from abris_transform.type_manipulation.translation.data_type_translation import translate_data_type


class BooleanToNumberTransformer(BaseTransformer):
    def __init__(self, config):
        self.__config = config
        self.__boolean_columns = None

    def fit(self, data):
        self.__boolean_columns = self.__config.get_data_model().find_boolean_columns_indices()

    def transform(self, data):
        for column_index in self.__boolean_columns:
            column_name = data.dtype.names[column_index]
            column = data[column_name]

            numbers = column.astype(translate_data_type("float"))

            # Create a new description so we can change the old dtype to the new numeric type.
            old_type = data.dtype
            description = old_type.descr
            description[column_index] = (description[column_index][0], translate_data_type("float"))

            data[column_name] = numbers
            data = data.astype(description)

        return data


class PdBooleanTransformer(BaseTransformer):
    def __init__(self, config):
        self.__config = config
        self.__map = None

    def fit(self, data):
        self.__map = {}
        for true_alias in string_aliases.true_boolean_aliases:
            self.__map[true_alias] = 1
        for false_alias in string_aliases.false_boolean_aliases:
            self.__map[false_alias] = 0

    def transform(self, data):
        model = self.__config.get_data_model()
        for feature in model.find_boolean_features():
            name = feature.get_name()
            data = data[name].map(self.__map)
        return data