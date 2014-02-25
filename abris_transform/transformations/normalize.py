from abris_transform.transformations.base_transformer import BaseTransformer
import numpy as np


class NormalizeTransformer(BaseTransformer):
    def __init__(self, config):
        self.__config = config
        self.__columns_to_normalize = None
        self.__normalize_info = None

    def fit(self, data):
        model = self.__config.get_data_model()
        self.__columns_to_normalize = set(model.find_all_columns()) - set(model.find_text_columns()) \
                                      - set(model.find_boolean_columns())
        self.__normalize_info = {}
        for column_index in self.__columns_to_normalize:
            column = data[:, column_index]
            maximum = np.max(column)
            minimum = np.min(column)
            self.__normalize_info[column_index] = NormalizeInformation(minimum, maximum)

    def transform(self, data):
        for column_index in self.__columns_to_normalize:
            info = self.__normalize_info[column_index]
            data[:, column_index] = (data[:, column_index] - info.minimum) / (info.maximum - info.minimum)
        return data


class NormalizeInformation(object):
    def __init__(self, minimum, maximum):
        self.minimum = minimum
        self.maximum = maximum
