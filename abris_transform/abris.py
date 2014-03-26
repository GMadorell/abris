from abris_transform.parsing.csv_parsing import prepare_csv_to_dataframe
from abris_transform.configuration.configuration import Configuration
from abris_transform.transformations.cleaner import Cleaner
from abris_transform.transformations.transformer import Transformer
import pandas as pd


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
        self.__transformer = Transformer(self.__config)
        self.__cleaner = Cleaner(self.__config)

    def prepare(self, data_source):
        """
        Called with the training data.
        @param data_source: Either a pandas.DataFrame or a file-like object.
        """
        dataframe = self.__get_dataframe(data_source, use_target=True)
        self.__config.get_data_model().set_features_types_from_dataframe(dataframe)
        dataframe = self.__cleaner.prepare(dataframe)
        return self.__transformer.prepare(dataframe)

    def apply(self, data_source):
        """
        Called with the predict data (new information).
        @param data_source: Either a pandas.DataFrame or a file-like object.
        """
        dataframe = self.__get_dataframe(data_source, use_target=False)
        dataframe = self.__cleaner.apply(dataframe)
        dataframe = self.__transformer.apply(dataframe)
        return dataframe

    def __get_dataframe(self, input, use_target):
        if isinstance(input, pd.DataFrame):
            return input
        return prepare_csv_to_dataframe(input, self.__config, use_target)




