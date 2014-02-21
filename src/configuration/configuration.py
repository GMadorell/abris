import json
import collections
from src.configuration.data_model import DataModel


class Configuration(object):

    def __init__(self, config_file=None):
        self.__config = None
        self.__data_model = None

        if config_file:
            self.load_from_file(config_file)

    def load_from_file(self, config_file):
        self.__config = json.load(config_file, object_pairs_hook=collections.OrderedDict)
        self.__data_model = DataModel(self.__config["data_model"])

    def get_data_model(self):
        return self.__data_model

    def get_delimiter(self):
        return self.__config["delimiter"]
