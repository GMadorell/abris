import json
import collections

from abris_transform.configuration.data_model import DataModel
from abris_transform.parsing.parameter_parsing import parse_parameter
from abris_transform.parsing.string_aliases import true_boolean_aliases


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

    def is_option_enabled(self, name):
        option_dic = self.__config.get(name, False)
        if option_dic and option_dic["enabled"] in true_boolean_aliases:
            return True
        else:
            return False

    def get_option_parameter(self, option_name, parameter_name):
        """
        Note that the "enabled" configuration part is not considered a parameter.
        """
        parameters = self.get_option_parameters(option_name)
        return parameters[parameter_name]

    def get_option_parameters(self, name):
        """
        Returns all the parameters of the given option.
        Note that the "enabled" configuration part is not considered a parameter.
        """
        option_dic = self.__config[name]
        params = {}
        for key, value in option_dic.items():
            if key != "enabled":
                params[key] = parse_parameter(value)
        return params
