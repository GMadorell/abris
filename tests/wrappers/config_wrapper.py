from collections import OrderedDict
from StringIO import StringIO
import json
from abris_transform.configuration.configuration import Configuration


class ConfigWrapper(object):
    def __init__(self, dm_wrapper=None):
        """
        :param dm_wrapper: DataModelWrapper
        """
        self.__config_dict = OrderedDict()

        if dm_wrapper is not None:
            self.add_data_model_wrapper(dm_wrapper)

    def add_data_model_wrapper(self, dm_wrapper):

        data_model_dict = self.__config_dict.get("data_model", OrderedDict())

        for feature in dm_wrapper.data_model:
            name = feature.get_name()
            characteristics = feature.get_characteristics()
            data_model_dict[name] = characteristics

        self.__config_dict["data_model"] = data_model_dict

    def add_option(self, name, enabled, **kargs):
        """
        Adds the given option with the given enabled status and the given
        keywords to the configuration wrapper.
        Also adds all the given keywords
        """
        option_dict = OrderedDict()
        option_dict["enabled"] = bool(enabled)

        for key, value in kargs.iteritems():
            option_dict[str(key)] = value

        self.__config_dict[name] = option_dict

    def build_to_config(self):
        return Configuration(self.build_to_file_like())

    def build_to_file_like(self):
        return StringIO(self.build_to_string())

    def build_to_string(self):
        return json.dumps(self.__config_dict)

