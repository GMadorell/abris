from sklearn_pandas import DataFrameMapper

from abris_transform.parsing.csv_parsing import prepare_csv_to_dataframe
from abris_transform.transformations.mapping import DataFrameMapCreator
from abris_transform.configuration.configuration import Configuration
from abris_transform.type_manipulation.translation.data_type_translation import type_name_to_data_type


class Abris(object):
    """
    Main entry class for the whole preprocessing engine (and probably the only one that needs to be used
    if no more features are needed).
    """

    def __init__(self, config_file):
        self.__config = Configuration(config_file)
        self.__mapper = None

    def prepare(self, data_file):
        """
        Called with the training data.
        """
        data = prepare_csv_to_dataframe(data_file, self.__config)

        model = self.__config.get_data_model()
        target = None
        if model.has_target():
            name = model.find_target_feature().get_name()
            target = data[name].values.astype(type_name_to_data_type("float"))

        mapping = DataFrameMapCreator().get_mapping_from_config(self.__config)
        self.__mapper = DataFrameMapper(mapping)

        data = self.__mapper.fit_transform(data)

        if model.has_target():
            return data, target
        else:
            return data

    def apply(self, data_file):
        """
        Called with the predict data (new information).
        """
        data = prepare_csv_to_dataframe(data_file, self.__config, use_target=False)
        data = self.__mapper.transform(data)
        return data





