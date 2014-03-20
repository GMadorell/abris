from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn_pandas import DataFrameMapper

from abris_transform.type_manipulation.translation.data_type_translation import translate_data_type
from abris_transform.configuration.configuration import Configuration
from abris_transform.parsing.csv_parsing import apply_csv_structured, prepare_csv_structured
from abris_transform.transformations.null_transformer import NullTransformer
from abris_transform.transformations.transform_pipeline import TransformPipeline


def get_dummy_variables_mapping(config):
    mapping = []
    for feature in config.get_data_model().find_text_features():
        name = feature.get_name()
        mapping.append((name, LabelBinarizer()))
    return mapping


def get_normalize_variables_mapping(config):
    model = config.get_data_model()
    mapping = []
    features_to_normalize = set(model.find_all_features()) - set(model.find_text_features()) \
                           - set(model.find_boolean_features())
    if model.has_target():
            features_to_normalize -= {model.find_target_feature()}

    if config.is_option_enabled("scaling"):
        transform_class = StandardScaler
    else:
        transform_class = NullTransformer

    for feature in features_to_normalize:
        name = feature.get_name()
        mapping.append((name, transform_class()))
    return mapping


def get_boolean_features_mapping(config):
    model = config.get_data_model()
    mapping = []
    for feature in model.find_boolean_features():
        mapping.append((feature.get_name(), NullTransformer()))
    return mapping


class Abris(object):
    """
    Main entry class for the whole preprocessing engine (and probably the only one that needs to be used
    if no more features are needed).
    """
    def __init__(self, config_file):
        self.__config = Configuration(config_file)
        self.__pipeline = None
        self.__mapper = None

    def prepare(self, data_file):
        """
        Called with the training data.
        """
        self.__pipeline = TransformPipeline().build_from_config(self.__config)

        data = prepare_csv_structured(data_file, self.__config)

        model = self.__config.get_data_model()
        if model.has_target():
            name = model.find_target_feature().get_name()
            target = data[name].values.astype(translate_data_type("float"))

        mapping = []
        mapping += get_dummy_variables_mapping(self.__config)
        mapping += get_normalize_variables_mapping(self.__config)
        mapping += get_boolean_features_mapping(self.__config)

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
        data = apply_csv_structured(data_file, self.__config)
        # data = self.__pipeline.apply(data)

        data = self.__mapper.transform(data)

        return data





