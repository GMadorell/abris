from sklearn.preprocessing import StandardScaler, LabelBinarizer, Binarizer
from sklearn_pandas import DataFrameMapper

from abris_transform.transformations.as_numpy_array_transformer import AsNumpyArrayTransformer


def get_dummy_variables_mapping(config):
    mapping = []

    data_model = config.get_data_model()
    features = set(data_model.find_categorical_features())
    features -= set(data_model.find_ignored_features())
    if data_model.has_target():
        features -= {data_model.find_target_feature()}

    for feature in features:
        name = feature.get_name()
        mapping.append((name, LabelBinarizer()))
    return mapping


def get_normalize_variables_mapping(config):
    model = config.get_data_model()
    mapping = []
    features_to_normalize = set(model.find_all_features()) \
                            - set(model.find_boolean_features()) \
                            - set(model.find_categorical_features()) \
                            - set(model.find_ignored_features())
    if model.has_target():
        features_to_normalize -= {model.find_target_feature()}

    if config.is_option_enabled("scaling"):
        transform_class = StandardScaler
    else:
        transform_class = AsNumpyArrayTransformer

    for feature in features_to_normalize:
        name = feature.get_name()
        mapping.append((name, transform_class()))
    return mapping


def get_boolean_features_mapping(config):
    mapping = []
    model = config.get_data_model()

    features = set(model.find_boolean_features())
    features -= set(model.find_ignored_features())

    for feature in features:
        mapping.append((feature.get_name(), Binarizer()))
    return mapping


class DataFrameMapCreator(object):
    def __init__(self):
        pass

    def get_mapper_from_config(self, config):
        return DataFrameMapper(self.get_mapping_from_config(config))

    def get_mapping_from_config(self, config):
        mapping = []
        mapping += get_dummy_variables_mapping(config)
        mapping += get_normalize_variables_mapping(config)
        mapping += get_boolean_features_mapping(config)

        return mapping

