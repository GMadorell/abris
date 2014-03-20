from sklearn.preprocessing import StandardScaler, LabelBinarizer

from abris_transform.transformations.null_transformer import NullTransformer


def get_dummy_variables_mapping(config):
    mapping = []
    for feature in config.get_data_model().find_categorical_features():
        name = feature.get_name()
        mapping.append((name, LabelBinarizer()))
    return mapping


def get_normalize_variables_mapping(config):
    model = config.get_data_model()
    mapping = []
    features_to_normalize = set(model.find_all_features()) - set(model.find_text_features()) \
                            - set(model.find_boolean_features()) - set(model.find_categorical_features())
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


class DataFrameMapCreator(object):
    def __init__(self):
        pass

    def get_mapping_from_config(self, config):
        mapping = []
        mapping += get_dummy_variables_mapping(config)
        mapping += get_normalize_variables_mapping(config)
        mapping += get_boolean_features_mapping(config)

        return mapping

