from abris_transform.configuration.feature import Feature
from abris_transform.decorators.run_once import method_once
from abris_transform.type_manipulation.translation.data_type_translation import data_type_to_type_name


class DataModel(object):
    def __init__(self, data_model_dictionary):
        self.__model = []
        for key, value in data_model_dictionary.items():
            self.__model.append(Feature(key, value))

    @method_once
    def set_features_types_from_dataframe(self, data_frame):
        """
        Sets the features types from the given data_frame.
        All the calls except the first one are ignored.
        """
        dtypes = data_frame.dtypes
        for feature in self.__iter__():
            name = feature.get_name()
            type_name = data_type_to_type_name(dtypes[name])
            feature.set_type_name(type_name)

    def has_any_text_feature(self):
        return self.find_text_features() is True

    def has_target(self):
        return len(self.__find_features_matching(lambda feat: feat.is_target())) > 0

    def find_all_features(self):
        return [feature for feature in self.__iter__()]

    def find_numerical_features(self, include_target=False):
        """
        More formally, returns a set of those features that aren't boolean,
        categorical nor ignored (optionally [defaults to ignoring it]
        including the target feature).
        """
        numerical_features = set(self.find_all_features()) \
                             - set(self.find_boolean_features()) \
                             - set(self.find_categorical_features()) \
                             - set(self.find_ignored_features())
        if not include_target and self.has_target():
            numerical_features -= {self.find_target_feature()}
        return numerical_features

    def find_boolean_features(self):
        return self.__find_features_matching(lambda feature: feature.is_type_name("boolean"))

    def find_text_features(self):
        return self.__find_features_matching(lambda feature: feature.is_type_name("string"))

    def find_categorical_features(self):
        return self.__find_features_matching(lambda feature: feature.is_categorical())

    def find_ignored_features(self):
        return self.__find_features_matching(lambda feature: feature.has_characteristic("ignore"))

    def find_target_feature(self):
        features = self.__find_features_matching(lambda feature: feature.is_target())
        assert len(features) < 2, "Can't have two targets!"
        assert len(features) > 0, "This data model doesn't have any target feature"
        return features[0]

    def __find_features_matching(self, match_function):
        features = []
        for feature in self.__iter__():
            if match_function(feature):
                features.append(feature)
        return features

    def __iter__(self):
        for feature in self.__model:
            yield feature




