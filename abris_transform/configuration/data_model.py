from abris_transform.type_manipulation.translation.data_type_translation import translate_data_type


class DataModel(object):
    def __init__(self, data_model_dictionary):
        self.__model = []
        for key, value in data_model_dictionary.items():
            self.__model.append(Feature(key, value))

    def has_any_text_feature(self):
        return self.find_text_columns_indices() is not None

    def has_target(self):
        return len(self.__find_columns_indices_matching(lambda feat: feat.is_target())) > 0

    def find_all_columns(self):
        return range(sum(1 for _ in self.__iter__()))

    def find_all_features(self):
        return [feature for feature in self.__iter__()]

    def find_boolean_columns_indices(self):
        return self.__find_columns_indices_matching(lambda feature: feature.is_type("boolean"))

    def find_boolean_features(self):
        return self.__find_features_matching(lambda feature: feature.is_type("boolean"))

    def find_text_columns_indices(self):
        return self.__find_columns_indices_matching(lambda feature: feature.is_type("string"))

    def find_text_features(self):
        return self.__find_features_matching(lambda feature: feature.is_type("string"))

    def find_categorical_columns_indices(self):
        return self.__find_columns_indices_matching(lambda feature: feature.is_categorical())

    def find_categorical_features(self):
        return self.__find_features_matching(lambda feature: feature.is_categorical())

    def find_target_column_index(self):
        columns = self.__find_columns_indices_matching(lambda feature: feature.is_target())
        assert len(columns) < 2, "Can't have two targets!"
        return columns[0]

    def find_target_feature(self):
        features = self.__find_features_matching(lambda feature: feature.is_target())
        assert len(features) < 2, "Can't have two targets!"
        return features[0]

    def __find_columns_indices_matching(self, match_function):
        """
        Returns a list of column indices that match the given function.
        @match_function: Called with a single argument, a feature.
        """
        column_indices = []
        for i, feature in enumerate(self.__iter__()):
            if match_function(feature):
                column_indices.append(i)
        return column_indices

    def __find_features_matching(self, match_function):
        features = []
        for feature in self.__iter__():
            if match_function(feature):
                features.append(feature)
        return features

    def __iter__(self):
        for feature in self.__model:
            yield feature


class Feature(object):
    def __init__(self, name, characteristics_list):
        self.__name = str(name)
        self.__characteristics = characteristics_list

    def get_name(self):
        return self.__name

    def get_type(self):
        return self.__characteristics[0]

    def is_type(self, type_):
        return translate_data_type(self.get_type()) == translate_data_type(type_)

    def is_categorical(self):
        return self.has_characteristic("categorical")

    def is_target(self):
        return self.has_characteristic("target")

    def has_characteristic(self, characteristic):
        return characteristic.lower() in map(lambda string: string.lower(), self.__characteristics)




