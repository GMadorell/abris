from abris_transform.type_manipulation.translation.data_type_translation import type_name_to_data_type, \
    data_type_to_type_name


class DataModel(object):
    def __init__(self, data_model_dictionary):
        self.__model = []
        for key, value in data_model_dictionary.items():
            self.__model.append(Feature(key, value))
        self.__feature_types_set = False

    def set_features_types_from_dataframe(self, data_frame):
        """
        Sets the features types from the given data_frame.
        All the calls except the first one are ignored.
        """
        if self.__feature_types_set:
            return
        self.__feature_types_set = True

        dtypes = data_frame.dtypes
        for feature in self.__iter__():
            name = feature.get_name()
            type_name = data_type_to_type_name(dtypes[name])
            feature.set_type_name(type_name)

    def has_any_text_feature(self):
        return self.find_text_features() is not None

    def has_target(self):
        return len(self.__find_features_matching(lambda feat: feat.is_target())) > 0

    def find_all_features(self):
        return [feature for feature in self.__iter__()]

    def find_boolean_features(self):
        return self.__find_features_matching(lambda feature: feature.is_type_name("boolean"))

    def find_text_features(self):
        return self.__find_features_matching(lambda feature: feature.is_type_name("string"))

    def find_categorical_features(self):
        return self.__find_features_matching(lambda feature: feature.is_categorical())

    def find_target_feature(self):
        features = self.__find_features_matching(lambda feature: feature.is_target())
        assert len(features) < 2, "Can't have two targets!"
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


class Feature(object):
    def __init__(self, name, characteristics_list):
        self.__name = str(name)
        self.__characteristics = characteristics_list
        self.__type_name = None

    def get_name(self):
        return self.__name

    def get_type_name(self):
        assert self.__type_name is not None
        return self.__type_name

    def set_type_name(self, type_name):
        self.__type_name = type_name

    def is_type_name(self, type_name):
        return type_name_to_data_type(self.get_type_name()) == type_name_to_data_type(type_name)

    def is_categorical(self):
        return self.has_characteristic("categorical")

    def is_target(self):
        return self.has_characteristic("target")

    def has_characteristic(self, characteristic):
        return characteristic.lower() in map(lambda string: string.lower(), self.__characteristics)




