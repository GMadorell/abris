

class Cleaner(object):
    def __init__(self, config):
        self.__config = config
        self.__model = config.get_data_model()

    def prepare(self, dataframe):
        dataframe = self.__drop_ignored_features(dataframe)
        return dataframe

    def apply(self, dataframe):
        return self.prepare(dataframe)

    def __drop_ignored_features(self, dataframe):
        names = self.__extract_feature_names(self.__model.find_ignored_features())
        return dataframe.drop(names, axis=1)

    def __extract_feature_names(self, features):
        return map(lambda feature: feature.get_name(), features)

