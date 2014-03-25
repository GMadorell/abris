

class Cleaner(object):
    def __init__(self, config):
        self.__config = config
        self.__model = config.get_data_model()

    def prepare(self, dataframe):
        dataframe = self.__drop_ignored_features(dataframe)
        dataframe = self.__apply_nan_treatment(dataframe)
        return dataframe

    def apply(self, dataframe):
        return self.prepare(dataframe)

    def __drop_ignored_features(self, dataframe):
        names = self.__extract_feature_names(self.__model.find_ignored_features())
        return dataframe.drop(names, axis=1)

    def __extract_feature_names(self, features):
        return map(lambda feature: feature.get_name(), features)

    def __apply_nan_treatment(self, dataframe):
        if self.__config.is_option_enabled("nan_treatment"):
            method = self.__config.get_option_parameter("nan_treatment", "method")
            if method == "mean":
                dataframe = dataframe.fillna(dataframe.mean())
            elif method == "drop_rows":
                dataframe = dataframe.dropna(axis=0)
                dataframe = dataframe.reset_index(drop=True)
            else:
                raise ValueError("Method not understood: %s" % method)
        return dataframe


