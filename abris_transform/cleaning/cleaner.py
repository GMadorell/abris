

class Cleaner(object):
    def __init__(self, config):
        self.__config = config
        self.__model = config.get_data_model()

    def prepare(self, dataframe):
        dataframe = self.__trim_text_features(dataframe)
        dataframe = self.__drop_ignored_features(dataframe)
        dataframe = self.__apply_nan_treatment(dataframe)
        return dataframe

    def apply(self, dataframe):
        return self.prepare(dataframe)

    def __trim_text_features(self, dataframe):
        text_features = self.__model.find_text_features()
        text_feature_names = map(lambda feature: feature.get_name(), text_features)
        for name in text_feature_names:
            dataframe[name] = dataframe[name].str.strip()
        return dataframe

    def __drop_ignored_features(self, dataframe):
        names = self.__extract_feature_names(self.__model.find_ignored_features())
        return dataframe.drop(names, axis=1)

    def __extract_feature_names(self, features):
        return map(lambda feature: feature.get_name(), features)

    def __apply_nan_treatment(self, dataframe):
        if self.__config.is_option_enabled("nan_treatment"):
            method = self.__config.get_option_parameter("nan_treatment", "method")
            numerical_features_names = self.__extract_feature_names(
                self.__config.get_data_model().find_numerical_features())
            if method == "mean":
                dataframe[numerical_features_names] = dataframe[numerical_features_names].fillna(dataframe.mean())
                dataframe = dataframe.fillna(dataframe.mode())
            elif method == "median":
                dataframe[numerical_features_names] = dataframe[numerical_features_names].fillna(dataframe.mean())
                dataframe = dataframe.fillna(dataframe.mode())
            elif method == "mode":
                dataframe = dataframe.fillna(dataframe.mode())
            elif method == "drop_rows":
                dataframe = self.__drop_rows(dataframe)
            else:
                raise ValueError("Method not understood: %s" % method)
        return dataframe

    def __drop_rows(self, dataframe):
        dataframe = dataframe.dropna(axis=0)
        dataframe = dataframe.reset_index(drop=True)
        return dataframe


