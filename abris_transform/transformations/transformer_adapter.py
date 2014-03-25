

class TransformerAdapter(object):
    """
    Abstraction over any transformer.
    It exposes a single data processing method, which will output the result of
    fit_transforming the data the first time it's called.
    All consecutive calls will simply transform and then return the data.
    """
    def __init__(self, transformer):
        self.__transformer = transformer
        self.__is_fitted = False

    def transform(self, data):
        if self.__is_fitted:
            return self.__transformer.transform(data)
        else:
            self.__is_fitted = True
            return self.__transformer.fit_transform(data)

    def is_fitted(self):
        return self.__is_fitted
