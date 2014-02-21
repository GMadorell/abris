

class DataModel(object):
    def __init__(self, data_model_dictionary):
        self.__model = data_model_dictionary

    def __iter__(self):
        for key, value in self.__model.items():
            yield key, value


