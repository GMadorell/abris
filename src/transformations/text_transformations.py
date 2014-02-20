from sklearn.feature_extraction.text import CountVectorizer
from src.translation.data_type_translation import translate_data_type


class TextToNumberStructuredTransformer(object):
    def __init__(self, config):
        """
        @param config: Loaded configuration as an ordered dictionary.
        """
        self.__config = config
        self.__vectorizers = None
        self.__column_indices = None

    def fit_transform(self, data):
        """
        @param data: STRUCTURED Numpy array to be transformed.
        @return: Numpy array with all the columns specified as text parsed to numbers.
        """
        self.__vectorizers = []
        self.__column_indices = self.__find_text_column_indices(self.__config)
        for column_index in self.__column_indices:
            column_name = data.dtype.names[column_index]
            column = data[column_name]

            numbers, vectorizer = self.__transform_text_to_numbers(column)
            self.__vectorizers.append(vectorizer)

            # Create a new description so we can change the old dtype to the new numeric type.
            old_type = data.dtype
            description = old_type.descr
            description[column_index] = (description[column_index][0], translate_data_type("float"))

            data[column_name] = numbers
            data = data.astype(description)

        return data

    def transform(self, data):
        for i, column_index in enumerate(self.__column_indices):
            column_name = data.dtype.names[column_index]
            column = data[column_name]

            numbers = self.__apply_vectorizer(column, self.__vectorizers[i])

            # Create a new description so we can change the old dtype to the new numeric type.
            old_type = data.dtype
            description = old_type.descr
            description[column_index] = (description[column_index][0], translate_data_type("float"))

            data[column_name] = numbers
            data = data.astype(description)

        return data

    def __find_text_column_indices(self, config):
        column_indices = []
        for i, (key, value) in enumerate(config["data_model"].items()):
            data_type = value[0]
            if self.__is_text_type(data_type):
                column_indices.append(i)
        return column_indices

    def __is_text_type(self, data_type):
        return translate_data_type(data_type) == translate_data_type("string")

    def __transform_text_to_numbers(self, text_array):
        vectorizer = CountVectorizer()
        x = vectorizer.fit_transform(text_array).toarray()
        return x.argmax(1), vectorizer

    def __apply_vectorizer(self, column, vectorizer):
        x = vectorizer.transform(column).toarray()
        return x.argmax(1)
