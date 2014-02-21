from sklearn.feature_extraction.text import CountVectorizer
from src.type_manipulation.translation.data_type_translation import translate_data_type
from src.type_manipulation.type_checking import is_text_type


class TextToNumberStructuredTransformer(object):
    def __init__(self, config, verbose=True):
        """
        @param config: Loaded configuration as an ordered dictionary.
        """
        self.__config = config
        self.__verbose = verbose
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
            if is_text_type(data_type):
                column_indices.append(i)
        return column_indices

    def __transform_text_to_numbers(self, text_array):
        vectorizer = CountVectorizer()
        x = vectorizer.fit_transform(text_array).toarray()
        return x.argmax(1), vectorizer

    def __apply_vectorizer(self, column, vectorizer):
        if self.__verbose:
            self.__check_column_words_inside_vocabulary(column, vectorizer)
        x = vectorizer.transform(column).toarray()
        return x.argmax(1)

    def __check_column_words_inside_vocabulary(self, column, vectorizer):
        word_set = set()
        for word in column:
            word_set.add(word)
        for word in word_set:
            if word.lower() not in vectorizer.get_feature_names():
                print "Warning: %s not found in the vectorizer. Assigning a value of zero to it.\n" \
                      "Vectorizer features:%s" \
                      % (word, str(vectorizer.get_feature_names()))
