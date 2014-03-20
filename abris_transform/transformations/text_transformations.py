from sklearn.feature_extraction.text import CountVectorizer

import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from sklearn_pandas import DataFrameMapper

from abris_transform.transformations.base_transformer import BaseTransformer
from abris_transform.type_manipulation.translation.data_type_translation import translate_data_type


class TextToNumberStructuredTransformer(BaseTransformer):
    def __init__(self, config, verbose=True):
        """
        @param config: Loaded configuration as an ordered dictionary.
        """
        self.__config = config
        self.__verbose = verbose
        self.__vectorizers = None
        self.__column_indices = None
        self.mapper = None

    def fit(self, data):
        self.__vectorizers = []

        mapping = []

        for feature in self.__config.get_data_model().find_text_features():
            name = feature.get_name()
            mapping.append((name, LabelBinarizer()))

        self.mapper = DataFrameMapper(mapping)
        self.mapper.fit(data)

    def transform(self, data):
        # for i, column_index in enumerate(self.__column_indices):
        #     column_name = data.dtype.names[column_index]
        #     column = data[column_name]
        #
        #     numbers = self.__apply_vectorizer(column, self.__vectorizers[i])
        #
        #     # Create a new description so we can change the old dtype to the new numeric type.
        #     old_type = data.dtype
        #     description = old_type.descr
        #     description[column_index] = (description[column_index][0], translate_data_type("float"))
        #
        #     data[column_name] = numbers
        #     data = data.astype(description)
        data = self.mapper.transform(data)

        return data

    def __construct_vectorizer(self, text_array):
        vectorizer = CountVectorizer()
        vectorizer.fit(text_array)
        return vectorizer

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
