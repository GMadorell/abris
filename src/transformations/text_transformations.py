from sklearn.feature_extraction.text import CountVectorizer
from src.translation.data_type_translation import translate_data_type


def transform_text_to_numbers_structured(np_array, config):
    """
    @param np_array: STRUCTURED Numpy array to be transformed.
    @param config: Loaded configuration as an ordered dictionary.
    @return: Numpy array with all the columns specified as text parsed to numbers.
    """
    column_indices = __find_text_column_indices(config)
    for column_index in column_indices:
        column_name = np_array.dtype.names[column_index]
        column = np_array[column_name]

        numbers = __transform_text_to_numbers(column)

        # Create a new description so we can change the old dtype to the new numeric type.
        old_type = np_array.dtype
        description = old_type.descr
        description[column_index] = (description[column_index][0], translate_data_type("float"))

        np_array[column_name] = numbers
        np_array = np_array.astype(description)

    return np_array


def __find_text_column_indices(config):
    column_indices = []
    for i, (key, value) in enumerate(config["data_model"].items()):
        data_type = value[0]
        if __is_text_type(data_type):
            column_indices.append(i)
    return column_indices


def __is_text_type(data_type):
    return translate_data_type(data_type) == translate_data_type("string")


def __transform_text_to_numbers(text_array):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(text_array).toarray()
    return X.argmax(1)
