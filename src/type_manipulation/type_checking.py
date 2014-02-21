from src.type_manipulation.translation.data_type_translation import translate_data_type


def is_text_type(data_type):
    return translate_data_type(data_type) == translate_data_type("string")


def is_boolean_type(data_type):
    return translate_data_type(data_type) == translate_data_type("boolean")
