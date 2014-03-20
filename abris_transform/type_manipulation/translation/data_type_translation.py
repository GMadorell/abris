import numpy as np

__type_name_to_data_type_map = {
    "float": np.float_,
    "integer": np.float_,
    "string": "S10",
    "text": "S10",
    "boolean": bool,
    "bool": bool
}


def type_name_to_data_type(type_name):
    return __type_name_to_data_type_map[type_name.lower()]


__data_type_to_type_name_map = {
    np.dtype(object): "string",
    np.dtype(np.float64): "float",
    np.dtype(bool): "boolean",
    np.dtype(np.int64): "integer"
}


def data_type_to_type_name(data_type):
    return __data_type_to_type_name_map[data_type]

