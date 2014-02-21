import numpy as np
from src.type_manipulation.translation.data_type_translation import translate_data_type


def parse_csv_structured(data_file, config):
    """
    Parses the given data file following the data model of the given configuration.
    @return: numpy.recarray
    """
    names, dtypes = [], []
    for key, value in config["data_model"].items():
        assert key not in names
        data_type = value[0]
        dtypes.append(translate_data_type(data_type))
        names.append(str(key))

    data = np.genfromtxt(data_file, delimiter=config["delimiter"], dtype=dtypes, names=names)
    return data
