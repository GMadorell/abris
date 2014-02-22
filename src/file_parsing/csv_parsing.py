import numpy as np
from src.type_manipulation.translation.data_type_translation import translate_data_type


def parse_csv_structured(data_file, config):
    """
    Parses the given data file following the data model of the given configuration.
    @return: numpy.recarray
    """
    names, dtypes = [], []
    for feature in config.get_data_model():
        assert feature.get_name() not in names
        data_type = feature.get_type()
        dtypes.append(translate_data_type(data_type))
        names.append(feature.get_name())

    data = np.genfromtxt(data_file, delimiter=config.get_delimiter(), dtype=dtypes, names=names)
    return data
