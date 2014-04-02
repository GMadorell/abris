import unittest

from abris_transform.parsing import boolean_aliases


def parse_parameter(value):
    """
    @return: The best approximation of a type of the given value.
    """
    if any((isinstance(value, float), isinstance(value, int), isinstance(value, bool))):
        return value

    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            if value in boolean_aliases.true_boolean_aliases:
                return True
            elif value in boolean_aliases.false_boolean_aliases:
                return False
            else:
                return str(value)
