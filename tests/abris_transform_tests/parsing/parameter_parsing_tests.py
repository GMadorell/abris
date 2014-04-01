import unittest
from abris_transform.parsing.parameter_parsing import parse_parameter


class ParseParameterTest(unittest.TestCase):

    def test_integer(self):
        integer = parse_parameter("1")
        self.assertIsInstance(integer, int)

    def test_float(self):
        float_numbers = [
            parse_parameter("1.001"),
            parse_parameter("1."),
            parse_parameter(".001")]
        for float_number in float_numbers:
            self.assertIsInstance(float_number, float)

    def test_true_boolean(self):
        true_booleans = [
            parse_parameter("True"),
            parse_parameter("true"),
            parse_parameter(True)
        ]
        for boolean in true_booleans:
            self.assertIsInstance(boolean, bool)
            self.assertTrue(boolean)

    def test_false_boolean(self):
        false_booleans = [
            parse_parameter("False"),
            parse_parameter("false"),
            parse_parameter(False)
        ]
        for boolean in false_booleans:
            self.assertIsInstance(boolean, bool)
            self.assertFalse(boolean)

    def test_string(self):
        strings = [
            parse_parameter("Some random text"),
            parse_parameter("more_text")
        ]
        for string in strings:
            self.assertIsInstance(string, str)
