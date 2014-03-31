from textwrap import dedent
from unittest import TestCase
from StringIO import StringIO
from abris_transform.configuration.configuration import Configuration
from mock import patch


class ConfigurationTests(TestCase):
    sample_config = dedent(
        """
        {
            "data_model": {
                "feature1": ["Categorical"],
                "feature2": ["Boolean"]
            },
            "some_option": {
                "enabled": "True",
                "some_parameter": 123.123
            },
            "some_disabled_option": {
                "enabled": "False",
                "some_parameter": 123.123
            }
        }
        """)

    def setUp(self):
        self.config = Configuration()
        self.config.load_from_file(StringIO(self.sample_config))

    def test_load_from_file_constructor(self):
        with patch.object(Configuration, "load_from_file") as mock_method:
            instantiation = Configuration()
        assert mock_method.call_count == 0

        with patch.object(Configuration, "load_from_file") as mock_method:
            instantiation = Configuration(config_file=StringIO(self.sample_config))
        assert mock_method.call_count == 1

    def test_is_option_enabled(self):
        self.assertTrue(self.config.is_option_enabled("some_option"))
        self.assertFalse(self.config.is_option_enabled("some_disabled_option"))

    def test_is_unexisting_option_enabled_should_return_false(self):
        self.assertTrue(self.config.is_option_enabled("unexisting_option"))




