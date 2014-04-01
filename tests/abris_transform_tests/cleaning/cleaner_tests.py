import unittest

from abris_transform.cleaning.cleaner import Cleaner
from tests.wrappers.data_model_wrapper import DataModelWrapper
from tests.wrappers.wrap_utils import adapt_dm_wrapper_to_config


class CleanerTest(unittest.TestCase):
    def test_trim_before_text(self):
        dm_wrapper = DataModelWrapper()
        dm_wrapper.add_categorical_text_feature(1)
        dm_wrapper.add_categorical_text_feature(2)
        dm_wrapper.add_numerical_feature(1)
        dm_wrapper.add_custom_text_feature(["     trim_me", "         please"], 1)
        dm_wrapper.add_custom_text_feature(["     also trim me", " please"], 2)

        config = adapt_dm_wrapper_to_config(dm_wrapper)

        cleaner = Cleaner(config)

        df = cleaner.prepare(dm_wrapper.dataframe)

        self.assertEqual(df.custom_text_feature1[0], "trim_me")
        self.assertEqual(df.custom_text_feature1[1], "please")
        self.assertEqual(df.custom_text_feature2[0], "also trim me")
        self.assertEqual(df.custom_text_feature2[1], "please")


