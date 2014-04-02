import unittest

from abris_transform.cleaning.cleaner import Cleaner
from tests.wrappers.config_wrapper import ConfigWrapper
from tests.wrappers.data_model_wrapper import DataModelWrapper
from tests.wrappers.wrap_utils import adapt_dm_wrapper_to_config

import numpy as np
import pandas as pd


class CleanerTest(unittest.TestCase):
    def test_trim_before_text(self):
        dm_wrapper = DataModelWrapper() \
            .add_categorical_text_feature() \
            .add_categorical_text_feature() \
            .add_numerical_feature() \
            .add_categorical_text_feature(["      trim_me", " please", " ple ase", "   please"]) \
            .add_categorical_text_feature(["also trim me              ", "please ", "please   ", "please  "])

        config = adapt_dm_wrapper_to_config(dm_wrapper)
        cleaner = Cleaner(config)

        df = cleaner.prepare(dm_wrapper.dataframe)

        assert np.all(df.categorical_text_feature3 == pd.Series(["trim_me", "please", "ple ase", "please"]))
        assert np.all(df.categorical_text_feature4 == pd.Series(["also trim me", "please", "please", "please"]))

    def test_trim_after_text(self):
        dm_wrapper = DataModelWrapper() \
            .add_categorical_text_feature() \
            .add_categorical_text_feature() \
            .add_numerical_feature() \
            .add_categorical_text_feature(["trim_me     ", "please", "ple ase", "please   "]) \
            .add_categorical_text_feature(["also trim me              ", "please ", "please ", "please  "])

        config = adapt_dm_wrapper_to_config(dm_wrapper)
        cleaner = Cleaner(config)

        df = cleaner.prepare(dm_wrapper.dataframe)

        assert np.all(df.categorical_text_feature3 == pd.Series(["trim_me", "please", "ple ase", "please"]))
        assert np.all(df.categorical_text_feature4 == pd.Series(["also trim me", "please", "please", "please"]))

    def test_missing_values_mean_enabled(self):
        dm_wrapper = DataModelWrapper() \
            .add_numerical_feature([1, 2, 3, np.nan]) \
            .add_categorical_target([2, np.nan, 2, 5])

        config_wrapper = ConfigWrapper() \
            .add_option("nan_treatment", enabled=True, method="mean")

        config = adapt_dm_wrapper_to_config(dm_wrapper, config_wrapper=config_wrapper)
        cleaner = Cleaner(config)

        df = cleaner.prepare(dm_wrapper.dataframe)

        print df.numerical_feature0

        assert np.all(df.numerical_feature0 == pd.Series([1, 2, 3, 2]))
        assert np.all(df.categorical_target1 == pd.Series([2, 3, 2, 5]))




