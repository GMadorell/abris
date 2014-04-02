import unittest
from pandas.util.testing import assert_series_equal

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

        assert_series_equal(df.categorical_text_feature3, pd.Series(["trim_me", "please", "ple ase", "please"]))
        assert_series_equal(df.categorical_text_feature4, pd.Series(["also trim me", "please", "please", "please"]))

    def test_missing_values_mean(self):
        dm_wrapper = DataModelWrapper() \
            .add_numerical_feature([1.0, 2.0, 3.0, np.nan]) \
            .add_numerical_feature([2.0, np.nan, 2.0, 5.0]) \
            .add_categorical_target([2.0, np.nan, 2.0, 5.0])

        config_wrapper = ConfigWrapper() \
            .add_option("nan_treatment", enabled=True, method="mean")

        config = adapt_dm_wrapper_to_config(dm_wrapper, config_wrapper=config_wrapper)
        cleaner = Cleaner(config)

        df = cleaner.prepare(dm_wrapper.dataframe)

        assert_series_equal(df.numerical_feature0, pd.Series([1.0, 2.0, 3.0, 2.0]))
        assert_series_equal(df.numerical_feature1, pd.Series([2.0, 3.0, 2.0, 5.0]))
        assert_series_equal(df.categorical_target2, pd.Series([2.0, 2.0, 2.0, 5.0]))

    def test_missing_values_median(self):
        dm_wrapper = DataModelWrapper() \
            .add_numerical_feature([1.0, 2.0, 3.0, np.nan]) \
            .add_categorical_target([2.0, np.nan, 2.0, 5.0])

        config_wrapper = ConfigWrapper() \
            .add_option("nan_treatment", enabled=True, method="median")

        config = adapt_dm_wrapper_to_config(dm_wrapper, config_wrapper=config_wrapper)
        cleaner = Cleaner(config)

        df = cleaner.prepare(dm_wrapper.dataframe)

        assert_series_equal(df.numerical_feature0, pd.Series([1.0, 2.0, 3.0, 2.0]))
        assert_series_equal(df.categorical_target1, pd.Series([2.0, 2.0, 2.0, 5.0]))

    def test_missing_values_mode(self):
        dm_wrapper = DataModelWrapper() \
            .add_numerical_feature([2.0, 2.0, 3.0, np.nan]) \
            .add_categorical_target([2.0, np.nan, 2.0, 5.0])

        config_wrapper = ConfigWrapper() \
            .add_option("nan_treatment", enabled=True, method="mode")

        config = adapt_dm_wrapper_to_config(dm_wrapper, config_wrapper=config_wrapper)
        cleaner = Cleaner(config)

        df = cleaner.prepare(dm_wrapper.dataframe)

        assert_series_equal(df.numerical_feature0, pd.Series([2.0, 2.0, 3.0, 2.0]))
        assert_series_equal(df.categorical_target1, pd.Series([2.0, 2.0, 2.0, 5.0]))

    def test_missing_values_drop_rows(self):
        dm_wrapper = DataModelWrapper() \
            .add_numerical_feature([2.0, 2.0, 3.0, np.nan]) \
            .add_categorical_target([2.0, np.nan, 2.0, 5.0])

        config_wrapper = ConfigWrapper() \
            .add_option("nan_treatment", enabled=True, method="drop_rows")

        config = adapt_dm_wrapper_to_config(dm_wrapper, config_wrapper=config_wrapper)
        cleaner = Cleaner(config)

        df = cleaner.prepare(dm_wrapper.dataframe)

        assert_series_equal(df.numerical_feature0, pd.Series([2.0, 3.0]))
        assert_series_equal(df.categorical_target1, pd.Series([2.0, 2.0]))

    def test_missing_values_disabled(self):
        dm_wrapper = DataModelWrapper() \
            .add_numerical_feature([1, 2, 3, np.nan]) \
            .add_categorical_target([2, np.nan, 2, 5])

        config_wrapper = ConfigWrapper() \
            .add_option("nan_treatment", enabled=False, method="mean")

        config = adapt_dm_wrapper_to_config(dm_wrapper, config_wrapper=config_wrapper)
        cleaner = Cleaner(config)

        df = cleaner.prepare(dm_wrapper.dataframe)

        assert_series_equal(df.numerical_feature0, pd.Series([1, 2, 3, np.nan]))
        assert_series_equal(df.categorical_target1, pd.Series([2, np.nan, 2, 5]))

    def test_nan_treatment_unknown_method_should_raise(self):
        dm_wrapper = DataModelWrapper() \
            .add_numerical_feature([1, 2, 3, np.nan])

        config_wrapper = ConfigWrapper() \
            .add_option("nan_treatment", enabled=True, method="some_unknown_method")

        config = adapt_dm_wrapper_to_config(dm_wrapper, config_wrapper=config_wrapper)
        cleaner = Cleaner(config)

        with self.assertRaises(ValueError):
            cleaner.apply(dm_wrapper.dataframe)

    def test_drop_ignored(self):
        dm_wrapper = DataModelWrapper() \
            .add_ignored_numerical_feature() \
            .add_numerical_feature([1, 2, 3, np.nan]) \
            .add_categorical_target([2, np.nan, 2, 5]) \
            .add_ignored_numerical_feature()

        ignored_name_1 = "ignored_numerical_feature0"
        ignored_name_2 = "ignored_numerical_feature3"

        dm_wrapper.dataframe[ignored_name_1]
        dm_wrapper.dataframe[ignored_name_2]

        config = adapt_dm_wrapper_to_config(dm_wrapper)
        cleaner = Cleaner(config)

        df = cleaner.prepare(dm_wrapper.dataframe)

        # Check that the ignored features are no longer there.
        with self.assertRaises(AttributeError):
            df.dataframe[ignored_name_1]

        with self.assertRaises(AttributeError):
            df.dataframe[ignored_name_2]






