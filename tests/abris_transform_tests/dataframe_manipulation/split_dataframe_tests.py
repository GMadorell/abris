import random
from unittest import TestCase

import pandas as pd
import numpy as np
from abris_transform.dataframe_manipulation import dataframe_split


class SplitDataframeTests(TestCase):
    def setUp(self):
        self.dataframe = pd.DataFrame(data={"test_feature": range(100)})

    def test_split_dataframe_non_zero_params(self):
        self.__apply_test(0.6, 0.3, 0.1)

    def test_split_dataframe_zero_training(self):
        self.__apply_test(0, 0.5, 0.5)

    def test_split_dataframe_zero_cv(self):
        self.__apply_test(0.3, 0, 0.7)

    def test_split_dataframe_zero_test(self):
        self.__apply_test(0.6, 0.4, 0)

    def test_two_decimals(self):
        self.__apply_test(0.58, 0.23, 0.19)

    def test_fail_if_params_sum_more_than_1(self):
        with self.assertRaises(AssertionError):
            self.__apply_test(1.1, 0, 0)

    def test_randomize_some_runs(self):
        for i in xrange(25):
            self.__do_random_test()

    def __apply_test(self, train_percentage, cv_percentage, test_percentage):
        train, cv, test = dataframe_split.split_dataframe(self.dataframe, train_percentage, cv_percentage, test_percentage)

        self.assertAlmostEqual(train_percentage * 100, len(train))
        self.assertAlmostEqual(cv_percentage * 100, len(cv))
        self.assertAlmostEqual(test_percentage * 100, len(test))

        merged = train["test_feature"].values.tolist() \
                 + cv["test_feature"].values.tolist() \
                 + test["test_feature"].values.tolist()
        assert len(set(merged)) == 100

    def __do_random_test(self):
        train_percentage = self.__get_random_percentage()

        cv_percentage = self.__get_random_percentage()
        while train_percentage + cv_percentage > 1:
            cv_percentage = self.__get_random_percentage()

        test_percentage = 1 - train_percentage - cv_percentage

        self.__apply_test(train_percentage, cv_percentage, test_percentage)

    def __get_random_percentage(self):
        return round(random.uniform(0, 1), 2)


class SplitDataframeTrainTests(TestCase):
    def setUp(self):
        self.dataframe = pd.DataFrame(data={"test_feature": range(100)})

    def test_zero_training(self):
        self.__apply_test(0)

    def test_full_training(self):
        self.__apply_test(1)

    def test_some_middle_values(self):
        for i in np.arange(0, 1, 0.15):
            self.__apply_test(i)

    def test_fail_with_negative_train_percentage(self):
        with self.assertRaises(AssertionError):
            self.__apply_test(-0.1)

    def test_fail_when_train_percentage_higher_than_one(self):
        with self.assertRaises(AssertionError):
            self.__apply_test(1.01)

    def __apply_test(self, train_percentage):
        train, test = dataframe_split.split_dataframe_train_test(self.dataframe, train_percentage)

        self.assertAlmostEqual(train_percentage * 100, len(train))
        self.assertAlmostEqual((1 - train_percentage) * 100, len(test))

        merged = train["test_feature"].values.tolist() \
                 + test["test_feature"].values.tolist()
        assert len(set(merged)) == 100

