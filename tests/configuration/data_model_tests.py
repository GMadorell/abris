from collections import OrderedDict
from unittest import TestCase
from abris_transform.configuration.data_model import DataModel
import pandas as pd


class DataModelTest(TestCase):

    def setUp(self):
        self.model_dict = {}
        self.dataframe_dict = {}
        self.rebuild()

    def test_has_target_without_target(self):
        self.assertFalse(self.data_model.has_target())

    def test_has_target_with_target(self):
        self.add_numerical_target()
        self.assertTrue(self.data_model.has_target())

    def test_find_target_feature_with_one_target(self):
        self.add_numerical_target(number=1)
        feature = self.data_model.find_target_feature()
        self.assertEqual(feature.get_name(), "numerical_target1")

    def test_find_target_feature_without_target_should_raise(self):
        with self.assertRaises(AssertionError):
            self.data_model.find_target_feature()

    def test_find_target_feature_with_two_targets_should_raise(self):
        self.add_numerical_target(number=1)
        self.add_numerical_target(number=2)
        with self.assertRaises(AssertionError):
            self.data_model.find_target_feature()

    def test_find_all_features_empty(self):
        self.assertEqual([], self.data_model.find_all_features())

    def test_find_all_features(self):
        self.add_numerical_feature()
        self.add_categorical_text_feature()
        features = self.data_model.find_all_features()
        names = map(lambda feature: feature.get_name(), features)
        self.assertIn("numerical_feature1", names)
        self.assertIn("categorical_text_feature1", names)

    def test_has_any_text_feature_without_text_features(self):
        self.assertFalse(self.data_model.has_any_text_feature())

    def test_has_any_text_feature_with_text_features(self):
        self.add_categorical_text_feature()
        self.assertFalse(self.data_model.has_any_text_feature())

    def test_find_boolean_features_without_any(self):
        self.assertEqual([], self.data_model.find_boolean_features())

    def test_find_boolean_features(self):
        self.add_boolean_feature()
        self.add_categorical_text_feature()
        self.add_numerical_target()
        self.assert_feature_names(self.data_model.find_boolean_features(), "boolean_feature1")

    def test_find_text_features_without_any(self):
        self.assertEqual([], self.data_model.find_text_features())

    def test_find_text_features(self):
        self.add_boolean_feature()
        self.add_categorical_text_feature()
        self.add_numerical_target()
        self.assert_feature_names(self.data_model.find_text_features(), "categorical_text_feature1")

    def test_find_categorical_features_without_any(self):
        self.assertEqual([], self.data_model.find_categorical_features())

    def test_find_categorical_features(self):
        self.add_boolean_feature()
        self.add_categorical_text_feature()
        self.add_numerical_target()
        self.assert_feature_names(self.data_model.find_categorical_features(), "categorical_text_feature1")

    def test_find_ignored_features_without_any(self):
        self.assertEqual([], self.data_model.find_ignored_features())

    def test_find_ignored_features(self):
        self.add_boolean_feature()
        self.add_ignored_numerical_feature()
        self.add_numerical_target()
        self.assert_feature_names(self.data_model.find_ignored_features(), "ignored_numerical_feature1")

    def assert_feature_names(self, list_features, *names):
        assert len(list_features) == len(names)
        for i, feature in enumerate(list_features):
            self.assertEqual(feature.get_name(), names[i])

    def add_numerical_feature(self, number=1):
        self.model_dict.update({
            "numerical_feature%d" % number: ["Numerical"]
        })
        self.dataframe_dict.update({
            "numerical_feature%d" % number: pd.Series([1, 2])
        })
        self.rebuild()

    def add_boolean_feature(self, number=1):
        self.model_dict.update({
            "boolean_feature%d" % number: ["Boolean"]
        })
        self.dataframe_dict.update({
            "boolean_feature%d" % number: pd.Series([True, False])
        })
        self.rebuild()

    def add_ignored_numerical_feature(self, number=1):
        self.model_dict.update({
            "ignored_numerical_feature%d" % number: ["Numerical", "Ignore"]
        })
        self.dataframe_dict.update({
            "ignored_numerical_feature%d" % number: pd.Series([1, 2])
        })
        self.rebuild()

    def add_categorical_numerical_feature(self, number=1):
        self.model_dict.update({
            "categorical_numerical_feature%d" % number: ["Categorical"]
        })
        self.dataframe_dict.update({
            "categorical_numerical_feature%d" % number: pd.Series([1, 2])
        })
        self.rebuild()

    def add_categorical_text_feature(self, number=1):
        self.model_dict.update({
            "categorical_text_feature%d" % number: ["Categorical"]
        })
        self.dataframe_dict.update({
            "categorical_text_feature%d" % number: pd.Series(["Hi", "Hello"])
        })
        self.rebuild()

    def add_numerical_target(self, number=1):
        self.model_dict.update({
            "numerical_target%d" % number: ["Numerical", "Target"]
        })
        self.dataframe_dict.update({
            "numerical_target%d" % number: pd.Series([1, 2])
        })
        self.rebuild()

    def add_categorical_target(self, number=1):
        self.model_dict.update({
            "categorical_target%d" % number: ["Categorical", "Target"]
        })
        self.dataframe_dict.update({
            "categorical_target%d" % number: pd.Series([1, 2])
        })
        self.rebuild()

    def rebuild(self):
        self.dataframe = pd.DataFrame(self.dataframe_dict)
        self.data_model = DataModel(self.model_dict)
        self.data_model.set_features_types_from_dataframe(self.dataframe)
