from unittest import TestCase

from tests.wrappers.data_model_wrapper import DataModelWrapper


class DataModelTest(TestCase):
    def setUp(self):
        self.dm_wrapper = DataModelWrapper()

    def test_has_target_without_target(self):
        self.assertFalse(self.dm_wrapper.data_model.has_target())

    def test_has_target_with_target(self):
        self.dm_wrapper.add_numerical_target()
        self.assertTrue(self.dm_wrapper.data_model.has_target())

    def test_find_target_feature_with_one_target(self):
        self.dm_wrapper.add_numerical_target()
        feature = self.dm_wrapper.data_model.find_target_feature()
        self.assertEqual(feature.get_name(), "numerical_target0")

    def test_find_target_feature_without_target_should_raise(self):
        with self.assertRaises(AssertionError):
            self.dm_wrapper.data_model.find_target_feature()

    def test_find_target_feature_with_two_targets_should_raise(self):
        self.dm_wrapper.add_numerical_target()
        self.dm_wrapper.add_numerical_target()
        with self.assertRaises(AssertionError):
            self.dm_wrapper.data_model.find_target_feature()

    def test_find_all_features_empty(self):
        self.assertEqual([], self.dm_wrapper.data_model.find_all_features())

    def test_find_all_features(self):
        self.dm_wrapper.add_numerical_feature()
        self.dm_wrapper.add_categorical_text_feature()
        features = self.dm_wrapper.data_model.find_all_features()
        names = map(lambda feature: feature.get_name(), features)
        self.assertIn("numerical_feature0", names)
        self.assertIn("categorical_text_feature1", names)

    def test_has_any_text_feature_without_text_features(self):
        self.assertFalse(self.dm_wrapper.data_model.has_any_text_feature())

    def test_has_any_text_feature_with_text_features(self):
        self.dm_wrapper.add_categorical_text_feature()
        self.assertFalse(self.dm_wrapper.data_model.has_any_text_feature())

    def test_find_boolean_features_without_any(self):
        self.assertEqual([], self.dm_wrapper.data_model.find_boolean_features())

    def test_find_boolean_features(self):
        self.dm_wrapper.add_boolean_feature()
        self.dm_wrapper.add_categorical_text_feature()
        self.dm_wrapper.add_numerical_target()
        self.assert_feature_names(self.dm_wrapper.data_model.find_boolean_features(), "boolean_feature0")

    def test_find_text_features_without_any(self):
        self.assertEqual([], self.dm_wrapper.data_model.find_text_features())

    def test_find_text_features(self):
        self.dm_wrapper.add_boolean_feature()
        self.dm_wrapper.add_categorical_text_feature()
        self.dm_wrapper.add_numerical_target()
        self.assert_feature_names(self.dm_wrapper.data_model.find_text_features(), "categorical_text_feature1")

    def test_find_categorical_features_without_any(self):
        self.assertEqual([], self.dm_wrapper.data_model.find_categorical_features())

    def test_find_categorical_features(self):
        self.dm_wrapper.add_boolean_feature()
        self.dm_wrapper.add_categorical_text_feature()
        self.dm_wrapper.add_numerical_target()
        self.assert_feature_names(self.dm_wrapper.data_model.find_categorical_features(), "categorical_text_feature1")

    def test_find_ignored_features_without_any(self):
        self.assertEqual([], self.dm_wrapper.data_model.find_ignored_features())

    def test_find_ignored_features(self):
        self.dm_wrapper.add_boolean_feature()
        self.dm_wrapper.add_ignored_numerical_feature()
        self.dm_wrapper.add_numerical_target()
        self.assert_feature_names(self.dm_wrapper.data_model.find_ignored_features(), "ignored_numerical_feature1")

    def test_find_numerical_features(self):
        self.dm_wrapper.add_boolean_feature()
        self.dm_wrapper.add_ignored_numerical_feature()
        self.dm_wrapper.add_numerical_target()
        self.dm_wrapper.add_numerical_feature()
        self.dm_wrapper.add_numerical_feature()

        self.assert_feature_names(self.dm_wrapper.data_model.find_numerical_features(), "numerical_feature3", "numerical_feature4")

    def assert_feature_names(self, list_features, *names):
        """
        Asserts that the given list of features has the given names, in the
        exact order as they are given.
        """
        assert len(list_features) == len(names)
        for i, feature in enumerate(list_features):
            self.assertEqual(feature.get_name(), names[i])


