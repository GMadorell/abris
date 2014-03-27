from unittest import TestCase
from abris_transform.configuration.feature import Feature


class FeatureTests(TestCase):

    def setUp(self):
        self.f = Feature("feature", [])

    def test_name(self):
        f = Feature("name", [])
        self.assertEqual("name", f.get_name())

    def test_type_name(self):
        self.f.set_type_name("type_name")
        self.assertEqual("type_name", self.f.get_type_name())

    def test_get_type_name_should_raise_if_set_is_not_called_before(self):
        with self.assertRaises(AssertionError):
            self.f.get_type_name()

    def test_is_categorical(self):
        f = Feature("categorical feature", ["categorical"])
        self.assertTrue(f.is_categorical())

    def test_is_categorical_false(self):
        self.assertFalse(self.f.is_categorical())

    def test_is_categorical_case_insensitive(self):
        f = Feature("categorical feature", ["caTeGorIcaL"])
        self.assertTrue(f.is_categorical())

    def test_is_target(self):
        f = Feature("target feature", ["target"])
        self.assertTrue(f.is_target())

    def test_is_target_false(self):
        self.assertFalse(self.f.is_target())

    def test_is_target_case_insensitive(self):
        f = Feature("target feature", ["TarGeT"])
        self.assertTrue(f.is_target())

    def test_has_characteristic(self):
        f = Feature("some feature", ["characteristic"])
        self.assertTrue(f.has_characteristic("characteristic"))

    def test_has_characteristic_false(self):
        self.assertFalse(self.f.is_target())

    def test_has_characteristic_case_insensitive(self):
        f = Feature("some feature", ["chaRactErisTic"])
        self.assertTrue(f.has_characteristic("characteristic"))
        self.assertTrue(f.has_characteristic("chaRactEristiC"))


