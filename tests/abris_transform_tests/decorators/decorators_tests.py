from datetime import datetime
import unittest
import time

from abris_transform.decorators.run_once import func_once, method_once


class FuncOnceTests(unittest.TestCase):

    def test_function_call_once(self):

        @func_once
        def some_function():
            return datetime.now()

        results = []
        for i in xrange(30):
            result = some_function()
            results.append(result)
            first_result = results[0]
            for result in results:
                self.assertEqual(result, first_result)
            time.sleep(0.001)


class MethodOnceTests(unittest.TestCase):

    def test_method_call_once(self):

        class SomeClass(object):
            @method_once
            def some_method(self):
                return datetime.now()

        instance = SomeClass()

        results = []
        for i in xrange(30):
            result = instance.some_method()
            results.append(result)
            first_result = results[0]
            for result in results:
                self.assertEqual(result, first_result)
            time.sleep(0.001)


