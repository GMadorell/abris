def method_once(method):
    """
    A decorator that runs a method only once.
    """
    attribute_name = "_%s_once_result" % id(method)

    def decorated(self, *args, **kwargs):
        try:
            return getattr(self, attribute_name)
        except AttributeError:
            setattr(self, attribute_name, method(self, *args, **kwargs))
            return getattr(self, attribute_name)

    return decorated


def func_once(func):
    """
    A decorator that runs a function only once."
    """

    def decorated(*args, **kwargs):
        try:
            return decorated._once_result
        except AttributeError:
            decorated._once_result = func(*args, **kwargs)
            return decorated._once_result

    return decorated

