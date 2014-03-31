
def run_once(function):
    """
    Decorator that will make the decorated function execute and return once and on
    all the next executions simply return None without executing it.
    """
    def wrapper(*args, **kwargs):
        if not wrapper.has_run:
            wrapper.has_run = True
            return function(*args, **kwargs)
    wrapper.has_run = False
    return wrapper
