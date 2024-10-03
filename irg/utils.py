import functools
from typing import Callable


def placeholder(func: Callable):
    @functools.wraps(func)
    def wrapped_function(*args, **kwargs):
        print(f"Executing placeholder {func.__name__} ... "
              f"(this may not have the actual behavior as per described in the paper)")
        return func(*args, **kwargs)
    return wrapped_function
