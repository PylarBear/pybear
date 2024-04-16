# Authors:
#
#       Bill Sousa
#
# License: BSD 3 clause


from functools import wraps
import datetime


def logger(orig_func):
    """Generates a log file with the name {orig_func}.log that is saved in the
    same directory that contains this logger wrapper. The log file contains
    timestamps indicating when the wrapped function was called and what
    parameters were passed.

    Parameters
    ---------
    None


    Returns
    ------
    wrapper: wrapped original function


    See Also
    ------
    None


    Notes
    ----
    None


    Example
    ------
    >>> from pybear.debug import _logger

    >>> @logger
    >>> def some_function(a, offset=1):
    >>>     return a + 2 + offset
    >>> print(some_function(4, offset=5))
    >>> 11
    """



    import logging
    logging.basicConfig(filename=f'{orig_func.__name__}.log',
                        level=logging.INFO)

    @wraps(orig_func)
    def wrapper(*args, **kwargs):
        logging.info(f'Ran at {datetime.datetime.now()} with args: {args}, '
                     f'and kwargs: {kwargs}')
        return orig_func(*args, **kwargs)

    return wrapper










