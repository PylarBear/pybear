# Authors:
#
#       Bill Sousa
#
# License: BSD 3 clause



from functools import wraps
import datetime



def logger(orig_func: callable) -> callable:

    """
    Generates a log file with the name {orig_func}.log that is saved in
    the same directory that contains this logger wrapper. The log file
    contains timestamps indicating when the wrapped function was called
    and what parameters were passed.


    Parameters
    ----------
    orig_func:
        callable - the function to be wrapped for logging


    Return
    ------
    wrapper -
        wrapper: callable - wrapped original function


    """


    import logging

    logging.basicConfig(
        filename=f'{orig_func.__name__}.log',
        level=logging.INFO
    )

    @wraps(orig_func)
    def wrapper(*args, **kwargs):

        logging.info(
            f'Ran at {datetime.datetime.now()} with args: {args}, '
            f'and kwargs: {kwargs}'
        )
        return orig_func(*args, **kwargs)


    return wrapper










