# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


from typing import Union

import numpy as np


def _validate_verbose(
        _verbose: Union[bool, int, float]
    ) -> int:

    """
    Validate _verbose, the amount of verbosity to display to screen
    during the grid search.

    Take in a bool, int, or float, and return an integer in the range of
    0 to 10, inclusive. Non-numbers are rejected. Bool False is converted
    to zero, bool True is converted to 10. Floats are rounded to integers.
    Negative numbers are rejected. Numbers greater than 10 are set to 10.

    Parameters
    ---------
    _verbose:
        bool, int, float - the amount of verbosity to display to screen
        during the grid search.

    Return
    ------
    -
        _verbose: int - scaled from 0 to 10

    """


    err_msg = f"verbose must be a bool or a numeric > 0"

    try:
        if isinstance(_verbose, bool):
            if _verbose is True:
                _verbose = 10
            elif _verbose is False:
                _verbose = 0
        float(_verbose)
        if _verbose is np.nan:
            _verbose = 0
        _verbose = min(int(round(_verbose, 0)), 10)
    except:
        raise TypeError(err_msg)

    if _verbose < 0:
        raise ValueError(err_msg)

    del err_msg

    return _verbose















