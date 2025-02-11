# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Sequence

import numpy as np



def _val_strings(strings: Sequence[str]) -> None:

    """
    Validate 'strings'.

    - Must be 1D sequence of strings
    - cannot be empty
    - all strings must be under 30 characters
    - individual strings cannot have spaces


    Parameters
    ----------
    strings:
        Sequence[str] - a single list-like vector of strings to report
        statistics for. Strings do not need to be in the pybear Lexicon.


    Return
    ------
    -
        None


    """



    err_msg = (f"'strings' must be passed as a list-like vector of strings, "
               f"cannot be empty.")

    try:
        iter(strings)
        if isinstance(strings, (dict, str)):
            raise Exception
        if not len(np.array(list(strings)).shape) == 1:
            raise Exception
    except:
        raise TypeError(err_msg)

    if len(strings) == 0:
        raise ValueError(err_msg)

    if not all(map(isinstance, strings, (str for _ in strings))):
        raise TypeError(err_msg)

    del err_msg











