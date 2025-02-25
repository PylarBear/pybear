# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Sequence

import numpy as np



def _val_1D_str_sequence(
    _str_sequence: Sequence[str]
) -> None:

    """
    Validate things that are expected to be 1D vectors of strings.


    Parameters
    ----------
    _str_sequence:
        Sequence[str] - something that is expected to be a 1D sequence
        of strings.


    Return
    ------
    -
        None

    """


    try:
        iter(_str_sequence)
        if isinstance(_str_sequence, (str, dict)):
            raise Exception
        if len(np.array(_str_sequence).shape) > 1:
            raise Exception
        if not all(map(isinstance, _str_sequence, (str for _ in _str_sequence))):
            raise Exception
    except:
        raise TypeError(f"'expected a 1D sequence of strings'")











