# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Sequence




def _val_X(_X: Sequence[str]) -> None:

    """
    Validate X. Must be a 1D sequence of strings.


    Parameters
    ----------
    _X:
        Sequence[str] - the data


    Return
    ------
    -
        None

    """


    try:
        iter(_X)
        if isinstance(_X, (str, dict)):
            raise Exception
        if not all(map(isinstance, _X, (str for _ in _X))):
            raise Exception
    except Exception as e:
        raise TypeError(f"X must be a 1D sequence of strings")




