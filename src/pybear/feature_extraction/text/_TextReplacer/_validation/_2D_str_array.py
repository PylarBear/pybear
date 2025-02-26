# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Sequence



def _val_2D_str_array(
    _str_array: Sequence[Sequence[str]]
) -> None:

    """
    Validate things that are expected to be 2D arrays of strings. Need
    to consider raggedness, which means arrays may not be in convenient
    containers like numpy arrays and cannot be quickly turned into
    numpy arrays.


    Parameters
    ----------
    _str_array:
        Sequence[Sequence[str]] - something that is expected to be a 2D
        array of strings.


    Return
    ------
    -
        None

    """


    try:
        iter(_str_array)
        if isinstance(_str_array, (str, dict)):
            raise Exception
        map(iter, _str_array)
        if any(map(isinstance, _str_array, ((str, dict) for _ in _str_array))):
            raise Exception
        if not all([all(map(isinstance, i, (str for _ in i))) for i in _str_array]):
            raise Exception
    except:
        raise TypeError(f"'expected a 2D array of strings'")











