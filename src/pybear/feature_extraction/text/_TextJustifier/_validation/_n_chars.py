# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import numbers



def _val_n_chars(_n_chars: numbers.Integral) -> None:

    """
    Validate n_chars. Must be an integer greater than zero.


    Parameters
    ----------
    _n_chars:
        numbers.Integral - the number of characters per line.


    Return
    ------
    -
        None

    """



    err_msg = f"'n_chars' must be an integer greater than zero."


    if not isinstance(_n_chars, numbers.Integral):
        raise TypeError(err_msg)

    if isinstance(_n_chars, bool):
        raise TypeError(err_msg)

    if _n_chars < 1:
        raise ValueError(err_msg)






