# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing_extensions import Union


def _val_sep(_sep: Union[str, None]) -> None:

    """
    Validate 'sep'. Must be a string or None.


    Parameters
    ----------
    _sep:
        Union[str, None] - The character sequence where the strings are
        separated.


    Return
    ------
    -
        None


    """




    if not isinstance(_sep, (str, type(None))):
        raise TypeError(f"'sep' must be str or None, got {type(_sep)}")








