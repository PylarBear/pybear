# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing_extensions import Union



def _val_sep(_sep: Union[str, None]) -> None:

    """
    Validate sep. Must be a string or None.


    Parameters
    ----------
    _sep:
        Union[str, None] - the separator that joins words in the n-grams.


    Returns
    -------
    -
        None

    """


    if _sep is None:
        return


    if not isinstance(_sep, str):
        raise TypeError(f"'sep' must be a str or None.")












