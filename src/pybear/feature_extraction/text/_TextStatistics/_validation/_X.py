# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from .._type_aliases import XContainer


from .....base._check_dtype import check_dtype



def _val_X(_X: XContainer) -> None:

    """
    Validate X. Must be 1D list-like or 2D array-like of strings.


    Parameters
    ----------
    _X:
        XContainer - The text data.


    Returns
    -------
    -
        None

    """


    check_dtype(_X, allowed='str', require_all_finite=False)

    if len(_X) == 0:
        raise ValueError(f"'X' cannot be empty.")





