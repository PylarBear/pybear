# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from .._type_aliases import XContainer


from .....base._check_1D_str_sequence import check_1D_str_sequence



def _val_X(_X: XContainer) -> None:

    """
    Validate X. Must be 1D list-like vector of strings.


    Parameters
    ----------
    _X:
        XContainer - The text data. Must be a 1D list-like of strings.


    Returns
    -------
    -
        None

    """


    check_1D_str_sequence(_X, require_all_finite=False)

    if len(_X) == 0:
        raise ValueError(
            f"'strings' must be passed as a list-like vector of "
            f"strings, cannot be empty."
        )





