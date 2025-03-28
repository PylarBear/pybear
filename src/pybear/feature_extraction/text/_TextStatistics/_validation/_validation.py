# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from .._type_aliases import XContainer

from ._X import _val_X

from ._store_uniques import _val_store_uniques





def _validation(
    _X: XContainer,
    _store_uniques: bool
) -> None:

    """
    Centralized hub for validation. See the individual modules for more
    details.


    Parameters
    ----------
    _X:
        XContainer - The text data. Must be a 1D list-like or 2D
        array-like of strings.
    _store_uniques:
        bool - If True, all attributes and print methods are fully
        informative. If False, the 'string_frequencies_' and 'uniques_'
        attributes are always empty, and functionality that depends on
        these attributes have reduced capability.


    Returns
    -------
    -
        None

    """


    _val_X(_X)

    _val_store_uniques(_store_uniques)






