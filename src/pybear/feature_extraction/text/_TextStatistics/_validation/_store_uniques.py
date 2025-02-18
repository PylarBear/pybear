# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



def _val_store_uniques(_store_uniques: bool) -> None:

    """
    Validate store_uniques. Must be boolean.


    Parameters
    ----------
    _store_uniques:
        bool - whether to compile all unique strings seen across all
        partial fits.

    Return
    ------
    -
        None


    """


    if not isinstance(_store_uniques, bool):

        raise TypeError(f"'store_uniques' must be boolean")







