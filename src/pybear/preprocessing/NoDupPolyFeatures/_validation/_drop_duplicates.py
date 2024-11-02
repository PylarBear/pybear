# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



def _val_drop_duplicates(
    _drop_duplicates: bool
) -> None:

    """
    Validate drop_duplicates; must be bool


    Parameters
    ----------
    _drop_duplicates:
        bool, default = False -


    Return
    ------
    -
        None


    """


    if not isinstance(_drop_duplicates, bool):
        raise TypeError(f"'drop_duplicates' must be bool")











