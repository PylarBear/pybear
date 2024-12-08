# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import warnings




def _val_scan_X(
    _scan_X: bool
) -> None:

    """
    Validate _scan_X; must be bool


    Parameters
    ----------
    _scan_X:
        bool, default = False - pizza finish this


    Return
    ------
    -
        None


    """


    if not isinstance(_scan_X, bool):
        raise TypeError(f"'scan_X' must be bool")


    if _scan_X is False:
        warnings.warn(
            f"'scan_X' is set to False. Do this with caution, only when "
            f"you are certain that X does not have constant or duplicate "
            f"columns. Otherwise the results from :method: transform will "
            f"be nonsensical."
        )







