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
    Validate _scan_X; must be bool.


    Parameters
    ----------
    _scan_X:
        bool - SPF requires that the data being fit has no columns of
        constants and no duplicate columns. When True, SPF does not
        assume that the user knows these states of the data and diagnoses
        them, which can be very expensive to do, especially finding
        duplicate columns. If it is known that there are no constant or
        duplicate columns in the data, setting this to False can greatly
        reduce the cost of the polynomial expansion. When in doubt,
        pybear recommends setting this to True (the default). When this
        is False, it is possible to pass columns of constants or
        duplicates, but pybear will continue to operate by the stated
        design requirement, and the output will be nonsensical.


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







