# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



def _val_include_bias(_include_bias: bool) -> None:

    """

    Validate include_bias; must be bool


    Parameters
    ----------
    _include_bias:
        bool, default = False -


    Return
    ------
    -
        None


    """

    if not isinstance(_include_bias, bool):
        raise TypeError(f"'include_bias' must be bool")



