# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



def _val_sparse_output(_sparse_output: bool) -> None:

    """

    Validate sparse_output; must be bool


    Parameters
    ----------
    _sparse_output:
        bool, default = False -


    Return
    ------
    -
        None


    """

    if not isinstance(_sparse_output, bool):
        raise TypeError(f"'sparse_output' must be bool")












