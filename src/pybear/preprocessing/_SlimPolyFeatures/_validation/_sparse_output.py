# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



def _val_sparse_output(_sparse_output: bool) -> None:

    """
    Validate sparse_output; must be bool.


    Parameters
    ----------
    _sparse_output:
        bool - If set to True, the polynomial expansion is returned
        from :method: transform as a scipy sparse csr array. If set to
        False, the polynomial expansion is returned in the same format
        as passed to :method: transform.


    Return
    ------
    -
        None


    """


    if not isinstance(_sparse_output, bool):
        raise TypeError(f"'sparse_output' must be bool")












