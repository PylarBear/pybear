# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



def _val_output_sparse(_output_sparse: bool) -> None:

    """

    Validate output_sparse; must be bool


    Parameters
    ----------
    _output_sparse:
        bool, default = False -


    Return
    ------
    -
        None


    """

    if not isinstance(_output_sparse, bool):
        raise TypeError(f"'output_sparse' must be bool")












