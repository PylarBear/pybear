# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from ..._type_aliases import OriginalDtypesType

import numpy as np



def _val_original_dtypes(
    _original_dtypes: OriginalDtypesType
) -> None:

    """
    Validate that datatypes read from the data and being passed to
    _make_instructions are valid MCT internal datatypes.


    Parameters
    ----------
    _original_dtypes:
        npt.NDArray[Union[Literal['bin_int', int', 'float', 'obj']]] -
        the datatypes read from the data. must be a 1D numpy array with
        values in 'bin_int', int', 'float', or 'obj'.


    Return
    ------
    -
        None


    """


    if not isinstance(_original_dtypes, np.ndarray):
        raise TypeError(f"'_original_dtypes' must be a numpy array")

    ALLOWED = ['bin_int', 'int', 'float', 'obj']

    err_msg = f"entries in '_original_dtypes' must be {', '.join(ALLOWED)}"

    if not all(map(isinstance, _original_dtypes, (str for _ in _original_dtypes))):
        raise TypeError(err_msg)

    if not all(map(lambda x: x in ALLOWED, _original_dtypes)):
        raise ValueError(err_msg)

    del ALLOWED, err_msg






