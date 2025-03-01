# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from .._type_aliases import OriginalDtypesType

import numpy as np

from .._validation._n_features_in import _val_n_features_in



def _val_original_dtypes(
    _original_dtypes: OriginalDtypesType,
    _n_features_in: int
) -> None:

    """
    Validate that datatypes in the passed '_original_datatypes' container
    are valid MCT internal datatypes. Allowed values are 'bin_int', 'int',
    'float', 'obj', case-sensitive. Validate number of entries against
    the number of features in the data.


    Parameters
    ----------
    _original_dtypes:
        npt.NDArray[Union[Literal['bin_int', int', 'float', 'obj']]] -
        the datatypes read from the data. must be a 1D list-like with
        values in 'bin_int', int', 'float', or 'obj'.
    _n_features_in:
        int - the number of features in the data.


    Return
    ------
    -
        None


    """


    _val_n_features_in(_n_features_in)

    _allowed = ['bin_int', 'int', 'float', 'obj']

    _err_msg = (
        f"'_col_dtypes' must be a 1D vector of values in {', '.join(_allowed)} "
        f"and the number of entries must equal the number of features in the "
        f"data."
    )
    try:
        iter(_original_dtypes)
        if isinstance(_original_dtypes, (str, dict)):
            raise Exception
        if not len(np.array(list(_original_dtypes)).shape) == 1:
            raise Exception
        if not len(_original_dtypes) == _n_features_in:
            _addon = f""
            raise UnicodeError
        if not all(map(
            isinstance, _original_dtypes, (str for _ in _original_dtypes)
        )):
            raise Exception
        for _ in _original_dtypes:
            if _ not in _allowed:
                _addon = f" got '{_}'."
                raise UnicodeError
    except UnicodeError:
        raise ValueError(_err_msg + _addon)
    except:
        raise TypeError(_err_msg)













