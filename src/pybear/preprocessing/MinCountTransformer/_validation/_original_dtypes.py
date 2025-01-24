# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from .._type_aliases import OriginalDtypesType

import numpy as np



def _val_original_dtypes(
    _original_dtypes: OriginalDtypesType
) -> None:

    """
    Validate that datatypes in the passed '_original_datatypes' container
    are valid MCT internal datatypes. Allowed values are 'bin_int', 'int',
    'float', 'obj', case-sensitive.


    Parameters
    ----------
    _original_dtypes:
        Iterable[Union[Literal['bin_int', int', 'float', 'obj']]] -
        the datatypes read from the data. must be a 1D list-like with
        values in 'bin_int', int', 'float', or 'obj'.


    Return
    ------
    -
        None


    """


    _allowed = ['bin_int', 'int', 'float', 'obj']

    _err_msg = (
        f"'_col_dtypes' must be a 1D vector of values in {', '.join(_allowed)}. "
    )
    try:
        iter(_original_dtypes)
        if isinstance(_original_dtypes, (str, dict)):
            raise Exception
        if not len(np.array(list(_original_dtypes)).shape) == 1:
            raise Exception
        if not all(map(
            isinstance, _original_dtypes, (str for _ in _original_dtypes)
        )):
            raise Exception
        for _ in _original_dtypes:
            if _ not in _allowed:
                _addon = f"got '{_}'."
                raise UnicodeError
    except UnicodeError:
        raise ValueError(_err_msg + _addon)
    except:
        raise TypeError(_err_msg)













