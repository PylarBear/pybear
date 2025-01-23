# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from .._type_aliases import OriginalDtypesType
from typing_extensions import Union

import numpy as np



def _original_dtypes_merger(
    _col_dtypes: OriginalDtypesType,
    _previous_col_dtypes: Union[OriginalDtypesType, None]
) -> OriginalDtypesType:

    """
    Merge the datatypes found for the current partial fit with the
    datatypes seen in previous partial fits. Prior to the existence of
    this module, MCT would raise if datatypes in the current partial
    fit did not match those from previous partial fits.

    # if _previous_col_dtypes is not None, check its dtypes against the
    dtypes in the currently passed data, use the hierarchy to set the
    merged dtype.

    --'obj' trumps everything, anything that was not 'obj' but is now
    'obj' becomes 'obj'

    --'float' trumps 'bin_int' and 'int'

    --'int' trumps 'bin_int'

    --anything that matches stays the same


    Parameters
    ----------
    _col_dtypes:
        npt.NDArray[Union[Literal['bin_int', 'int', 'float', 'obj']] -
        the datatypes found by MCT in the data for the current partial
        fit.
    _previous_col_dtypes:
        npt.NDArray[Union[Literal['bin_int', 'int', 'float', 'obj']] -
        the datatypes found by MCT in data seen in previous partial fits.


    Return
    ------
    -
        _merged_col_dtypes:
            npt.NDArray[Union[Literal['bin_int', 'int', 'float', 'obj']] -
            the datatypes merged based on the hierarchy.


    """


    # validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    _allowed = ['bin_int', 'int', 'float', 'obj']

    # _col_dtypes -- -- -- -- -- -- -- -- -- --
    _err_msg = (
        f"'_col_dtypes' must be a 1D vector of values in "
        f"{', '.join(_allowed)}. "
    )
    try:
        iter(_col_dtypes)
        if isinstance(_col_dtypes, (str, dict)):
            raise Exception
        if not len(np.array(list(_col_dtypes)).shape) == 1:
            raise Exception
        if not all(map(
            isinstance, _col_dtypes, (str for _ in _col_dtypes)
        )):
            raise Exception
        for _ in list(map(str.lower, _col_dtypes)):
            if _ not in _allowed:
                _addon = f"got '{_}'"
                raise UnicodeError
    except UnicodeError:
        raise ValueError(_err_msg + _addon)
    except:
        raise TypeError(_err_msg)
    # END _col_dtypes -- -- -- -- -- -- -- -- -- --

    # _previous_col_dtypes -- -- -- -- -- -- -- -- -- --
    if _previous_col_dtypes is not None:
        _err_msg = (
            f"'if not None, _previous_col_dtypes' must be a 1D vector of values "
            f"in {', '.join(_allowed)}. "
        )
        try:
            iter(_previous_col_dtypes)
            if isinstance(_previous_col_dtypes, (str, dict)):
                raise Exception
            if not len(np.array(list(_previous_col_dtypes)).shape) == 1:
                raise Exception
            if not all(map(
                isinstance,
                _previous_col_dtypes,
                (str for _ in _previous_col_dtypes)
            )):
                raise Exception
            for _ in list(map(str.lower, _previous_col_dtypes)):
                if _ not in _allowed:
                    _addon = f"got '{_}'"
                    raise UnicodeError
        except UnicodeError:
            raise ValueError(_err_msg + _addon)
        except:
            raise TypeError(_err_msg)
    # END _previous_col_dtypes -- -- -- -- -- -- -- -- -- --

    # joint -- -- -- -- -- -- -- -- -- --

    if _previous_col_dtypes is not None:
        assert len(_col_dtypes) == len(_previous_col_dtypes)

    # END joint -- -- -- -- -- -- -- -- -- --

    # END validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    if _previous_col_dtypes is None:
        return np.array(list(_col_dtypes))


    # if partial_fit has already seen data previously...

    _new_dtypes = []
    for _idx in range(len(_col_dtypes)):

        _current_dtype = _col_dtypes[_idx].lower()
        _old_dtype = _previous_col_dtypes[_idx].lower()

        if _current_dtype == 'obj' or _old_dtype == 'obj':
            _new = 'obj'
        elif _current_dtype == 'float' or _old_dtype == 'float':
            _new = 'float'
        elif _current_dtype == 'int' or _old_dtype == 'int':
            _new = 'int'
        else:
            _new = 'bin_int'

        _new_dtypes.append(_new)

    del _current_dtype, _old_dtype, _new


    return np.array(_new_dtypes)












