# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from numbers import Real
import numpy as np


def _merge_constants(
    _old_constants: dict[int, any],
    _new_constants: dict[int, any],
    _rtol: Real,
    _atol: Real
) -> dict[int, any]:

    """
    Merge the constants found in the current partial fit with those
    found in previous partial fits. Constant columns can only stay the
    same or decrease on later partial fits, never increase.


    Parameters
    ----------
    _old_constants:
        dict[int, any] - the column indices of constant columns found in
        previous partial fits and the value in the columns.
    _new_constants:
        dict[int, any] - the column indices of constant columns found in
        the current partial fit and the value in the columns.
    _rtol:
        numbers.Real - The relative difference tolerance for equality.
        Must be a non-boolean, non-negative, real number. See
        numpy.allclose.
    _atol:
        numbers.Real - The absolute difference tolerance for equality.
        Must be a non-boolean, non-negative, real number. See
        numpy.allclose.


    Return
    ------
    -
        _final_constants: dict[int, any] - the compiled column indices
            and values of constant columns found over all partial fits.

    """

    assert isinstance(_old_constants, dict)
    if len(_old_constants):
        assert isinstance(list(_old_constants.keys())[0], int)
    assert isinstance(_new_constants, dict)
    if len(_new_constants):
        assert isinstance(list(_new_constants.keys())[0], int)

    _final_constants = {}

    # pizza, u pin head, this needs work. what if _old_constants is {},
    # but it isnt the first pass... that means there cant be any columns
    # of constants!

    _base_constants = _new_constants if _old_constants == {} else _old_constants

    for _col_idx, _value in _base_constants.items():

        # for a column of constants to carry forward, the currently found
        # indices must be in the previously found indices, and the value of the
        # constant must be the same

        # if _old_constants is

        if _col_idx in _new_constants:
            # need to handle nan - dont use dict.get here
            if str(_value) == 'nan' and str(_new_constants[_col_idx]) == 'nan':
                _final_constants[int(_col_idx)] = _value
            elif _new_constants[_col_idx] == _value:
                # this should get strings (or ints, or maybe some floats)
                _final_constants[int(_col_idx)] = _value
            elif np.isclose(
                _new_constants[_col_idx],
                _value,
                rtol=_rtol,
                atol=_atol
            ):
                # this should get floats
                _final_constants[int(_col_idx)] = _value



    # verify that outgoing constants were in base constants:
    for _col_idx, _value in _final_constants.items():
        assert _col_idx in _base_constants
        # need to handle nan
        if str(_final_constants[_col_idx]) == 'nan':
            assert str(_final_constants[_col_idx]) == \
                   str(_base_constants[_col_idx])
        else:
            assert _final_constants[_col_idx] == _base_constants[_col_idx]


    return _final_constants








