# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import itertools
import numpy as np
import numpy.typing as npt



def _combination_builder(
    _shape: tuple[int, int],
    _constants: npt.NDArray[int],
    _min_degree: int,
    _max_degree: int,
    _intx_only: bool
) -> list[tuple[int]]:

    """
    Fill a list with tuples of column indices, with the indices indicating
    sets of columns to be multiplied together.

    Parameters
    ----------
    _shape:
        tuple[int, int] - the rows and columns in X
    _constants:
        NDArray[int] - columns of constants; excluded from combinations
    _min_degree:
        int - pizza get this from NoDup
    _max_degree:
        int - pizza get this from NoDup
    _intx_only:
        bool - Whether to return only first-order interaction terms --- pizza check NoDup


    Return
    ------
    -
        _combinations: list[tuple[int]] - the combinations of column
            indices for the polynomial expansion.


    """

    try:
        iter(_shape)
        if isinstance(_shape, (dict, str)):
            raise Exception
    except:
        raise AssertionError
    assert len(_shape) == 2
    assert isinstance(_constants, np.ndarray)
    assert _constants.dtype == np.int32
    assert isinstance(_min_degree, int)
    assert not isinstance(_min_degree, bool)
    assert _min_degree >= 0
    assert isinstance(_max_degree, int)
    assert not isinstance(_max_degree, bool)
    assert _max_degree > 0, f"max_degree == 0 shouldnt be getting in here"
    assert _max_degree >= _min_degree
    assert isinstance(_intx_only, bool)


    # forget about 0 degree, that is handled at the end of it all.

    _min_degree = max(1, _min_degree)
    # if max_degrees comes in as zero, then validation should have blown up.
    # pizza anticipates that if max_degree is ever set to zero, then
    # all of the machinery will be bypassed, including this module,
    # and NoDup will just go straight to returning a column of ones.


    fxn = itertools.combinations if _intx_only else \
        itertools.combinations_with_replacement

    COLUMNS = set(range(_shape[1])) - set(_constants)

    _combinations = \
    itertools.chain.from_iterable(
        fxn(list(COLUMNS), _deg) for _deg in range(_min_degree, _max_degree+1)
    )


    return _combinations

















