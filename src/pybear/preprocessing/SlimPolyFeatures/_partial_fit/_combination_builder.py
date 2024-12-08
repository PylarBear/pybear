# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import itertools



def _combination_builder(
    _shape: tuple[int, int],
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
        tuple[int, int] - the (n_samples, n_features) shape of X
    _min_degree:
        int - pizza get this from SlimPoly
    _max_degree:
        int - pizza get this from SlimPoly
    _intx_only:
        bool - Whether to return only first-order interaction terms --- pizza check SlimPoly


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
    # and SlimPoly will just go straight to returning a column of ones.


    fxn = itertools.combinations if _intx_only else \
        itertools.combinations_with_replacement

    _combinations = \
    itertools.chain.from_iterable(
        fxn(list(range(_shape[1])), _deg) for _deg in range(_min_degree, _max_degree+1)
    )


    return _combinations

















