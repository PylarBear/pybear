# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import itertools

from ._num_combinations import _val_num_combinations



def _combination_builder(
    n_features_in_: int,
    _min_degree: int,
    _max_degree: int,
    _intx_only: bool
) -> list[tuple[int, ...]]:

    """
    Fill a list with tuples of column indices, with the indices indicating
    sets of columns to be multiplied together.
    Add pizza about validating for np.intp


    Parameters
    ----------
    n_features_in_:
        int - the number of features in X
    _min_degree:
        int - pizza get this from SlimPoly
    _max_degree:
        int - pizza get this from SlimPoly
    _intx_only:
        bool - Whether to return only first-order interaction terms --- pizza check SlimPoly


    Return
    ------
    -
        _combinations: list[tuple[int, ...]] - the combinations of column
            indices for the polynomial expansion.


    """


    assert isinstance(n_features_in_, int)
    assert not isinstance(n_features_in_, bool)
    assert n_features_in_ >= 1
    assert isinstance(_min_degree, int)
    assert not isinstance(_min_degree, bool)
    assert _min_degree >= 1, f"min_degree == 0 shouldnt be getting in here"
    assert isinstance(_max_degree, int)
    assert not isinstance(_max_degree, bool)
    assert _max_degree >= 2, f"max_degree in [0,1] shouldnt be getting in here"
    assert _max_degree >= _min_degree
    assert isinstance(_intx_only, bool)


    # forget about 0 degree, that is handled at the end of it all.

    _min_degree = max(2, _min_degree)
    # if max_degrees comes in as zero, then validation should have blown up.
    # pizza anticipates that if max_degree is ever set to zero, then
    # all of the machinery will be bypassed, including this module,
    # and SlimPoly will just go straight to returning a column of ones.


    fxn = itertools.combinations if _intx_only else \
        itertools.combinations_with_replacement

    _combinations = \
    list(itertools.chain.from_iterable(
        fxn(list(range(n_features_in_)), _deg) for _deg in range(_min_degree, _max_degree+1)
    ))

    # this checks the number of features in the output polynomial expansion for
    # indexability based on the max value allowed by np.intp
    _val_num_combinations(
        n_features_in_,
        _n_poly_features=len(_combinations),
        _min_degree=_min_degree,
        _max_degree=_max_degree,
        _intx_only=_intx_only
    )

    # PIZZA 24_12_10_16_11_00 _combinations MUST ALWAYS BE asc shortest
    # combos to longest combos, then sorted asc on combo idxs. maybe we should add a test
    # should be coming out of itertools like, but ensure always sorted
    _combinations = sorted(_combinations, key = lambda x: (len(x), x))


    return _combinations

















