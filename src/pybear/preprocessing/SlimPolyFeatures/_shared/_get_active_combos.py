# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



def _get_active_combos(
    _combos: list[tuple[int, ...]],
    poly_constants_: dict[tuple[int, ...]: any],
    dropped_poly_duplicates_: dict[tuple[int, ...]: tuple[int, ...]]
) -> tuple[tuple[int, ...], ...]:

    """
    Pizza. Find the tuples of index combinations that will be in the
    polynomial expansion. This supports both _transform and
    _get_feature_names_out.
    Index tuples in :param: _combos that are in :param:
    dropped_poly_duplicates_ or in :param: poly_constants_ are omitted
    from the expansion.


    Parameters
    ----------

    _combos:
        list[tuple[int, ...]] -
    dropped_poly_duplicates_:
        dict[tuple[int, ...], tuple[int, ...]] -
    dropped_poly_duplicates_:
        dict[tuple[int, ...], tuple[int, ...]] -
    _n_jobs:
        Union[numbers.Integral, None] - the number of parallel jobs to
        use when building the polynomial expansion.


    Return
    ------
    -
        _active_combos:
        tuple[tuple[int, ...], ...] - the index tuple combinations to be
        kept in the polynomial expansion.

    """



    # validation - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    assert isinstance(_combos, list)
    for _tuple in _combos:
        assert isinstance(_tuple, tuple)
        assert all(map(isinstance, _tuple, (int for _ in _tuple)))
    assert isinstance(dropped_poly_duplicates_, dict)
    for k, v in dropped_poly_duplicates_.items():
        assert isinstance(k, tuple)
        assert all(map(isinstance, k, (int for _ in k)))
        assert isinstance(v, tuple)
        assert all(map(isinstance, v, (int for _ in v)))
    assert isinstance(poly_constants_, dict)
    assert all(map(isinstance, poly_constants_, (tuple for _ in poly_constants_)))
    # END validation - - - - - - - - - - - - - - - - - - - - - - - - - - - -


    _active_combos = []
    for _combo in _combos:

        if _combo in dropped_poly_duplicates_:
            continue

        if _combo in poly_constants_:
            continue

        _active_combos.append(_combo)



    return tuple(_active_combos)



