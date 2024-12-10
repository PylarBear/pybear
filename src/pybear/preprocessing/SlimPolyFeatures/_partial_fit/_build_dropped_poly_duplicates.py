# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from copy import deepcopy




def _build_dropped_poly_duplicates(
    _poly_duplicates: list[list[tuple[int, ...]]],  # pizza, must be the version that has X columns in it, if any
    _kept_combos: tuple[tuple[int, ...], ...]
) -> dict[tuple[int, ...]: tuple[int, ...]]:

    """

    Build dropped_poly_duplicates_.

    dropped_poly_duplicates_ is the subset of poly_duplicates_ that is left
    out of the polynomial expansion. It is a dictionary with the excluded
    duplicates as keys. The value for each key is the combo that is kept
    that is identical to the key. pizza reword this.


    Parameters
    ---------
    _poly_duplicates:
        list[list[tuple[int, ...]]] - The groups of duplicates found
        in the polynomial expansions across all partial fits. If any
        combos were equal to a column in X, then the X idx tuple
        ... (c_idx, ) ... must be included.
    _kept_combos:
        tuple[tuple[int, ...], ...] - the combo to keep for each set of
        duplicates in _poly_duplicates. length must equal the length of _poly_duplicates.


    Return
    ------
    -
        dropped_poly_duplicates_: dict[tuple[int, ...]: tuple[int, ...]]


    """

    # validation - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    assert isinstance(_poly_duplicates, list)
    for _list in _poly_duplicates:
        assert isinstance(_list, list)
        assert len(_list) >= 2
        for _tuple in _list:
            assert isinstance(_tuple, tuple)
            assert len(_tuple) >= 1
            assert all(map(isinstance, _tuple, (int for _ in _tuple)))

    assert isinstance(_kept_combos, tuple)
    assert len(_kept_combos) == len(_poly_duplicates)
    for _tuple in _kept_combos:
        assert isinstance(_tuple, tuple)
        assert all(map(isinstance, _tuple, (int for _ in _tuple)))

    del _list, _tuple
    # END validation - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


    # need to know from :param: keep which one from each dupl set is kept,
    # all other poly_duplicates_ are dropped
    # kept_poly_duplicates_ is a dictionary whose keys are the kept
    # X idx tuple/poly combo tuple and values are a list of its associated poly combos that will be omitted.
    # dropped_poly_duplicates_ is a dictionary whose keys are the dropped poly combos
    # and the values is the idx tuple of the associated dupl that was kept.
    # need to have X tuples in here! use _poly_duplicates not poly_duplicates_
    dropped_poly_duplicates_: dict[tuple[int, ...]: tuple[int, ...]] = {}
    for _dupl_set_idx, _dupl_set in enumerate(_poly_duplicates):

        # dont need to sort _poly_duplicates or _dupl_sets here, taken care of
        # on the way out of _merge_partialfit_dupls

        _kept_combo = _kept_combos[_dupl_set_idx]
        _dropped_combos = deepcopy(_dupl_set)
        try:
            _dropped_combos.remove(_kept_combo)
        except:
            raise AssertionError(
                f"algorithm failure. the combo in _kept_combos is not in _dupl_set."
            )

        assert len(_dropped_combos) >= 1

        for _combo in _dropped_combos:
            dropped_poly_duplicates_[_combo] = _kept_combo


    if len(_poly_duplicates):
        del _kept_combo, _dropped_combos, _dupl_set_idx


    return dropped_poly_duplicates_











