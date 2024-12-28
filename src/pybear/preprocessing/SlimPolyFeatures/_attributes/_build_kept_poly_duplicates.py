# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from copy import deepcopy



def _build_kept_poly_duplicates(
    poly_duplicates_: list[list[tuple[int, ...]]],
    _kept_combos: tuple[tuple[int, ...], ...]
) -> dict[tuple[int, ...], list[tuple[int, ...]]]:

    """
    Build kept_poly_duplicates_.

    kept_poly_duplicates_ is the subset of poly_duplicates_ that is kept
    in the polynomial expansion. It is a dictionary with the kept combo
    tuples as keys. There should be only one representative from each
    group of duplicates. The values are lists of combo tuples that were
    from the same group of duplicates as the key, but they are the combos
    that are omitted from the final expansion.


    Parameters
    ---------
    poly_duplicates_:
        list[list[tuple[int, ...]]] - The groups of duplicates found
        in the polynomial expansions across all partial fits. If any
        combos were equal to a column in X, then the X idx tuple
        ... (c_idx, ) ... must be included, must be first, and there can
        only be one.
    _kept_combos:
        tuple[tuple[int, ...], ...] - the combo to keep for each set of
        duplicates in poly_duplicates_. Length must equal the length of
        poly_duplicates_.


    Return
    ------
    -
        kept_poly_duplicates_: dict[tuple[int, ...], list[tuple[int, ...]]]


    """

    # validation - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    assert isinstance(poly_duplicates_, list)
    for _list in poly_duplicates_:
        assert isinstance(_list, list)
        assert len(_list) >= 2
        for _tuple in _list:
            assert isinstance(_tuple, tuple)
            assert len(_tuple) >= 1
            assert all(map(isinstance, _tuple, (int for _ in _tuple)))

    if len(poly_duplicates_):
        del _list, _tuple

    assert isinstance(_kept_combos, tuple)
    assert len(_kept_combos) == len(poly_duplicates_)
    for _tuple in _kept_combos:
        assert isinstance(_tuple, tuple)
        assert all(map(isinstance, _tuple, (int for _ in _tuple)))

    # _kept_combos might have len == 0, might not be any poly duplicates
    if len(_kept_combos):
        del _tuple
    # END validation - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


    # need to know from :param: _kept_combos which one from each dupl set
    # is kept, all other poly_duplicates_ are dropped

    kept_poly_duplicates_: dict[tuple[int, ...], list[tuple[int, ...]]] = {}
    for _dupl_set_idx, _dupl_set in enumerate(poly_duplicates_):

        # dont need to sort poly_duplicates_ or _dupl_sets here, taken care of
        # on the way out of _merge_partialfit_dupls

        _kept_combo = _kept_combos[_dupl_set_idx]
        _dropped_combos = deepcopy(_dupl_set)
        try:
            _dropped_combos.remove(_kept_combo)
        except:
            raise AssertionError(
                f"algorithm failure. the combo in _kept_combos is not in "
                f"_dupl_set."
            )

        assert len(_dropped_combos) >= 1
        kept_poly_duplicates_[_kept_combo] = _dropped_combos


    if len(poly_duplicates_):
        del _kept_combo, _dropped_combos, _dupl_set_idx, _dupl_set


    return kept_poly_duplicates_











