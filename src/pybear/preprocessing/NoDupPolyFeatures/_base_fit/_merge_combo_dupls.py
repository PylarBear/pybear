# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#





def _merge_combo_dupls(
    _dupls_for_this_combo: list[tuple[int]],
    _poly_dupls_current_partial_fit: list[list[tuple[int]]]
) -> list[list[tuple[int]]]:

    """

    if the X idxs or poly idxs are in any of the dupl sets in
    _poly_dupls_current_partial_fit, then append the current
    combo idxs to that list.
    otherwise add the entire _dupls_for_this_combo to
    _poly_dupls_current_partial_fit.


    Parameters
    ----------
    _dupls_for_this_combo:
        list[tuple[int]] - single python list of tuples of ints, like [(1,), (1,2)]
        len != 1
        if len == 0, then the column combo is not a dupl of any column in X or POLY
        if len == 2, then the first tuple is (X_idx,) if the combo matched a column
        in X, or is (poly_idx1,poly_idx2,...) if the combo matched a column in POLY.
        the second tuple is the combo idxs.
        so if combo (X_1, X_5, X_6) matched poly (X_1, X_5), then
        _dupls_for_this_combo would look like [(X_1, X_5), (X_1, X_5, X_6)]

    _poly_dupls_current_partial_fit:
        list[list[tuple[int]]] - python list of python lists of tuples of ints,
        like [[(1,), (2,)], [(3,5), (4,5)]]
        sets of duplicates found in the current partial fit.
        no restrictions on shapes, can be empty



    """


    # validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **
    assert isinstance(_dupls_for_this_combo, list)
    assert len(_dupls_for_this_combo) in [0, 2]
    for _idx_tuple in _dupls_for_this_combo:
        assert isinstance(_dupls_for_this_combo, tuple)
        assert all(map(isinstance, _idx_tuple, (int for _ in _idx_tuple)))

    assert isinstance(_poly_dupls_current_partial_fit, list)
    for _dupl_list in _poly_dupls_current_partial_fit:
        assert isinstance(_dupl_list, list)
        for _idx_tuple in _dupl_list:
            assert isinstance(_idx_tuple, tuple)
            assert all(map(isinstance, _idx_tuple, (int for _ in _idx_tuple)))
    # END validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **


    if len(_dupls_for_this_combo):
        if not len(_poly_dupls_current_partial_fit):
            # if _poly_dupls_current_partial_fit is empty, put _dupls_for_this_combo in
            _poly_dupls_current_partial_fit.append(_dupls_for_this_combo)
        else:
            # if _poly_dupls_current_partial_fit is not empty, look if a dupl set in it
            # has the X idx or poly idxs in it already....
            for _dupl_set_idx, _dupls in enumerate(_poly_dupls_current_partial_fit):
                if _dupls_for_this_combo[0] in _dupls:
                    # if yes, put the current combo idxs into that dupl set
                    _poly_dupls_current_partial_fit[_dupl_set_idx].append(_dupls_for_this_combo[-1])
                    break
            else:
                # if not, put the entire _dupls_for_this_combo into _poly_dupls_current_partial_fit
                _poly_dupls_current_partial_fit.append(_dupls_for_this_combo)

            # there should only be unique entries in _poly_dupls_current_partial_fit
            import itertools  # pizza move this!
            all_dupl_idxs = itertools.chain(*_poly_dupls_current_partial_fit)
            assert len(set(all_dupl_idxs)) == len(list(all_dupl_idxs))
            del all_dupl_idxs

    # else
    # if there are no _dupls_for_this_combo, then
    # _poly_dupls_current_partial_fit does not change

    return _poly_dupls_current_partial_fit





