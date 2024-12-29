# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from copy import deepcopy
import itertools




def _merge_combo_dupls(
    _dupls_for_this_combo: list[tuple[int, ...]],
    _poly_dupls_current_partial_fit: list[list[tuple[int, ...]]]
) -> list[list[tuple[int, ...]]]:

    """
    For the current partial fit, merge duplicates found for one combo
    with duplicates found for all other previous combos.

    In SPF :method: partial_fit, just before this is called, SPF has
    completed the check if combo produces a column of constants and has
    scanned the column against X and POLY. In that scan, SPF produced a
    list called '_dupls_for this_combo'. Merge this list with
    '_poly_dupls_current_partial_fit'.

    If the first entry in '_dupls_for this_combo' matches the first
    entry in any of the dupl groups in '_poly_dupls_current_partial_fit',
    then append the current combo tuple to that list. If not, then add
    the entire '_dupls_for_this_combo' list to
    '_poly_dupls_current_partial_fit'.


    Parameters
    ----------
    _dupls_for_this_combo:
        list[tuple[int, ...]] - single python list of tuples of ints,
        like [(1,), (1,2)].

        len cannot be 1.

        if len == 0,  the combo is not a dupl of any column in X or POLY.

        if len == 2, the second tuple is the current combo idxs.

        ---if the combo matched in X, then the first tuple is (X_idx,)

        ---if the combo matched in POLY, then the first tuple is the
            combo tuple from poly

        So if combo (X_2, X_6) matched poly (X_1, X_5), then
        _dupls_for_this_combo would look like [(X_1, X_5), (X_2, X_6)].

        if len is not in [0, 2], then there is a degenerate condition.
        When there are duplicate columns in X, if a combo is a  duplicate
        of one of them, then it is a duplicate of both of them, and there
        would be multiple duplicates from X in _dupls_for_this_combo.
        This condition is understood and handled by SPF. However, if the
        degenerate condition is multiple duplicates in POLY, then that
        is an algorithm failure which *should never happen*.

    _poly_dupls_current_partial_fit:
        list[list[tuple[int, ...]]] - list of lists of tuples of ints,
        like [[(1,), (3,5)], [(2,), (4,5)]].
        The sets of duplicates found in the current partial fit.
        No restrictions on shapes, can be empty.


    Return
    ------
    -
        __poly_dupls_current_partial_fit: list[list[tuple[int, ...]]] -
        the passed '_poly_dupls_current_partial_fit' updated with the
        duplicates from the current combo, if any.

    """


    # validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **
    assert isinstance(_dupls_for_this_combo, list)
    # always must be 0 or at least 2 entries in _dupls_for_this_combo,
    # unless there are duplicates in X, which is accommodated by SPF
    assert len(_dupls_for_this_combo) != 1
    for _posn, _idx_tuple in enumerate(_dupls_for_this_combo):
        assert isinstance(_idx_tuple, tuple)
        assert all(map(isinstance, _idx_tuple, (int for _ in _idx_tuple)))
        # last tuple in _dupls_for_this_combo must always be len >= 2
        # because it is the current combo
        if _posn == len(_dupls_for_this_combo)-1:
            assert len(_idx_tuple) >= 2
        # len(this tuple) must always be >= len(previous tuple)
        elif _posn > 0:
            assert len(_idx_tuple) >= len(_dupls_for_this_combo[_posn-1])
    # all _dupls_for_this_combo values must be unique
    assert len(set(_dupls_for_this_combo)) == len(_dupls_for_this_combo)

    assert isinstance(_poly_dupls_current_partial_fit, list)
    for _dupl_list in _poly_dupls_current_partial_fit:
        assert isinstance(_dupl_list, list)
        # always must be at least 2 entries in each _dupl_list
        assert len(_dupl_list) >= 2
        for _posn, _idx_tuple in enumerate(_dupl_list):
            assert isinstance(_idx_tuple, tuple)
            assert all(map(
                isinstance, _idx_tuple, (int for _ in _idx_tuple)
            ))
            # last tuple in _dupl_list must always be len >= 2
            if _posn == len(_dupl_list) - 1:
                assert len(_idx_tuple) >= 2
            # len(this tuple) must always be >= len(previous tuple)
            elif _posn > 0:
                assert len(_idx_tuple) >= len(_dupl_list[_posn-1])
    # all _poly_dupls_current_partial_fit values must be unique
    all_dupl_idxs = list(itertools.chain(*_poly_dupls_current_partial_fit))
    assert len(set(all_dupl_idxs)) == len(all_dupl_idxs), f'{all_dupl_idxs=}'
    del all_dupl_idxs

    # END validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **

    __poly_dupls_current_partial_fit = deepcopy(_poly_dupls_current_partial_fit)

    if len(_dupls_for_this_combo):
        if not len(_poly_dupls_current_partial_fit):
            # if _poly_dupls_current_partial_fit is empty,
            # put _dupls_for_this_combo in
            __poly_dupls_current_partial_fit.append(_dupls_for_this_combo)
        else:
            # if _poly_dupls_current_partial_fit is not empty, look if a
            # dupl set in it has the current X idx or poly idxs that is
            # duplicate of combo in it already....
            for _dupl_set_idx, _dupls in enumerate(_poly_dupls_current_partial_fit):
                if _dupls_for_this_combo[0] == _dupls[0]:
                    # if yes, put the current combo idxs into that dupl set
                    __poly_dupls_current_partial_fit[_dupl_set_idx].append(
                        _dupls_for_this_combo[-1]
                    )
                    break
            else:
                # if not, put the entire _dupls_for_this_combo into
                # _poly_dupls_current_partial_fit
                __poly_dupls_current_partial_fit.append(_dupls_for_this_combo)


    # else
    # if there are no _dupls_for_this_combo, then
    # _poly_dupls_current_partial_fit does not change

    return __poly_dupls_current_partial_fit





