# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#





from typing import Literal








def _build_attributes(
    _poly_duplicates: list[list[tuple[int, ...]]],  # pizza, must be the version that has X columns in it, if any
    _keep: Literal['first', 'last', 'random'],
    _rand_idxs: tuple[tuple[int, ...], ...]
) -> tuple[
    dict[tuple[int, ...]: tuple[int, ...]],
    dict[tuple[int, ...]: list[tuple[int, ...]]]
]:

    """

    Build dropped_poly_duplicates_ and kept_poly_duplicates_.

    dropped_poly_duplicates_ is the subset of poly_duplicates_ that is left
    out of the polynomial expansion. It is a dictionary with the excluded
    duplicates as keys. The value for each key is the combo that is kept
    that is identical to the key. pizza reword this.

    kept_poly_duplicates is the subset of poly_duplicates that is kept in
    the polynomial expansion. It is a dictionary with the kept combo tuples
    as keys. There should be only one representative from each set of duplicates.
    The values are lists of combo tuples that were from the same
    set of duplicates as the key; they are the combos that are omitted.



    Parameters
    ---------
    _poly_duplicates:
        list[list[tuple[int, ...]]] - The groups of duplicates found
        in the polynomial expansions across all partial fits. If any
        combos were equal to a column in X, then the X idx tuple
        ... (c_idx, ) ... must be included.
    _keep:
        Literal['first', 'last', 'random'] - when there is no X idx in
        a set of duplicates (all duplicates were only within POLY) then apply
        this rule to keep one of them and omit the rest.
    _rand_idxs:
        tuple[tuple[int, ...], ...] - the random indices to keep when
        :param: keep == 'random'. This must be static when :method:
        transform is called.


    Return
    ------
    -
        tuple[
            dropped_poly_duplicates_: dict[tuple[int, ...]: tuple[int, ...]],
            kept_poly_duplicates_: dict[tuple[int, ...], list[tuple[int, ...]]]
        ]:

    """

    # validation - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    assert isinstance(_poly_duplicates, list)
    for _list in _poly_duplicates:
        assert isinstance(_list, list)
        for _tuple in _list:
            assert isinstance(_tuple, tuple)
            assert all(map(isinstance, _tuple, (int for _ in _tuple)))
    assert _keep in ['first', 'last', 'random']
    assert isinstance(_rand_idxs, tuple)
    for _tuple in _rand_idxs:
        assert isinstance(_tuple, tuple)
        assert all(map(isinstance, _tuple, (int for _ in _tuple)))
    # END validation - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


    # need to know from :param: keep which one from each dupl set is kept,
    # all other poly_duplicates_ are dropped
    # kept_poly_duplicates_ is a dictionary whose keys are the kept
    # X idx tuple/poly combo tuple and values are a list of its associated poly combos that will be omitted.
    # dropped_poly_duplicates_ is a dictionary whose keys are the dropped poly combos
    # and the values is the idx tuple of the associated dupl that was kept.
    # need to have X tuples in here! use _poly_duplicates not poly_duplicates_
    kept_poly_duplicates_: dict[tuple[int, ...]: list[tuple[int, ...]]] = {}
    dropped_poly_duplicates_: dict[tuple[int, ...]: tuple[int, ...]] = {}
    for _dupl_set_idx, _dupl_set in enumerate(_poly_duplicates):

        # just to be sure, sort _dupl_set by asc len(tuple), then asc on tuple idxs
        _same_len_tuples = {i: [] for i in range(1, max(map(len, _dupl_set))+1)}
        for _tuple in _dupl_set:
            _same_len_tuples[len(_tuple)].append(_tuple)

        # pizza, this would cause exception
        if 1 in _same_len_tuples and len(_same_len_tuples[1]) > 1:
            raise AssertionError(f"algorithm failure. more than one X col idx in a poly dupl set.")
        _sorted_dupl_set = []
        for _len, _tuples in _same_len_tuples.items():
            _sorted_dupl_set += sorted(_tuples)

        del _tuples, _len, _tuples, _same_len_tuples
        # END sort ----------------------------------------------------------

        # if there is a column from X (len(tuple)==1) there should only be 1
        # if any, that automatically is kept and the rest are omitted
        if len(_sorted_dupl_set[0]) == 1:
            kept_poly_duplicates_[_sorted_dupl_set[0]] = _sorted_dupl_set[1:]
            dropped_poly_duplicates_ = {t: _sorted_dupl_set[0] for t in  _sorted_dupl_set[1:]}
        elif _keep == 'first':
            kept_poly_duplicates_[_sorted_dupl_set[0]] = _sorted_dupl_set[1:]
            dropped_poly_duplicates_ = {t: _sorted_dupl_set[0] for t in  _sorted_dupl_set[1:]}
        elif _keep == 'last':
            kept_poly_duplicates_[_sorted_dupl_set[-1]] = _sorted_dupl_set[:-1]
            dropped_poly_duplicates_ = {t: _sorted_dupl_set[-1] for t in  _sorted_dupl_set[:-1]}
        elif _keep == 'random':
            if _rand_idxs[_dupl_set_idx] not in _sorted_dupl_set:
                raise AssertionError(f"algorithm failure. static random keep tuple not in dupl_set.")
            kept_poly_duplicates_[_rand_idxs[_dupl_set_idx]] = \
                _sorted_dupl_set.remove(_rand_idxs[_dupl_set_idx])
            dropped_poly_duplicates_ = {}
            for t in _sorted_dupl_set:
                if t != _rand_idxs[_dupl_set_idx]:
                    dropped_poly_duplicates_ = {t: _rand_idxs[_dupl_set_idx]}
        else:
            raise Exception(f'algorithm failure')



    return dropped_poly_duplicates_, kept_poly_duplicates_











