# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#





from pybear.preprocessing.SlimPolyFeatures._partial_fit._build_kept_poly_duplicates \
    import _build_attributes


import pytest



pytest.skip(reason=f"pizza not started, not finished", allow_module_level=True)


class TestBuildAttributes:

    # def _build_attributes(
    #     _poly_duplicates: list[list[tuple[int, ...]]],
    #     _keep: Literal['first', 'last', 'random'],
    #     _rand_idxs: tuple[tuple[int, ...], ...]
    # ) -> tuple[
    #     dict[tuple[int, ...]: tuple[int, ...]],
    #     dict[tuple[int, ...]: list[tuple[int, ...]]]
    # ]:

    """
    # need to know from :param: keep which one from each dupl set is kept,
    # all other poly_duplicates_ are dropped
    # kept_poly_duplicates_ is a dictionary whose keys are the kept
    # X idx tuple/poly combo tuple and values are a list of its associated poly combos that will be omitted
    # dropped_poly_duplicates_ is a dictionary whose keys are the dropped poly combos
    # and the values is the idx tuple of the associated dupl that was kept.
    # need to have X tuples in here! use _poly_duplicates not poly_duplicates_
    """

    kept_poly_duplicates_: dict[tuple[int, ...]: list[tuple[int, ...]]] = {}
    dropped_poly_duplicates_: dict[tuple[int, ...]: tuple[int, ...]] = {}
    for _dupl_set_idx, _dupl_set in enumerate(_poly_duplicates):

        # just to be sure, sort _dupl_set by asc len(tuple), then asc on tuple idxs
        _same_len_tuples = {i: [] for i in range(1, max(map(len, _dupl_set))+1)}
        for _tuple in _dupl_set:
            _same_len_tuples[len(_tuple)].append(_tuple)
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














