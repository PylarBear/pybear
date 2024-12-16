# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Literal




def _identify_combos_to_keep(
    _poly_duplicates: list[list[tuple[int, ...]]],  # pizza, must be the version that has X columns in it, if any
    _keep: Literal['first', 'last', 'random'],
    _rand_combos: tuple[tuple[int, ...], ...]
) -> tuple[tuple[int, ...], ...]:

    """

    Apply two rules to determine which X idx / poly combo to keep from a set of
    duplicates:
    1) if there is a column from X in the dupl set (there should only be one!)
        then override :param: keep and keep the column in X (X cannot be mutated by SlimPoly!)
    2) if the only duplicates are in poly, then apply :param: keep to the set of
        duplicate combos to find the combo to keep.


    Parameters
    ----------
    _poly_duplicates:
        list[list[tuple[int, ...]]],  # pizza, must be the version that has X columns in it, if any
    _keep:
        Literal['first', 'last', 'random'] -
    _rand_combos:
        tuple[tuple[int, ...], ...] -


    Return
    ------
    -
        _idxs_to_keep: tuple[tuple[int, ...], ...]



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

    if len(_poly_duplicates):
        del _list, _tuple

    assert _keep in ['first', 'last', 'random']
    assert isinstance(_rand_combos, tuple)
    assert len(_rand_combos) == len(_poly_duplicates)
    for _tuple in _rand_combos:
        assert isinstance(_tuple, tuple)
        assert all(map(isinstance, _tuple, (int for _ in _tuple)))
    # _kept_combos might have len > 0, might not be any poly duplicates
    if len(_rand_combos):
        del _tuple
    # END validation - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


    _idxs_to_keep: list[tuple[int, ...]] = []

    for _dupl_set_idx, _dupl_set in enumerate(_poly_duplicates):

        if len(_dupl_set[0]) == 1:
            # the overrides :param: keep, even for 'random'


            # pizza revisit this validation... the current thinking 24_12_09_15_50_00
            # is that because partial fits could have situations early in fitting where
            # columns look like they are duplicates (but end up being not duplicates)
            # we cant validate about number of X columns in a dupl set!
            # if len(_dupl_set[1]) == 1:
            #     raise AssertionError(f"algorithm failure. more than one X column in dupl set in _identify_idxs_to_keep")


            # if any, that automatically is kept and the rest (which must be in poly) are omitted
            _idxs_to_keep.append(_dupl_set[0])
        elif _keep == 'first':
            _idxs_to_keep.append(_dupl_set[0])
        elif _keep == 'last':
            _idxs_to_keep.append(_dupl_set[-1])
        elif _keep == 'random':
            if _rand_combos[_dupl_set_idx] not in _dupl_set:
                raise AssertionError(f"algorithm failure. static random keep tuple not in dupl_set.")
            # setting random to _dupl_set[0] is now being done earlier, in _lock_in_rand_combos.
            # _rand_combos[_dupl_set_idx] should already be _dupl_set[0] if len(_dupl_set[0])==1
            if len(_dupl_set[0]) == 1:
                assert _rand_combos[_dupl_set_idx] == _dupl_set[0]

            _idxs_to_keep.append(_rand_combos[_dupl_set_idx])
        else:
            raise Exception(f"algorithm failure. keep not in ['first', 'last', 'random'].")


    assert len(_idxs_to_keep) == len(_poly_duplicates)

    _idxs_to_keep: tuple[tuple[int, ...], ...] = tuple(_idxs_to_keep)


    return _idxs_to_keep




