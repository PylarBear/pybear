# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Literal




def _identify_combos_to_keep(
    poly_duplicates_: list[list[tuple[int, ...]]],
    _keep: Literal['first', 'last', 'random'],
    _rand_combos: tuple[tuple[int, ...], ...]
) -> tuple[tuple[int, ...], ...]:

    """
    Apply two rules to determine which X idx / poly combo to keep from a
    set of duplicates:

        1) if there is a column from X in the dupl set (there should only
        be one, if any!) then override :param: keep and keep the column
        in X (X cannot be mutated by SlimPoly!)

        2) if the only duplicates are in the polynomial expansion, then
        apply :param: keep to the set of duplicate combos in
        poly_duplicates_ to find the combo to keep. If :param: keep is
        'random', then the random tuples are selected prior to this
        module in _lock_in_random_combos() and are passed here via
        :param: _rand_combos.


    Parameters
    ----------
    poly_duplicates_:
        list[list[tuple[int, ...]]] - a list of the groups of identical
        columns, containing lists of tuples of column index positions in
        the originally fit data. Columns from the original data itself
        can be in a group of duplicates, along with any duplicates from
        the polynomial expansion. It is important that poly_duplicates_
        is sorted correctly before it gets here. Sorted correctly means
        each group of duplicates is sorted on degree first (number of
        indices in the tuple) then on the indices themselves. Then the
        groups of duplicates are sorted between each other by applying
        the same rule across the first term in each group.
    _keep:
        Literal['first', 'last', 'random'] - The strategy for keeping a
        single representative from a set of identical columns in the
        polynomial expansion. See the long explanation in the main SPF
        module.
    _rand_combos:
        tuple[tuple[int, ...], ...] - An ordered tuple whose values are
        tuples of column indices from X, each tuple being selected from
        a group of duplicates in poly_duplicates_. One tuple is selected
        from each group of duplicates.


    Return
    ------
    -
        _idxs_to_keep: tuple[tuple[int, ...], ...] - An ordered tuple
        whose values are tuples of column indices from X, each tuple
        being selected from a group of duplicates in poly_duplicates_.
        This output differs from :param: _rand_combos in that
        _rand_combos just picks any random combo from each set of
        duplicates in poly_duplicates_ without regard to anything else
        simply to make the random tuples available to this module. But
        this module observes rules imposed as above, and may or may not
        use all or even any of the random tuples made available by
        _rand_combos.



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

    assert _keep in ['first', 'last', 'random']

    assert isinstance(_rand_combos, tuple)
    assert len(_rand_combos) == len(poly_duplicates_)
    for _tuple in _rand_combos:
        assert isinstance(_tuple, tuple)
        assert all(map(isinstance, _tuple, (int for _ in _tuple)))
    # _rand_combos might have len == 0, might not be any poly duplicates
    if len(_rand_combos):
        del _tuple
    # END validation - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


    _idxs_to_keep: list[tuple[int, ...]] = []

    for _dupl_set_idx, _dupl_set in enumerate(poly_duplicates_):

        # partial fits could have situations early in fitting where
        # columns in X look like they are duplicates but end up not
        # being duplicates after fitting is complete. because of this,
        # we cannot validate the number of X columns in a dupl set (which
        # at most should be one under normal situations) or it will
        # terminate or constantly warn, in addition to other warnings.
        # So for the first if statement, we assume, without validation,
        # that there is only one column from X, if any. And this
        # assumption should be correct at transform time if all the
        # no-op blocks in place in the main SPF module work correctly
        # when there are duplicates in X, which will prevent :method:
        # transform from carrying out any nonsensical instructions made
        # here.

        if len(_dupl_set[0]) == 1:
            # this overrides :param: keep, even for 'random'
            # if there is one, there can only be one, and that
            # automatically is kept and the rest (which must be in poly)
            # are omitted
            _idxs_to_keep.append(_dupl_set[0])
        elif _keep == 'first':
            _idxs_to_keep.append(_dupl_set[0])
        elif _keep == 'last':
            _idxs_to_keep.append(_dupl_set[-1])
        elif _keep == 'random':
            if _rand_combos[_dupl_set_idx] not in _dupl_set:
                raise AssertionError(
                    f"algorithm failure. static random keep tuple not in "
                    f"the respective dupl_set."
                )
            # setting random to _dupl_set[0] is now being done earlier, in _lock_in_rand_combos.
            # _rand_combos[_dupl_set_idx] should already be _dupl_set[0] if len(_dupl_set[0])==1
            if len(_dupl_set[0]) == 1:
                assert _rand_combos[_dupl_set_idx] == _dupl_set[0]

            _idxs_to_keep.append(_rand_combos[_dupl_set_idx])
        else:
            raise Exception(f"algorithm failure. keep not in ['first', 'last', 'random'].")


    assert len(_idxs_to_keep) == len(poly_duplicates_)

    # it is important that if there is only one tuple it be returned
    # like ((0,1),).  The list-to-tuple method as used here is tested and
    # appears to be robust for this purpose.
    _idxs_to_keep: tuple[tuple[int, ...], ...] = tuple(_idxs_to_keep)


    return _idxs_to_keep




