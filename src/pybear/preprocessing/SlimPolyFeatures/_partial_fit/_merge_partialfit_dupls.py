# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing_extensions import Union

from collections import defaultdict
import itertools






def _merge_partialfit_dupls(
    _old_duplicates: Union[list[list[tuple[int, ...]]], None],
    _new_duplicates: list[list[tuple[int, ...]]],
) -> list[list[tuple[int, ...]]]:

    """
    Find groups of identical columns for the current partial fit with
    _find_duplicates. Compare the newest duplicates found in the current
    partial fit with previous duplicates found on earlier partial fits
    and meld together to produce overall duplicates. Any columns
    previously not identified as equal but currently are equal, are
    coincidentally equal and are not added to the final list. Columns
    previously found to be equal but are not currently equal are removed
    from the final lists of duplicates. The only duplicates retained are
    those columns found to be identical for all partial fits.

    Pizza say something about sorting the merged dupls.


    Parameters
    ----------
    _old_duplicates:
        Union[list[list[tuple[int, ...]]], None] - the duplicate columns
        carried over from the previous partial fits. Is None if on the
        first partial fit.
    _new_duplicates:
        list[list[tuple[int, ...]]] - the duplicate columns found during
        the current partial fit. Is None if on the first partial fit.


    Return
    ------
    -
        duplicates_: list[list[tuple[int, ...]]] - the groups of
            identical columns, indicated by their zero-based column
            index positions.


    """



    assert isinstance(_old_duplicates, (list, type(None)))
    if _old_duplicates is not None:
        for _set in _old_duplicates:
            assert len(_set) >= 2
            assert isinstance(_set, list)
            assert all(map(isinstance, _set, (tuple for _ in _set)))
            for _tuple in _set:
                for _int in _tuple:
                    assert isinstance(_int, int)

    assert isinstance(_new_duplicates, list)
    for _set in _new_duplicates:
        assert len(_set) >= 2
        assert isinstance(_set, list)
        assert all(map(isinstance, _set, (tuple for _ in _set)))
        for _tuple in _set:
            for _int in _tuple:
                assert isinstance(_int, int)



    # if _duplicates is None, this is the first pass
    if _old_duplicates is None:
        _duplicates = _new_duplicates
    elif _old_duplicates is not None:
        # duplicates found on subsequent partial fits cannot increase the
        # number of duplicates over what was found on previous partial
        # fits. If later partial fits find new identical columns, it can
        # only be coincidental, as those columns were previously found to
        # be unequal. The number of duplicates can decrease, however, if
        # later partial fits find non-equality in columns that were
        # previously found to be equal.

        # compare the newest duplicates against the previously found
        # duplicates. Only a group of 2+ columns that appear together in
        # a set of dupls in both duplicates can carry forward. make sense?
        # _old_duplicates = [[0,1,2], [4,5]]
        # _new_duplicates = [[0, 3], [1,2], [4,5]]
        # only [1,2] and [4,5] carry forward.

        # v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^
        # get the possible combinations of pairs for both duplicates, then
        # find the intersection, to find all pairs of numbers that are in the
        # same subset for both duplicates.

        all_old_comb = []
        for _set in _old_duplicates:
            all_old_comb += list(itertools.combinations(_set, 2))

        all_new_comb = []
        for _set in _new_duplicates:
            all_new_comb += list(itertools.combinations(_set, 2))

        _intersection = set(all_old_comb).intersection(all_new_comb)

        if len(_old_duplicates) or len(_new_duplicates):
            del _set
        del all_old_comb, all_new_comb

        # v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^

        # v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^
        # use this "union-find" stuff that CHATGPT came up with to convert
        # pairs of duplicates like [(0,1), (1,2), (0,2), (4,5)] to [[0,1,2], [4,5]]

        # Find connected components using union-find
        # Union-Find data structure
        parent = {}

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])  # Path compression
            return parent[x]

        def union(x, y):
            root_x = find(x)
            root_y = find(y)
            if root_x != root_y:
                parent[root_y] = root_x

        # Initialize Union-Find
        for x, y in _intersection:
            if x not in parent:
                parent[x] = x
            if y not in parent:
                parent[y] = y
            union(x, y)

        # Group elements by their root
        components = defaultdict(list)
        for node in parent:
            root = find(node)
            components[root].append(node)

        _duplicates = list(components.values())


    # Sort each component and the final result for consistency
    # within dupl sets, sort on len asc, then within the same lens sort on values asc
    _duplicates = [sorted(component, key=lambda x: (len(x), x)) for component in _duplicates]
    # across all dupl sets, only look at the first value in a dupl set, sort on len asc,
    # then values asc
    _duplicates = sorted(_duplicates, key=lambda x: (len(x[0]), x[0]))

    # if any dupl set contains more than 1 tuple of len==1 (i.e., more than one column from X)
    # raise exception for duplicate columns in X
    for _dupl_set in _duplicates:
        # pizza this may need to come out or change to warn.
        if sum(map(lambda x: x==1, map(len, _dupl_set))) > 1:
            raise AssertionError(
                f"There are duplicate columns in X. Use pybear "
                f"ColumnDeduplicateTransformer to remove them before using SlimPoly."
            )


    return _duplicates












