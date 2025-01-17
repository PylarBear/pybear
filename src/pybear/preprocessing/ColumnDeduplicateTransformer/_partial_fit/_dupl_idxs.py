# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


from .._type_aliases import DataContainer
from typing_extensions import Union

from numbers import Real
import itertools
from collections import defaultdict

from ._find_duplicates import _find_duplicates




def _dupl_idxs(
    _X: DataContainer,
    _duplicates: Union[list[list[int]], None],
    _rtol: Real,
    _atol: Real,
    _equal_nan: bool,
    _n_jobs: Union[int, None]
) -> list[list[int]]:

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


    Parameters
    ----------
    _X:
        {array-like, scipy sparse matrix} of shape (n_samples,
        n_features) - the data to be deduplicated.
    _duplicates:
        Union[list[list[int]], None] - the duplicate columns carried over
        from the previous partial fits. Is None if on the first partial
        fit.
    _rtol:
        Real, default = 1e-5 - The relative difference tolerance for
        equality. Must be a non-boolean, non-negative, real number.
        See numpy.allclose.
    _atol:
        Real, default = 1e-8 - The absolute difference tolerance for
        equality. Must be a non-boolean, non-negative, real number.
        See numpy.allclose.
    _equal_nan:
        bool, default = False - When comparing pairs of columns row by
        row:
        If equal_nan is True, exclude from comparison any rows where one
        or both of the values is/are nan. If one value is nan, this
        essentially assumes that the nan value would otherwise be the
        same as its non-nan counterpart. When both are nan, this
        considers the nans as equal (contrary to the default numpy
        handling of nan, where numpy.nan != numpy.nan) and will not in
        and of itself cause a pair of columns to be marked as unequal.
        If equal_nan is False and either one or both of the values in
        the compared pair of values is/are nan, consider the pair to be
        not equivalent, thus making the column pair not equal. This is
        in line with the normal numpy handling of nan values.
    _n_jobs:
        Union[int, None] - The number of jobs to use with joblib Parallel
        while comparing columns. Default is to use processes, but can be
        overridden externally using a joblib parallel_config context
        manager. The default number of jobs is None, which uses the
        joblib default when n_jobs is None.


    Return
    ------
    -
        duplicates_: list[list[int]] - the groups of identical columns,
            indicated by their zero-based column index positions.


    """



    assert isinstance(_duplicates, (list, type(None)))
    if _duplicates is not None:
        for _set in _duplicates:
            assert isinstance(_set, list)
            assert all(map(isinstance, _set, (int for _ in _set)))
    try:
        float(_rtol)
    except:
        raise Exception
    try:
        float(_atol)
    except:
        raise Exception
    assert isinstance(_equal_nan, bool)
    assert isinstance(_n_jobs, (int, type(None)))


    duplicates_ = _find_duplicates(_X, _rtol, _atol, _equal_nan, _n_jobs)

    # if _duplicates is None, this is the first pass
    if _duplicates is None:
        return duplicates_
    elif _duplicates is not None:
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
        # _duplicates = [[0,1,2], [4,5]]
        # duplicates_ = [[0, 3], [1,2], [4,5]]
        # only [1,2] and [4,5] carry forward.

        # v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^
        # get the possible combinations of pairs for both duplicates, then
        # find the intersection, to find all pairs of numbers that are in the
        # same subset for both duplicates.

        all_old_comb = []
        for _set in _duplicates:
            all_old_comb += list(itertools.combinations(_set, 2))

        all_new_comb = []
        for _set in duplicates_:
            all_new_comb += list(itertools.combinations(_set, 2))

        _intersection = set(all_old_comb).intersection(all_new_comb)

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


        del find, union

        duplicates_ = list(components.values())

        # Sort each component and the final result for consistency
        duplicates_ = [sorted(component) for component in duplicates_]
        duplicates_ = sorted(duplicates_, key=lambda x: x[0])


        return duplicates_












