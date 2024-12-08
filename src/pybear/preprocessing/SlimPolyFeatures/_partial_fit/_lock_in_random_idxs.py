# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#




import itertools

import numpy as np






def _lock_in_random_idxs(
    poly_duplicates_: list[list[tuple[int, ...]]],
    _combinations: list[tuple[int, ...]]
) -> tuple[tuple[int, ...], ...]:

    """

    mention something somewhere that we can randomly pick columns for
    dupl groups where a column from X is included, e.g., [[(1,), (1,2), 1,3)]],
    because it will just be ignored when we choose columns to keep, because
    in this case, the column from X is always kept.


    PIZZA REWRITE THIS
    :method: transform needs to mask the same indices for all batch-wise
    transforms, otherwise the outputted batches will have different
    columns. When :param: keep is set to 'random', :method: transform
    needs a static set of random column indices to use repeatedly, rather
    than a set of dynamic indices that are regenerated with each call to
    :method: transform.

    Goal: Create a static set of random indices that is regenerated with
    each call to :method: partial_fit, but is unchanged when :method:
    transform is called.

    This module builds a static ordered tuple of randomly selected
    indices, one index from each set of duplicates. For example,
    a simple case would be if :param: duplicates is [[(1,2), (3,5)], [(0, 8)]],
    then a possible _rand_idxs
    might look like ((1,2), (0,8)). THE ORDER OF THE INDICES IN _rand_idxs IS
    CRITICALLY IMPORTANT AND MUST ALWAYS MATCH THE ORDER IN :param:
    poly_duplicates_.

    This module assumes that 'keep' == 'random', even though that may
    not be the case. This makes the static list ready and waiting for
    use by :method: transform should at any time 'keep' be changed to
    'random' via :method: set_params after fitting.


    Parameters
    ----------
    poly_duplicates_: list[list[tuple[int]] - the groups of identical
        columns, indicated by their zero-based column index positions.
    _combinations:
        list[tuple[int]] - the combinations ot column indices being
        multiplied together to produce the polynomial columns.


    Return
    ------
    -
        _rand_idxs: tuple[tuple[int]] - An ordered tuple whose values
        are a sequence of column indices, one index selected from each
        set of duplicates in :param: duplicates.


    """



    # validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    # poly_duplicates_ must be list of lists of tuples of ints
    assert isinstance(poly_duplicates_, list)
    for _set in poly_duplicates_:
        assert isinstance(_set, list)
        assert len(_set) >= 1
        for _tuple in _set:
            assert isinstance(_tuple, tuple)
            assert all(map(isinstance, _tuple, (int for _ in _tuple)))
            # all tuples must be poly
            assert len(_tuple) >= 2

    # all idx tuples in poly_duplicates_ must be unique
    a = set(itertools.chain(*poly_duplicates_))
    b = list(itertools.chain(*poly_duplicates_))
    assert len(a) == len(b), f"{a=}, {len(a)=}, {b=}, {len(b)=}"
    del a, b

    try:
        iter(_combinations)
        assert not isinstance(_combinations, (str, dict))
        for item in _combinations:
            assert isinstance(item, tuple)
            assert len(item) >= 2
            assert all(map(isinstance, item, (int for _ in item)))
    except:
        raise AssertionError(
            f"'_combinations' must be Iterable[tuple[int, ...]]"
        )

    # END validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *



    _rand_idxs = []

    for _idx, _set in enumerate(poly_duplicates_):

        # we can just randomly pick anything from _set even if _set[0] turns
        # out to be from X, because then all of this is overruled anyway
        _keep_tuple_idx = np.random.choice(np.arange(len(_set)))

        _rand_idxs.append(_set[_keep_tuple_idx])

    # this cant be a set, it messes up the order against duplicates_
    _rand_idxs = tuple(_rand_idxs)


    # output validation ** * ** * ** * ** * ** * ** * ** * ** * ** * **

    err_msg = (
        f"'_rand_idxs' must be a tuple of tuples of integers, each tuple "
        f"of integers must be in _combinations, and a single tuple from "
        f"each set of duplicates must be represented, that is, "
        f"len(_rand_idxs)==len(poly_duplicates_)."
    )

    assert isinstance(_rand_idxs, tuple), err_msg
    # all idx tuples in _rand_idxs must be in combinations
    if len(_rand_idxs):
        assert all([_ in _combinations for _ in _rand_idxs]), err_msg
    # len _rand_idxs must match number of sets of duplicates
    assert len(_rand_idxs) == len(poly_duplicates_), \
        err_msg + f"{_rand_idxs=}, {poly_duplicates_=}"
    # if there are duplicates, every entry in _rand_idxs must match one tuple
    # in each set from poly_duplicates_
    for _idx, _dupl_set in enumerate(poly_duplicates_):
        assert _rand_idxs[_idx] in _dupl_set, \
            err_msg + f'rand tuple = {_rand_idxs[_idx]}, dupl set = {_dupl_set}'


    return _rand_idxs










