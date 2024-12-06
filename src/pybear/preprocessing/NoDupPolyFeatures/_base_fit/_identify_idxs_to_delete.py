# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Iterable, Literal
from typing_extensions import Union

from copy import deepcopy
import itertools

import numpy as np




def _identify_idxs_to_delete(
    _duplicates: list[list[int]],
    _keep: Literal['first', 'last', 'random'],
    _do_not_drop: Union[Iterable[int], Iterable[str], None],
    _columns: Union[Iterable[str], None],
    _rand_idxs: tuple[int, ...]
) -> dict[int, int]:

    """
    Apply the rules given by :param: keep, :param: conflict, and :param:
    do_not_drop to the sets of duplicates in :param: duplicates. Produce
    the removed_columns_ dictionary, which has all the deleted column
    indices as keys and the respective kept columns as values.


    Parameters
    ----------
    _duplicates:
        list[list[int]] - the groups of identical columns, indicated by
        their zero-based column index positions.
    _keep:
        Literal['first', 'last', 'random'] - The strategy for keeping a
        single representative from a set of identical columns. 'first'
        retains the column left-most in the data; 'last' keeps the column
        right-most in the data; 'random' keeps a single randomly-selected
        column of the set of duplicates.
    _columns:
        Union[Iterable[str], None] of shape (n_features,) - if fitting
        is done on a pandas dataframe that has a header, this is a
        ndarray of strings, otherwise is None.
    _rand_idxs:
        tuple[int] - An ordered tuple whose values are a sequence of
        column indices, one index selected from each set of duplicates
        in :param: duplicates. For example, if duplicates_ is
        [[1, 5, 9], [0, 8]], then a possible _rand_idxs might look like
        (1, 8).


    Return
    ------
    -
        removed_columns_: dict[int, int] - the keys are the indices of
        duplicate columns removed from the original data, indexed by
        their column location in the original data; the values are the
        column index in the original data of the respective duplicate
        that was kept.

    """





    # validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    # _duplicates must be list of list of ints
    assert isinstance(_duplicates, list)
    for _set in _duplicates:
        assert isinstance(_set, list)
        assert len(_set) >= 2
        assert all(map(isinstance, _set, (int for _ in _set)))

    # all idxs in duplicates must be unique
    __ = list(itertools.chain(*_duplicates))
    assert len(np.unique(__)) == len(__)
    del __

    assert isinstance(_keep, str)
    assert _keep.lower() in ['first', 'last', 'random']


    err_msg = "if not None, '_columns' must be an iterable of strings"
    if _columns is not None:
        try:
            iter(_columns)
            if isinstance(_columns, (str, dict)):
                raise Exception
        except:
            raise AssertionError(err_msg)

        assert all(map(isinstance, _columns, (str for _ in _columns))), err_msg


    err_msg = (
        f"'_rand_idxs' must be a tuple of integers 0 <= idx < X.shape[1], "
        f"and a single idx from each set of duplicates must be represented "
        f"in _rand_idxs"
    )
    assert isinstance(_rand_idxs, tuple), err_msg
    # all idxs in _rand_idxs must be in range of num features in X
    if len(_rand_idxs):
        assert min(_rand_idxs) >= 0, err_msg
        if _columns is not None:
            assert max(_rand_idxs) < len(_columns)
    # len _rand_idxs must match number of sets of duplicates
    assert len(_rand_idxs) == len(_duplicates)
    # if there are duplicates, every entry in _rand_idxs must match one idx
    # in each set of duplicates
    if len(_duplicates):
        for _idx, _dupl_set in enumerate(_duplicates):
            assert list(_rand_idxs)[_idx] in _dupl_set, \
                f'rand idx = {list(_rand_idxs)[_idx]}, dupl set = {_dupl_set}'

    # END validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    # apply the keep rules to the duplicate idxs


    removed_columns_ = {}
    for _idx, _set in enumerate(_duplicates):

        if _keep == 'first':
            _keep_idx = sorted(_set)[0]
        elif _keep == 'last':
            _keep_idx = sorted(_set)[-1]
        elif _keep == 'random':
            _keep_idx = list(_rand_idxs)[_idx]
        else:
            raise Exception

        __ = deepcopy(_set)
        __.remove(_keep_idx)
        for _ in __:
            removed_columns_[_] = int(_keep_idx)


    return removed_columns_










