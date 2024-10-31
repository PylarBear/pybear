# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Iterable
from typing_extensions import Union
from .._type_aliases import (
    ColumnsType,
    ConflictType,
    KeepType
)

from copy import deepcopy
import itertools

import numpy as np




def _identify_idxs_to_delete(
    _duplicates: list[list[int]],
    _keep: KeepType,   # Literal['first', 'last', 'random']
    _do_not_drop: Union[Iterable[int], Iterable[str], None],
    _columns: ColumnsType,   # Union[Iterable[str], None]
    _conflict: ConflictType   # Literal['raise', 'ignore']
) -> dict[int, int]:

    """

    Apply the rules given by :param: keep, :param: conflict, and :param:
    do_not_drop to the sets of duplicates in :param: duplicates. Produce
    the removed_columns_ dictionary, which has all the deleted column
    indices as keys and the respective kept columns as values.


    Parameters
    ----------
    _duplicates: list[list[int]] - the groups of identical columns,
        indicated by their zero-based column index positions.
    _keep:
        Literal['first', 'last', 'random'], default = 'first' -
        The strategy for keeping a single representative from a set
        of identical columns. 'first' retains the column left-most
        in the data; 'last' keeps the column right-most in the data;
        'random' keeps a single randomly-selected column of the set of
        duplicates.
    _do_not_drop:
        Union[Iterable[int], Iterable[str], None], default=None - A list
        of columns not to be dropped. If fitting is done on a pandas
        dataframe that has a header, a list of feature names may be
        provided. Otherwise, a list of column indices must be provided.
        If a conflict arises, such as two columns specified in :param:
        do_not_drop are duplicates of each other, the behavior is managed
        by :param: conflict.
    _columns:
        Union[Iterable[str], None] of shape (n_features,) - if fitting
        is done on a pandas dataframe that has a header, this is a
        ndarray of strings, otherwise is None.
    _conflict:
        Literal['raise', 'ignore'] - Ignored when :param: do_not_drop is
        not passed. Instructs CDT how to deal with a conflict between
        the instructions in :param: keep and :param: do_not_drop. A
        conflict arises when the instruction in :param: keep ('first',
        'last', 'random') is applied and column in :param: do_not_drop
        is found to be a member of the columns to be deleted. When :param:
        conflict is 'raise', an exception is raised in the case of such
        a conflict. When :param: conflict is 'ignore', there are 2
        possible scenarios:

        1) when only one column in :param: do_not_drop is among the
        columns to be deleted, the :param: keep instruction is overruled
        and the do_not_drop column is kept

        2) when multiple columns in :param: do_not_drop are among the
        columns to be deleted, the :param: keep instruction ('first',
        'last', 'random') is applied to the set of do-not-delete columns
        that would be marked for deletion --- this may not give the same
        result as applying the :param: keep instruction  to the entire
        set of duplicate columns. This also causes at least one member
        of the columns not to be dropped to be deleted.


    Return
    ------
    -
        removed_columns_: dict[int, int] - the keys are the indices of
        duplicate columns removed from the original data, indexed by
        their column location in the original data; the values are the
        respective column index in the original data of the respective
        duplicate that was kept.

    """





    # validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    assert isinstance(_duplicates, list)
    for _set in _duplicates:
        assert isinstance(_set, list)
        assert len(_set) >= 2
        assert all(map(isinstance, _set, (int for _ in _set)))

    __ = list(itertools.chain(*_duplicates))
    assert len(np.unique(__)) == len(__)

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


    err_msg = "if not None, 'do_not_drop' must be an iterable of integers or strings"
    if _do_not_drop is not None:
        try:
            iter(_do_not_drop)
            if isinstance(_do_not_drop, (dict, str)):
                raise Exception
        except:
            raise AssertionError(err_msg)

        assert not any(map(isinstance, _do_not_drop, (bool for _ in _do_not_drop)))
        assert all(
            map(isinstance, _do_not_drop, ((int, str) for _ in _do_not_drop))
        ), err_msg

        if isinstance(_do_not_drop[0], str) and _columns is None:
            raise AssertionError(
                f"if _columns is not passed, _do_not_drop can only be passed "
                f"as integers"
            )

    assert isinstance(_conflict, str)
    assert _conflict.lower() in ['raise', 'ignore']

    # END validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    # apply the keep, do_not_drop, and conflict rules to the duplicate idxs

    # do_not_drop can be None! if do_not_drop is strings, convert to idxs
    if _do_not_drop is not None and isinstance(_do_not_drop[0], str):
        _do_not_drop = [list(_columns).index(col) for col in set(_do_not_drop)]


    removed_columns_ = {}
    for _set in _duplicates:

        if _do_not_drop is None:
            _n = 0
        else:
            _dnd_idxs = sorted(set(_do_not_drop).intersection(_set))
            _n = len(_dnd_idxs)

        if _n == 0:   # no _do_not_drops in _set, or _do_not_drop is None
            if _keep == 'first':
                _keep_idx = min(_set)
            elif _keep == 'last':
                _keep_idx = max(_set)
            elif _keep == 'random':
                _keep_idx = np.random.choice(_set)

        elif _n == 1:

            # if the idx that is do_not_drop is the one we are keeping,
            # then all good

            _dnd_idx = _dnd_idxs[0]

            if _dnd_idx == min(_set) and _keep == 'first' or \
                _dnd_idx == max(_set) and _keep == 'last' or \
                _keep == 'random' or _conflict == 'ignore':
                _keep_idx = _dnd_idx
            else:
                if _columns is None:
                    __ = f""
                else:
                    __ = f", '{_columns[_dnd_idx]}'"
                raise ValueError(
                    f"duplicate indices={_set}, do_not_drop={_do_not_drop}, ",
                    f"keep={_keep}, wants to drop column index {_dnd_idx}{__}, "
                    f"conflict with do_not_drop."
                )

            del _dnd_idx

        elif _n > 1:

            if _conflict == 'ignore':
                # since _dnd_idxs has multiple values, apply the 'keep' rules to
                # that list
                if _keep == 'first':
                    _keep_idx = min(_dnd_idxs)
                elif _keep == 'last':
                    _keep_idx = max(_dnd_idxs)
                elif _keep == 'random':
                    _keep_idx = np.random.choice(_dnd_idxs)

            elif _conflict == 'raise':
                if _columns is None:
                    __ = f""
                else:
                    __ = " (" + ", ".join([_columns[_] for _ in _dnd_idxs]) + ")"
                raise ValueError(
                    f"duplicate indices={_set}, do_not_drop={_do_not_drop}, ",
                    f"keep={_keep}, wants to keep multiple column indices "
                    f"{', '.join(map(str, _dnd_idxs))}"
                    f"{__}, conflict with do_not_drop."
                )

        else:
            raise Exception

        __ = deepcopy(_set)
        __.remove(_keep_idx)
        for _ in __:
            removed_columns_[int(_)] = int(_keep_idx)


    return removed_columns_










