# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from .._type_aliases import (
    KeepType,
    InstructionType
)

import numpy as np



def _make_instructions(
    _keep: KeepType,
    constant_columns_: dict[int, any]
) -> InstructionType:

    """
    Based on the keep instructions provided, and the constant columns
    found during fitting, build a dictionary that gives explicit
    instructions about what constant columns to keep, delete, or add.

    The form of the dictionary is:
    {
        'keep': Union[None, list[constant column indices to keep]],
        'delete: Union[None, list[constant column indices to delete]],
        'add: Union[None, dict['{column name}', fill value]]
    }


    if keep == 'first', keep first, add none, delete all but first
    if keep == 'last', keep last, add none, delete all but last
    if keep == 'random', keep a random idx, add none, delete remaining
    if keep == 'none', keep none, add none, delete all
    if keep == a dict, keep none, delete all, add value in last position


    Parameters
    ----------
    _keep:
        Union[Literal['first', 'last', 'random', 'none'], dict[str, any]] -
        pizza finish
    constant_columns_:
        dict[int, any] - finish your pizza!


    Return
    ------
    -
        _instructions:
            dict['keep':various, 'delete':various, 'add':various] -
            instructions for keeping, deleting, or adding constant
            columns to be applied during :method: transform




    """



    # validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
    err_msg = \
        f"'_keep' must be 'first', 'last', 'random', 'none', or dict[str, any]"
    try:
        iter(_keep)
        if not isinstance(_keep, (str, dict)):
            raise Exception
        if isinstance(_keep, str):
            assert _keep.lower() in ('first', 'last', 'random', 'none')
        if isinstance(_keep, dict):
            assert isinstance(list(_keep.keys())[0], str)
    except:
        raise AssertionError(err_msg)


    assert isinstance(constant_columns_, dict)
    if len(constant_columns_) > 0:
        assert all(map(
            isinstance, constant_columns_, (int for _ in constant_columns_)
        ))
    # END validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    _sorted_constant_column_idxs = sorted(list(constant_columns_))

    _instructions: InstructionType = {
        'keep': None,
        'delete': None,
        'add': None
    }


    if len(constant_columns_) == 0:
        # if there are no constant columns, skip out and return Nones
        return _instructions
    elif _keep == 'first':
        # if keep == 'first', keep first, add none, delete all but first
        _instructions['keep'] = [_sorted_constant_column_idxs[0]]
        _instructions['delete'] = _sorted_constant_column_idxs[1:]
        return _instructions
    elif _keep == 'last':
        # if keep == 'last', keep last, add none, delete all but last
        _instructions['keep'] = [_sorted_constant_column_idxs[-1]]
        _instructions['delete'] = _sorted_constant_column_idxs[:-1]
        return _instructions
    elif _keep == 'random':
        # if keep == 'random', keep a random idx, add none, delete remaining
        _rand_idx = np.random.choice(_sorted_constant_column_idxs)
        _instructions['keep'] = [_rand_idx]
        _sorted_constant_column_idxs.remove(_rand_idx)
        _instructions['delete'] = _sorted_constant_column_idxs
        del _rand_idx
        return _instructions
    elif _keep == 'none':
        # if keep == 'none', keep none, add none, delete all
        _instructions['delete'] = _sorted_constant_column_idxs
        return _instructions
    elif isinstance(_keep, dict):
        # if keep == a dict, keep none, delete all, add value in last position
        _instructions['delete'] = _sorted_constant_column_idxs
        _instructions['add'] = _keep

        return _instructions
    else:
        raise Exception(f"algorithm failure, invalid 'keep'")










