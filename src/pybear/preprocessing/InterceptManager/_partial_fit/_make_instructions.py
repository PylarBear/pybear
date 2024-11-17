# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


from pybear.preprocessing.InterceptManager._validation._instructions import _val_instructions

from .._type_aliases import (
    KeepType,
    InstructionType,
    ColumnsType
)

import numpy as np



def _make_instructions(
    _keep: KeepType,
    constant_columns_: dict[int, any],
    _columns: ColumnsType,
    _shape: tuple[int, int]
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
    _columns:
        Union[Iterable[str], None] - pizza pizza! pan pan!
    _shape:
        tuple[int, int] - the (n_samples, n_features) shape of the data.


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
        f"'_keep' must be Literal['first', 'last', 'random', 'none'], dict[str, any], int, str, or callable"
    try:
        iter(_keep)
        if not isinstance(_keep, (str, dict)):
            raise UnicodeError
        # pizza cant validate str really, could be column name, could be anything + keep literals
        if isinstance(_keep, dict) and not isinstance(list(_keep.keys())[0], str):
            raise UnicodeError
    except UnicodeError:
        raise AssertionError(err_msg)
    except:
        try:
            float(_keep)
            if isinstance(_keep, bool):
                raise UnicodeError
            if int(_keep) != _keep:
                raise UnicodeError
            _keep = int(_keep)
        except UnicodeError:
            raise ValueError(err_msg)
        except:
            if not callable(_keep):
                raise AssertionError(err_msg)


    assert isinstance(constant_columns_, dict)
    if len(constant_columns_) > 0:
        assert all(map(
            isinstance, constant_columns_, (int for _ in constant_columns_)
        ))
    # END validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    _sorted_constant_column_idxs = sorted(list(constant_columns_))

    _instructions: InstructionType = {
        'keep': None,   #pizza, dont really need this for operating on X, 'keep' just makes it easy to make kept_columns_ later on.
        'delete': None,
        'add': None
    }


    if len(constant_columns_) == 0:
        # if there are no constant columns, skip out and return Nones
        return _instructions
    elif isinstance(_keep, int):
        # this is the first place where we could validate whether the _keep int is actually
        # a constant column in the data
        if _keep not in constant_columns_:
            raise ValueError(f"'keep' column index has been set to {_keep}, but that column is not constant.")
        # pizza did this 24_11_15_16_54_00 and was so tired. doublecheck this.
        _instructions['keep'] = [_keep]
        _sorted_constant_column_idxs.remove(_keep)
        _instructions['delete'] = _sorted_constant_column_idxs
    elif isinstance(_keep, str) and _keep not in ('first', 'last', 'random', 'none'):
        # then must be a header.
        # this is the first place where we could validate whether the _keep str is actually
        # a constant column in the data

        # pizza
        # shouldnt need to validate str keep on None header, that should have
        # been done in validation

        idx = np.arange(len(_columns))[_columns==_keep][0]
        if idx not in constant_columns_:
            raise ValueError(f"'keep' has been set to column '{_keep}', but that column is not constant.")
        # pizza did this 24_11_15_16_54_00 and was so tired. doublecheck this.
        _instructions['keep'] = [idx]
        _sorted_constant_column_idxs.remove(idx)
        _instructions['delete'] = _sorted_constant_column_idxs
    elif callable(_keep):
        raise ValueError(f"callable 'keep' has gotten into _make_instructions but should already be an int")
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


    _val_instructions(_instructions, _shape)


    return _instructions







