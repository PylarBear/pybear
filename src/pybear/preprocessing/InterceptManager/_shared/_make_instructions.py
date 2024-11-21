# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


from pybear.preprocessing.InterceptManager._validation._instructions import _val_instructions

from pybear.preprocessing.InterceptManager._type_aliases import (
    KeepType,
    InstructionType
)
from typing_extensions import Union
from typing import Iterable

import numpy as np



def _make_instructions(
    _keep: KeepType,
    constant_columns_: dict[int, any],
    _columns: Union[Iterable[str], None],
    _shape: tuple[int, int]
) -> InstructionType:

    """
    :param: keep instructions must have been condition into dict[str, any],
    int, or Literal['none'] only before this module.
    Based on the keep instructions provided, and the constant columns
    found during fitting, build a dictionary that gives explicit
    instructions about what constant columns to keep, delete, or add.

    The form of the dictionary is:
    {
        'keep': Union[None, list[constant column indices to keep]],
        'delete: Union[None, list[constant column indices to delete]],
        'add: Union[None, dict['{column name}', fill value]]
    }


    if keep == 'none', keep none, add none, delete all
    if keep == a dict, keep none, delete all, add value in last position
    if keep == int, keep that column, delete the remaining constant columns
    keep callable & feature name, and the remaining str literals should not
    get in here, should have been converted to int in _manage_keep


    Parameters
    ----------
    _keep:
        Union[int, Literal['none'], dict[str, any]] -
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
            dict['keep':[list[int], None], 'delete':[list[int], None], 'add':[list[int], None]] -
            instructions for keeping, deleting, or adding constant
            columns to be applied during :method: transform




    """

    # pizza brains do we want _instructions to have np.ndarray instead of list?

    # validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
    err_msg = f"'_keep' must be Literal['none'], dict[str, any], or int"
    try:
        iter(_keep)
        if not isinstance(_keep, (str, dict)):
            raise UnicodeError
        if isinstance(_keep, dict) and not isinstance(list(_keep.keys())[0], str):
            raise UnicodeError
        if isinstance(_keep, str) and _keep != 'none':
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
            raise AssertionError(err_msg)


    assert isinstance(constant_columns_, dict)
    if len(constant_columns_) > 0:
        assert all(map(
            isinstance, constant_columns_, (int for _ in constant_columns_)
        ))
    if _columns is not None and len(constant_columns_):
        assert max(constant_columns_) <= len(_columns) - 1
    # END validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    _sorted_constant_column_idxs = sorted(list(constant_columns_))

    _instructions: InstructionType = {
        'keep': None,   #pizza, dont really need this for operating on X, 'keep' just makes it easy to make kept_columns_ later on.
        'delete': None,
        'add': None
    }


    if len(_sorted_constant_column_idxs) == 0:
        # if there are no constant columns, skip out and return Nones
        pass
    elif isinstance(_keep, int):
        _instructions['keep'] = [_keep]
        _sorted_constant_column_idxs.remove(_keep)
        _instructions['delete'] = _sorted_constant_column_idxs
    elif isinstance(_keep, str) and _keep != 'none':
        raise ValueError(f"str 'keep' not 'none' has gotten into _make_instructions but should already be an int")
    elif callable(_keep):
        raise ValueError(f"callable 'keep' has gotten into _make_instructions but should already be an int")
    elif _keep == 'none':
        # if keep == 'none', keep none, add none, delete all
        _instructions['delete'] = _sorted_constant_column_idxs
    elif isinstance(_keep, dict):
        # if keep == a dict, keep none, delete all, add value in last position
        _instructions['delete'] = _sorted_constant_column_idxs
        _instructions['add'] = _keep
    else:
        raise Exception(f"algorithm failure, invalid 'keep': {_keep}")


    _val_instructions(_instructions, _shape)


    return _instructions







