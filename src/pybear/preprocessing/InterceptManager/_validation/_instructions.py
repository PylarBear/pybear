# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing_extensions import Union
from typing import Literal
import warnings
import numpy as np



def _val_instructions(
    _instructions: dict[Literal['keep', 'delete', 'add'], Union[list[int], None]],
    _shape: tuple[int, int]
) -> None:

    """
    _instructions must be a dictionary with 3 keys: 'keep', 'delete',
    and 'add'. Values can only be None or list[int], where ints are
    column indices in the data. All column indices must be in range of
    the number of features in the data. A column index cannot be in
    multiple lists.


    Parameters
    ----------
    _instructions:
        dict[Literal['keep', 'delete', 'add'], Union[list[int], None]] -
        instructions for keeping, deleting, or adding constant columns,
        to be applied during :method: transform
    _shape:
        tuple[int, int] - (n_samples, n_features) shape of the data.


    Return
    ------
    -
        None


    """



    assert isinstance(_instructions, dict)
    assert len(_instructions) == 3
    assert 'keep' in _instructions
    assert 'delete' in _instructions
    assert 'add' in _instructions
    assert isinstance(_shape, tuple)

    # values must be None or list[int]
    # col idx cannot be in more than one instruction
    # if all the lists are combined into a set, then len(set) must
    # equal sum of lens of individual lists
    # raise if all columns in the data are deleted and not adding new intercept

    _used_idxs = []
    for k, v in _instructions.items():
        if v is None:
            continue
        elif isinstance(v, dict):
            assert k == 'add'
            assert isinstance(list(v.keys())[0], str)
        else:
            assert isinstance(v, list), f"{v=}"
            assert all(map(isinstance, v, (int for _ in v))), \
                f"_instructions['{k}'] = {list(_instructions.values())}"
            _used_idxs += v

    assert len(set(_used_idxs)) == len(_used_idxs), \
        f"a column index is in more that one set of instructions"

    assert np.all(0 <= np.array(_used_idxs)) and \
           np.all(np.array(_used_idxs) <= _shape[1] - 1)


    if np.array_equal(_instructions['delete'], range(_shape[1])):
        if _instructions['add'] is None:
            raise ValueError(
                f"All columns in the data are constant. The current :param: "
                f"keep configuration will delete all columns."
            )
        else:
            warnings.warn(
                f"All columns in the data are constant. The current :param: "
                f"keep configuration will delete all the original columns and "
                f"leave only the appended intercept."
            )





























