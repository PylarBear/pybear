# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#




from typing_extensions import Union

import numpy as np



def _val_instructions(
    _instructions: dict[str, Union[list[int], None]],
    _shape: tuple[int, int]
) -> None:

    """
    _instructions must be a dictionary with 3 keys: 'keep', 'delete', and 'add'.
    values can only be None or list[int] (column indices in the data).
    all column indices must be in range of the number of features in the data.
    a column index cannot be in multiple lists.


    Parameters
    ----------
    _instructions:
        dict[str, Union[list[int], None]] - pizza make some dough here
    _shape:
        tuple[int, int] - (n_samples, n_features) of the data.


    Return
    -
        None


    """



    assert isinstance(_instructions, dict)
    assert len(_instructions) == 3
    assert 'keep' in _instructions
    assert 'delete' in _instructions
    assert 'add' in _instructions

    # values must be None or list[int]
    # col idx cannot be in more than one instruction
    # pizza, come back to this --- does the 'add' col also go into 'keep'?
    # if all the lists are combined into a set, then len(set) must
    # equal sum of lens of individual lists
    _used_idxs = []
    for k, v in _instructions.items():
        if v is None:
            continue
        else:
            assert isinstance(v, list)
            assert all(map(isinstance, v, (int for _ in v)))
            _used_idxs += v

    assert len(set(_used_idxs)) == len(_used_idxs), \
        f"a column index is in more that one set of instructions"

    assert np.all(0 <= np.array(_used_idxs)) and \
           np.all(np.array(_used_idxs) <= _shape[1] - 1)

































