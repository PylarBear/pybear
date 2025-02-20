# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import numpy.typing as npt

import numpy as np

from .....data_validation.arg_kwarg_validater import arg_kwarg_validater
from .....data_validation import validate_user_input as vui




def _view_overall_uniques(
    _UNIQUES: npt.NDArray[str],
    _COUNTS: npt.NDArray[np.int32],
    _view_counts: bool
) -> None:


    """
    Print overall uniques and optionally counts to screen.


    Parameters
    ----------
    _UNIQUES:
        npt.NDArray[str] - The overall unique strings in the data.
    _COUNTS:
        npt.NDArray[np.int32] - The frequencies of the uniques.
    _view_counts:
        Union[bool, None] - whether to print the frequencies in addition
        to the uniques. If None, the user will be prompted by a Y/N
        input.



    Return
    ------
    -
        None


    """


    if not isinstance(_UNIQUES, np.ndarray):
        raise TypeError(f"'UNIQUES' must be a numpy array")

    if not isinstance(_COUNTS, np.ndarray):
        raise TypeError(f"'UNIQUES' must be a numpy array")

    if len(_UNIQUES) != len(_COUNTS):
        raise ValueError(f"'UNIQUES' and 'COUNTS' must be same len")

    arg_kwarg_validater(
        _view_counts,
        'view_counts',
        [True, False, None],
        'TC',
        'view_overall_uniques'
    )

    # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

    if _view_counts is None:
        _view_counts = {'Y': True, 'N': False}[
            vui.validate_user_str(f'View counts? (y/n) > ', 'YN')
        ]

    _max_len = max(map(len, _UNIQUES))
    _disp_len = 65
    _pad = 5
    _disp = min(_max_len, _disp_len)

    if _view_counts is True:

        _MASK = np.flip(np.argsort(_COUNTS))
        _UNIQUES = np.array(_UNIQUES)[_MASK]
        _COUNTS = np.array(_COUNTS)[_MASK]
        del _MASK

        print(f'\nOVERALL UNIQUES:')
        print(f'   UNIQUE'.ljust(_disp + _pad) + f'COUNT')
        print(f'   ' + f'-' * (_disp + _pad + 5))
        for idx in range(len(_UNIQUES)):
            print(f'   {_UNIQUES[idx][:_disp]}'.ljust(_disp + _pad) + f'{_COUNTS[idx]}')

    elif _view_counts is False:

        print(f'\nOVERALL UNIQUES:')
        print(f'   ' + '-' * 76)
        [print(f'   {_[:76]}') for _ in _UNIQUES]


    del _UNIQUES, _COUNTS
    del _max_len, _disp_len, _pad




















