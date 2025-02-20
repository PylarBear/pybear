# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Sequence

import numbers

from ..._validation._1D_str_sequence import _val_1D_str_sequence



def _view_snippet_validation(
    VECTOR: Sequence[str],
    idx: numbers.Integral,
    span: numbers.Integral
) -> None:


    _val_1D_str_sequence(VECTOR)
    if len(VECTOR) == 0:
        raise ValueError(f"'VECTOR' cannot be empty")


    # idx -- -- -- -- -- -- -- -- -- --
    err_msg = f"'idx' must be a non-negative integer in range of the given vector"
    if not isinstance(idx, numbers.Integral):
        raise TypeError(err_msg)
    if isinstance(idx, bool):
        raise TypeError(err_msg)
    if idx not in range(0, len(VECTOR)):
        raise ValueError(err_msg)
    del err_msg
    # END idx -- -- -- -- -- -- -- -- --

    # span -- -- -- -- -- -- -- -- -- --
    err_msg = f"'span' must be an integer > 3"
    if not isinstance(span, numbers.Integral):
        raise TypeError(err_msg)
    if isinstance(span, bool):
        raise TypeError(err_msg)
    if span < 3:
        raise ValueError(err_msg)
    del err_msg
    # END span -- -- -- -- -- -- -- -- --




