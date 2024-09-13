# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#




from typing import Iterable
from typing_extensions import TypeAlias, Union

import numpy as np
import joblib

from ._type_aliases import CleanedTextType



def _normalize(
    _CLEANED_TEXT: CleanedTextType,
    _upper: bool,
    _n_jobs: int
    ) -> CleanedTextType:

    """
    Set all text in CLEANED_TEXT object to upper case (default) or lower case.


    """

    _is_list_of_lists: bool = all(
        map(isinstance, _CLEANED_TEXT, (str for _ in _CLEANED_TEXT))
    )


    # WILL PROBABLY BE A RAGGED ARRAY AND np.char WILL THROW A FIT, SO GO ROW BY ROW
    if _is_list_of_lists:

        def _case_mapper(_upper:bool):
            _ROW = np.fromiter(
                map(str.upper if _upper else str.lower, _CLEANED_TEXT[row_idx]),
                dtype='U40'
            )
            return _ROW


        for row_idx in range(len(_CLEANED_TEXT)):
            if _upper:
                _CLEANED_TEXT[row_idx] = np.fromiter(map(str.upper, _CLEANED_TEXT[row_idx]), dtype='U30')
            elif not _upper:
                _CLEANED_TEXT[row_idx] = np.fromiter(map(str.lower, _CLEANED_TEXT[row_idx]), dtype='U30')
    elif not _is_list_of_lists:  # LIST OF strs
        if _upper:
            _CLEANED_TEXT = np.fromiter(map(str.upper, _CLEANED_TEXT), dtype='U100000')
        elif not _upper:
            _CLEANED_TEXT = np.fromiter(map(str.lower, _CLEANED_TEXT), dtype='U100000')


    del _is_list_of_lists

    return _CLEANED_TEXT




