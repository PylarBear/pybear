# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing_extensions import Union
import numpy.typing as npt

import operator
import functools

import numpy as np

from ._validation._remove_characters import _remove_characters_validation
from ._validation._wip_X import _val_wip_X



def _remove_characters(
    _WIP_X: Union[list[str], list[list[str]], npt.NDArray[str]],
    _is_2D: bool,
    _allowed_chars: Union[str, None],
    _disallowed_chars: Union[str, None]
) -> Union[list[str], list[list[str]], npt.NDArray[str]]:

    """
    Remove characters that are not allowed or are explicitly disallowed
    from the data. allowed_chars and disallowed_chars cannot
    simultaneously be strings and cannot simultaneously be None.


    Parameter
    ---------
    _WIP_X:
        Union[list[str], list[list[str]], npt.NDArray[str]] - the data
        from which to remove unwanted characters.
    _is_2D:
        bool - whether the data is a 1D or 2D container.
    _allowed_chars:
        str - the characters that are to be kept; cannot be passed if
        disallowed_chars is passed.
    _disallowed_chars:
        str - the characters that are to be removed; cannot be passed if
        allowed_chars is passed.


    Return
    ------
    -
        X: Union[list[str], list[list[str]], npt.NDArray[str]] - the
        data with unwanted characters removed.


    """

    # pizza there is a benchmark file for this from when this was in a prior state
    # revisit the benchmark file and see if it can be useful.


    _remove_characters_validation(_allowed_chars, _disallowed_chars)

    _val_wip_X(_WIP_X)

    if not isinstance(_is_2D, bool):
        raise TypeError(f"'_is_2D' must be boolean")


    _allowed_chars = _allowed_chars or ''
    _disallowed_chars = _disallowed_chars or ''


    def _remover_decorator(foo):

        """
        A wrapping function that serves to find the unique characters to
        remove and then remove them from the data using the wrapped
        function.


        """


        @functools.wraps(foo)
        def _remover(_X):

            nonlocal _allowed_chars, _disallowed_chars

            UNIQUES = ''
            for i in _X:
                UNIQUES = str("".join(np.unique(list(UNIQUES + i))))

            for char in UNIQUES:
                if (len(_allowed_chars) and char not in _allowed_chars) \
                        or char in _disallowed_chars:
                    _X = foo(_X, char)

            del UNIQUES

            return _X

        return _remover


    @_remover_decorator
    def _list_remover(_X, _char):  # _X must be 1D
        return list(map(operator.methodcaller("replace", _char, ''), _X))


    @_remover_decorator
    def _ndarray_remover(_X, _char):
        return np.char.replace(_X.astype(str), _char, '')



    if not _is_2D:  # MUST BE LIST OF strs

        if isinstance(_WIP_X, list):
            _WIP_X = [_ for _ in _list_remover(_WIP_X) if _ != '']

        elif isinstance(_WIP_X, np.ndarray):
            _WIP_X = _ndarray_remover(_WIP_X)
            _WIP_X = _WIP_X[(_WIP_X != '')]

    elif _is_2D:

        if isinstance(_WIP_X, list):

            for row_idx, list_of_strings in enumerate(_WIP_X):
                _WIP_X[row_idx] = \
                    [_ for _ in _list_remover(list_of_strings) if _ != '']

        elif isinstance(_WIP_X, np.ndarray):

            for row_idx, vector_of_strings in enumerate(_WIP_X):
                _WIP_X[row_idx] = _ndarray_remover(vector_of_strings)
                # if this is a full array it will throw a fit for cannot
                # cast shorter vector into the full array
                try:
                    _WIP_X[row_idx] = _WIP_X[row_idx][(_WIP_X[row_idx] != '')]
                except:
                    pass


    return _WIP_X







