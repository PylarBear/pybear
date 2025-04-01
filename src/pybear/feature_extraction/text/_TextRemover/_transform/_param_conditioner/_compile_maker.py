# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing_extensions import Union

import numbers
import re



def _compile_maker(
    _remove: Union[
        str, re.Pattern, tuple[Union[str, re.Pattern], ...],
        list[Union[None, str, re.Pattern, tuple[Union[str, re.Pattern], ...]]]
    ],
    _n_rows: numbers.Integral
) -> list[Union[list[None], list[re.Pattern]]]:

    """
    Convert any string literals to re.compile and map _remove to a list.
    Do not forget to escape string literals!


    Returns
    -------
    -
        _remove: list[Union[list[None], list[re.Pattern]]] - the remove
        criteria mapped to [None] or [re.Pattern, ...] for every row in
        whatever data _remove is associated with.


    Parameters
    ----------
    _remove:
        Union[str, re.Pattern, tuple[Union[str, re.Pattern], ...],
        list[Union[None, str, re.Pattern, tuple[Union[str, re.Pattern],
        ...]]]] - the string removal criteria as passed to the
        TextRemover instance.
    _n_rows:
        numbers.Integral - the number of rows in whatever data is
        associated with _remove.


    """

    if _remove is None:
        raise TypeError(f"'_remove' is None, should have been handled elsewhere")

    if isinstance(_remove, list) and len(_remove) != _n_rows:
        raise ValueError(f"validation failure: len(list(_remove)) != _n_rows")

    # -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

    if isinstance(_remove, str):
        _remove = [[re.compile(re.escape(_remove))] for _ in range(_n_rows)]
    elif isinstance(_remove, re.Pattern):
        _remove = [[_remove] for _ in range(_n_rows)]
    elif isinstance(_remove, tuple):
        # str becomes re.compile, re.compile stays re.compile, cant be None
        _remove = [
            re.compile(re.escape(i)) if isinstance(i, str) else i for i in _remove
        ]
        # ensure no duplicates
        _remove = [list(set(_remove)) for j in range(_n_rows)]
    elif isinstance(_remove, list):
        for _idx, k in enumerate(_remove):  # len is _n_rows by definition
            if k is None:
                _remove[_idx] = [None]
            elif isinstance(k, str):
                _remove[_idx] = [re.compile(re.escape(k))]
            elif isinstance(k, re.Pattern):
                _remove[_idx] = [k]
            elif isinstance(k, tuple):
                # str becomes re.compile, re.compile unchanged, cant be None
                k = [re.compile(re.escape(m)) if isinstance(m, str) else m for m in k]
                # ensure no duplicates
                _remove[_idx] = list(set(k))
                del k
            else:
                raise Exception(f'validation failure. {type(_remove)} in _remove.')
    else:
        raise Exception(f'validation failure. _remove is {type(_remove)}.')


    return _remove







