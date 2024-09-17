# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Iterable, Literal, Optional, Callable
from typing_extensions import Union

from copy import deepcopy

import numpy as np
import pandas as pd
import dask.array as da
import dask.dataframe as ddf
import dask_expr._collection as ddf2


def column_deduplicator(
    X: Iterable[Iterable[Union[int, float, str, bool]]],
    keep: Union[Literal['first'], Literal['last'], Literal['random']],
    do_not_drop: Optional[Union[Iterable[str], Iterable[int], Callable, None]]=None,
    verbose: Optional[bool]=False
) -> Iterable[Iterable[Union[int, float]]]:

    """





    Parameters
    ----------
    X:
        Iterable[Iterable[Union[int, float, str, bool]]] - Data to remove
        duplicate columns from.
    keep:
        Union[Literal['first'], Literal['last'], Literal['random'], str,
        Iterable[str]] -
        The strategy for keeping a single representative from a set of
        identical columns. 'first' retains the column left-most in the
        data; 'last' keeps the column right-most in the data; 'random'
        selects a single random column of the set to be retained.
        Pizza add some jargon about passing column names, lists of column names,
        substrings, yada yada yada pizza.
    do_not_drop:
        Union[Iterable[str], Iterable[int], Callable, None], default=None -
        Columns to never drop,
        overriding the positional 'keep' argument for the set of duplicates
        associated with the respective column. If a conflict arises, such as
        two columns specified in 'do_not_drop' are duplicates of each
        other, an error is raised.
    verbose:
        Optional[bool], default=False - display information to screen
        during duplicate column identification and removal.


    Return
    ------
    -
        X: Iterable[Iterable[int, float, str, bool]] - Deduplicated data;
        data with duplicate columns removed based on the given parameters.


    """


    # header handling - - - - -
    HEADER = None
    if isinstance(X, (pd.core.frame.DataFrame, ddf.core.Dataframe, ddf2.DataFrame)):
        HEADER = X.columns
        if isinstance(X, (ddf.core.Dataframe, ddf2.DataFrame)):
            X = X.to_dask_array(lengths=True)
        elif isinstance(X, pd.core.frame.DataFrame):
            X = X.to_numpy()
    elif isinstance(X, (np.ndarray, da.core.Array)):
        pass
    # END header handling - - - - -



    if not len(X.shape) == 2:
        raise ValueError(f"'X' must be 2 dimensional")

    assert isinstance(verbose, bool)


    # keep validation ** ** ** ** ** **
    if not isinstance(keep, str):
        raise TypeError(f"'keep' must be a single string")

    _required = ('left', 'right', 'random')
    if sum([_ in keep for _ in _required]) != 1:
        raise ValueError(f"'keep' must be one of {', '.join(_required)}")
    del _required
    # END keep validation ** ** ** ** ** **

    # do_not_drop validation ** ** ** ** ** **

    if callable(do_not_drop):
        if not HEADER:
            pass  # pizza
        elif HEADER:
            pass  # pizza
    else:
        try:
            iter(do_not_drop)
            if isinstance(do_not_drop, (str, dict)):
                raise
        except:
            raise TypeError(f"'do_not_drop' must be a list-like of strings or integers")


        def _validate_dnd_int(do_not_drop):

            if not all(map(lambda x: int(x)==x, do_not_drop)):
                return False

            #     raise TypeError(f"when a header is not passed with the data (as an array), "
            #         f"'do_not_drop' can only contain integers")
            # do_not_drop = sorted(do_not_drop)
            # if min(do_not_drop) < 0:
            #     raise ValueError(f"'do_not_drop' index {min(do_not_drop)} out of range")
            # if max(do_not_drop) >= X.shape[1]:
            #     raise ValueError(f"'do_not_drop' index {max(do_not_drop)} out of range")

            return True

        def _validate_dnd_str(do_not_drop):

            if not all(map(isinstance, do_not_drop, (str for _ in do_not_drop))):
                raise TypeError(f"when a header is passed with the data (as a dataframe), "
                    f"'do_not_drop' can contain strings")
            do_not_drop = sorted(do_not_drop)
            if min(do_not_drop) < 0:
                raise ValueError(f"'do_not_drop' index {min(do_not_drop)} out of range")
            if max(do_not_drop) >= X.shape[1]:
                raise ValueError(f"'do_not_drop' index {max(do_not_drop)} out of range")

            return True



        if not HEADER:

    if not all(map(isinstance, do_not_drop, ))




    # END do_not_drop validation ** ** ** ** ** **




    # ref objects v^v^v^v^v^v^v^v^
    _duplicates: dict[int, list[int]] = {i:[] for i in range(X.shape[1])}

    # END ref objects v^v^v^v^v^v^v^v^

    all_duplicates = []
    for col_idx1 in range(len(X)):

        if col_idx1 in all_duplicates:
            continue

        if verbose and (col_idx1 + 1) % 100 == 0:
            print(f'Comparing column # {col_idx1 + 1} of {X.shape[1]}...')

        for col_idx2 in range(col_idx1, len(X)):

            if col_idx2 in all_duplicates:
                continue

            if np.array_equiv(X[col_idx1], X[col_idx2]):
                _duplicates[col_idx1].append(col_idx2)
                all_duplicates.append(col_idx1)
                all_duplicates.append(col_idx2)


    # ONLY RETAIN INFO FOR COLUMNS THAT ARE DUPLICATE
    _duplicates = {k:v for k, v in _duplicates.items() if len(v) > 0}

    del all_duplicates

    # UNITE DUPLICATES INTO GROUPS
    all_duplicates = []
    GROUPS = []
    for idx1, v1 in deepcopy(_duplicates).items():

        if idx1 in all_duplicates:
            continue

        _GROUP = [idx1] + v1
        for idx2 in v1:

            if idx2 in all_duplicates:
                continue

            _GROUP += _duplicates[idx2]

        __ = list(set(_GROUP))
        GROUPS.append(__)
        all_duplicates.append(__)

    DELETE = []
    for group in GROUPS:
        if keep == 'left':



    X = np.delete(X_encoded_np, col_idx2, axis=0)
    COLUMNS = np.delete(COLUMNS, col_idx2, axis=0)


    # header handling - - - - -
    if isinstance(X, (pd.core.frame.DataFrame, ddf.core.Dataframe, ddf2.DataFrame)):
        if isinstance(X, (ddf.core.Dataframe, ddf2.DataFrame)):
            X = X.from_array(columns=HEADER)
        elif isinstance(X, pd.core.frame.DataFrame):
            X = pd.DataFrame(X, columns=HEADER)
    elif isinstance(X, (np.ndarray, da.core.Array)):
        pass
    # END header handling - - - - -

    return X








