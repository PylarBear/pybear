# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Generator, Optional
from ..._type_aliases import (
    XDaskWIPType,
    YDaskWIPType,
    DaskKFoldType
)

import time

import dask.array as da
from dask_ml.model_selection import KFold as dask_KFold



def _get_kfold(
    _X: XDaskWIPType,
    _n_splits: int,
    _iid: bool,
    _verbose: int,
    _y: Optional[YDaskWIPType] = None
) -> Generator[DaskKFoldType, None, None]:

    """
    Use dask_ml KFold to get train / test splits when cv is passed as an
    integer. KFold uses the number of rows in _X and _n_splits to
    determine the indices in each train / test split.
    _X must be a 2D dask.array.core.Array. y is optional in dask_ml KFold.
    If passed, it must be a 1D da.core.Array vector and the number of
    rows in _X and _y must be equal. As of 24_06_27, only dask arrays can
    be passed to dask_ml.KFold (not np, pd.DF, nor dask.DF). See
    dask_kfold_input_test in functional_tests folder for details.

    *** IMPORTANT!!!
    This function can be called multiple times within a single param grid
    permutation, first to fit, again to get test score, then again if
    return_train_score. Therefore, it must return the same indices for
    each call. The only things that should cause indices to be different
    are n_splits and the number of rows in _X. Since this is dask KFold,
    there is the wildcard of the 'iid' setting. If iid is False --
    meaning the data is known to have some non-random grouping along
    axis 0 -- via the 'shuffle' argument KFold will generate indices that
    sample across chunks to randomize the data in the splits. In that
    case, fix the random_state parameter to make selection repeatable.
    If iid is True, shuffle is False, random_state can be None, and the
    splits should be repeatable.


    Parameters
    ----------
    _X:
        dask.array.core.Array[Union[int,float]] - The data to be split.
        Must be 2D dask.array.core.Array.
    _y:
        dask.array.core.Array[int] - optional - The target the data is
        being fit against, to be split in the same way as the data. Must
        be 1D dask.array.core.Array.
    _iid:
        bool - True, the examples in X are distributed randomly; False,
        there is some kind of non-random ordering of the examples in X.
    _n_splits:
        int - the number of splits to produce; the number of split pairs
        yielded by the returned generator object.
    _verbose:
        int - a number from 0 to 10 indicating the amount of information
        to display to screen during the grid search trials. 0 means no
        output, 10 means full output.


    Return
    ------
    -
        KFOLD:
            Generator[tuple[da.core.Array[int], da.core.Array[int]] - A
            generator object yielding pairs of train test indices as
            da.core.Array[int].

    """


    err_msg = (f"_X ({type(_X)}) and _y ({type(_y)}) must both be dask "
               f"arrays.")

    if not isinstance(_X, da.core.Array):
        raise TypeError(err_msg)

    if not isinstance(_y, (da.core.Array, type(None))):
        raise TypeError(err_msg)

    del err_msg

    assert isinstance(_n_splits, int)
    assert _n_splits > 1
    assert isinstance(_iid, bool)
    try:
        float(_verbose)
    except:
        raise AssertionError(f"'_verbose' must be an int, float, or bool")
    assert _verbose >= 0


    split_t0 = time.perf_counter()
    # KFold keeps the same chunks ax X
    KFOLD = dask_KFold(
        n_splits=_n_splits,
        shuffle=not _iid,
        random_state=7 if not _iid else None,
        # shuffle is on if not iid. must use random_state so that later
        # calls for train score get same splits.
    ).split(_X, _y)

    if _verbose >= 5:
        print(f'split time = {time.perf_counter() - split_t0: ,.3g} s')

    del split_t0

    return KFOLD



























