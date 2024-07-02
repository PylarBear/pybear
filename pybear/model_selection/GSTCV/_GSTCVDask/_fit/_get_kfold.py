# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


from model_selection.GSTCV._type_aliases import XDaskWIPType, YDaskWIPType

import time
from typing import Generator
import dask.array as da
from dask_ml.model_selection import KFold as dask_KFold


def _get_kfold(
        _X: XDaskWIPType,
        _n_splits: int,
        _iid: bool,
        _verbose: int,
        _y: YDaskWIPType = None
    ) -> Generator[da.core.Array, None, None]:


    # X, y must both be da.array
    # AS OF 24_06_27_09_08_00 ONLY DASK ARRAYS CAN BE PASSED TO
    # dask_KFOLD (NOT np, pd.DF, dask.DF)
    # see dask_kfold_input_test in functional_tests folder

    # important!!! this function can be called multiple times within a
    # single param grid permutation, first to fit and get test score,
    # then again if return_train_score. Therefore, it must return the
    # same indices for each call. The only things that should cause
    # indices to be different are n_splits and the number of rows in X.
    # Shuffle is on if iid is False, therefore random_state state must
    # be set to a constant.

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

    KFOLD = dask_KFold(
        n_splits=_n_splits,
        shuffle=not _iid,
        random_state=7 if not _iid else None,
        # shuffle is on if not iid. must use random_state so that later
        # calls for train score get same splits.
        # AS TRAIN
    ).split(_X, _y)

    if _verbose >= 5:
        print(f'split time = {time.perf_counter() - split_t0: ,.3g} s')

    del split_t0

    return KFOLD



























