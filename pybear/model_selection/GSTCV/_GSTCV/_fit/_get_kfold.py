# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


from model_selection.GSTCV._type_aliases import XSKWIPType, YSKWIPType

import time
from typing import Generator
import numpy as np
import numpy.typing as npt
from sklearn.model_selection import StratifiedKFold


def _get_kfold(
        _X: XSKWIPType,
        _y: YSKWIPType,
        _n_splits: int,
        _verbose: int
    ) -> Generator[tuple[npt.NDArray[int], npt.NDArray[int]], None, None]:

    """
    Use sklearn StratifiedKFold to get train / test splits when cv is
    passed as an integer. StratifiedKFold uses the number of rows in
    _X and _y, _n_splits, and the distribution of values in _y to
    determine the indices in each train / test split. _X and _y must both
    be ndarray. The number of rows in _X and _y must be equal.

    *** IMPORTANT!!!
    This function can be called multiple times within a single param grid
    permutation, first to fit, again to get test score, then again if
    return_train_score. Therefore, it must return the same indices for
    each call. The only things that should cause indices to be different
    are n_splits and the number of rows in _X. Since this is stratified
    KFold examples are pulled based on the distribution of y. But the
    selection should be repeatable if shuffle is set to False.
    random_state does not matter when shuffle is False.

    Parameters
    ----------
    _X:
        NDArray[Union[int,float]] - The data to be split. Must be 2D ndarray.
    _y:
        NDArray[int] - The target the data is being fit against, to be
        split in the same way as the data. Must be 1D ndarray.
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
            Generator[tuple[NDArray[int], NDArray[int]] - A generator
            object yielding pairs of train test indices as NDArray[int].

    """










    err_msg = (f"_X ({type(_X)}) and _y ({type(_y)}) must both be numpy "
               f"arrays.")

    if not isinstance(_X, np.ndarray):
        raise TypeError(err_msg)

    if not isinstance(_y, (np.ndarray, type(None))):
        raise TypeError(err_msg)

    del err_msg

    assert isinstance(_n_splits, int)
    assert _n_splits > 1
    try:
        float(_verbose)
    except:
        raise AssertionError(f"'_verbose' must be an int, float, or bool")
    assert _verbose >= 0


    split_t0 = time.perf_counter()

    KFOLD = StratifiedKFold(
        n_splits=_n_splits,
        shuffle=False,
        random_state=None
    ).split(_X, _y)




    if _verbose >= 5:
        print(f'split time = {time.perf_counter() - split_t0: ,.3g} s')

    del split_t0

    return KFOLD



























