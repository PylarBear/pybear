# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing_extensions import Union
from ..._type_aliases import GenericSlicerType
from .._type_aliases import (
    DaskSlicerType,
    XDaskWIPType,
    YDaskWIPType
)

from dask import compute
import dask.array as da
import dask.dataframe as ddf



def _fold_splitter(
    train_idxs: Union[GenericSlicerType, DaskSlicerType],
    test_idxs: Union[GenericSlicerType, DaskSlicerType],
    *data_objects: Union[XDaskWIPType, YDaskWIPType]
) -> tuple[tuple[XDaskWIPType, YDaskWIPType, ...]]:


    """
    Split given data objects into train / test pairs using the given
    train and test indices. The train and test indices independently
    slice the given data objects; the entire data object need not be
    consumed in a train / test split and the splits can also possibly
    share indices. Standard indexing rules apply. Returns a tuple whose
    length is equal to the number of data objects passed, holding tuples
    of the train / test splits for the respective data objects.
    train_idxs and test_idxs must be 1D vectors of indices, not booleans.


    Parameters
    ----------
    # pizza fix these type hints!
    train_idxs:
        Iterable[int] - 1D vector of row indices used to slice train sets
        out ouf every given data object.
    test_idxs:
        Iterable[int] - 1D vector of row indices used to slice test sets
        out ouf every given data object.
    *data_objects:
        Union[XDaskWIPType, YDaskWIPType] - The data objects to slice.
        Need not be of equal size, and need not be completely consumed
        in the train / test splits. However, standard indexing rules
        apply when slicing by train_idxs and test_idxs.


    Return
    ------
    -
        SPLITS: tuple[tuple[da.core.Array, da.core.Array], ...] - return
        the train / test splits for the given data objects in the order
        passed in a tuple of tuples, each inner tuple containing a
        train/test pair.

    """


    # helper ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
    def _val_cond_idxs(_name, _idxs):
        """Helper for validating and conditioning slicers."""
        _err_msg = f"'{_name}' must be a 1D vector of integers, not bools."
        _idxs = da.array(_idxs)
        if _idxs.dtype == bool:
            raise TypeError(_err_msg)
        if len(_idxs.shape) != 1:
            raise ValueError(_err_msg)
        del _err_msg
        return _idxs
    # END helper ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    train_idxs = _val_cond_idxs('train_idxs', train_idxs)

    test_idxs = _val_cond_idxs('test_idxs', test_idxs)

    del _val_cond_idxs


    SPLITS = []
    for _data in data_objects:

        if not isinstance(_data, (da.core.Array, ddf.DataFrame, ddf.Series)):
            raise TypeError(f"object of disallowed dtype '{type(_data)}' "
                f"is in fold_splitter()")

        assert len(compute(_data.shape)[0]) in [1, 2], \
            f"data objects must be 1-D or 2-D"

        #  25_04_28 the compute()s need to be here for ddf slicing to work
        if isinstance(_data, da.core.Array):
            _data_train = _data[train_idxs]
            _data_test = _data[test_idxs]
        elif isinstance(_data, ddf.DataFrame):
            _data_train = _data.loc[train_idxs.compute(), :]
            _data_test = _data.loc[test_idxs.compute(), :]
        elif isinstance(_data, ddf.Series):
            _data_train = _data[train_idxs.compute()]
            _data_test = _data[test_idxs.compute()]


        SPLITS.append(tuple((_data_train, _data_test)))


    return tuple(SPLITS)







