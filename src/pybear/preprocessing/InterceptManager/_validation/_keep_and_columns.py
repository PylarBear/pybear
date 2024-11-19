# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from .._type_aliases import DataType, KeepType
from typing import Iterable
from typing_extensions import Union
import warnings




def _val_keep_and_columns(
    _keep:KeepType,
    _columns:Union[Iterable[str], None],
    _X: DataType
) -> None:

    # pizza do u want to have any limitations on what 'keep' dict value can be, currently any


    """

    Validate columns - must be None or Iterable[str] with len==X.shape[1]

    Validate keep - must be any of:
        Literal['first', 'last', 'random', 'none'],
        dict[str, any],
        int,
        str,
        callable on X, and return an integer (column index)


    Parameters
    ----------
    _keep:
        Literal['first', 'last', 'random', 'none'], dict[str, any], int, str, Callable[[DataType], int]
        default = 'last' -
        The strategy for keeping a single representative from a set of
        identical columns. 'first' retains the column left-most in the
        data; 'last' keeps the column right-most in the data; 'random'
        keeps a single randomly-selected column from the set of
        duplicates.
    _columns:
        Union[Iterable[str], None] - feature names of X, only available
        if X was passed as a pandas dataframe.
    _X:
        PizzaDataType - of shape (n_samples, n_features). The data to be
        searched for constant columns.


    Return
    ------
    -
        None


    """





    # validate types, shapes, of _columns ** * ** * ** * ** * ** * ** * **

    if _columns is None:
        pass
    else:
        err_msg = (
            f"If passed, '_columns' must be a vector of strings whose "
            f"length is equal to the number of features in the data."
        )

        try:
            iter(_columns)
            assert len(_columns) == _X.shape[1]
            assert all(map(isinstance, _columns, (str for _ in _columns)))
        except:
            raise ValueError(err_msg)

    # END validate types, shapes, of _columns ** * ** * ** * ** * ** * **



    _err_msg = (
        f"'keep' must be one of: "
        f"\nLiteral['first', 'last', 'random', 'none'], "
        f"\ndict[str, any], "
        f"\nint (column index), "
        f"\nstr (column header, if X was passed as pandas dataframe), "
        f"\ncallable on X, returning an integer indicating column index"
    )

    # validate keep as dict ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
    if isinstance(_keep, dict):
        _name = list(_keep.keys())[0]
        # must be one entry and key must be str
        if len(_keep) != 1 or not isinstance(_name, str):
            raise ValueError(err_msg)
        # if keep conflicts with existing column name, will overwrite
        if _columns is not None and _name in _columns:
            warnings.warn(
                f"'keep' column name '{_name}' is already in the data, "
                f"the existing data in that column will be overwritten."
            )
        # pizza, not validating dict value at the moment, make a decision on this

        del _name

        return # <====================================================
    # END validate keep as dict ** * ** * ** * ** * ** * ** * ** * ** * ** *

    # validate keep as int ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
    try:
        float(_keep)
        if _keep != int(_keep):
            raise UnicodeError
        if isinstance(_keep, bool):
            raise UnicodeError
        _keep = int(_keep)
        if _keep not in range(_X.shape[1]):
            raise UnicodeError
        return # <====================================================
    except UnicodeError:
        raise ValueError(err_msg)
    except:
        pass
    # END validate keep as int ** * ** * ** * ** * ** * ** * ** * ** * ** *

    # validate keep as callable ** * ** * ** * ** * ** * ** * ** * ** * ** *
    # returns an integer in range of num X features
    if callable(_keep):
        _test_keep = _keep(_X)
        try:
            float(_test_keep)
            if _test_keep != int(_test_keep):
                raise Exception
            if isinstance(_keep, bool):
                raise Exception
            _test_keep = int(_test_keep)
            if _test_keep not in range(_X.shape[1]):
                raise Exception
            return  # <====================================================
        except:
            ValueError(err_msg)

    # END validate keep as callable ** * ** * ** * ** * ** * ** * ** * ** *

    # validate keep as str ** * ** * ** * ** * ** * ** * ** * ** * ** * **
    if isinstance(_keep, str):
        # could be one of the literals or a column header
        _keep = _keep.lower()
        if _keep in ('first', 'last', 'random', 'none'):
            # this is when it is bad for 'keep' to be in _columns
            if _columns is not None and _keep in _columns:
                # if is one of the literals
                # pizza make a decision on this, case sensitivity and such!
                raise ValueError(
                    f"\nthere is a conflict with one of the feature names and "
                    f"the literals for :param: keep. Column '{_keep}' conflicts with keep literal."
                    f"\nkeep literals: 'first', 'last', 'random', 'none' pizza what about case sensitive?"
                    f"\nplease change the column name or use a different keep literal."
                    f"\n"
                )
        else:
            # is str, but is not one of the literals, then must be a column name
            if _columns is None:
                raise ValueError(f":param: keep '{_keep}' is string but header was not passed")
            elif _keep not in _columns:
                raise ValueError(f":param: keep '{_keep}' is ")



        return  # <====================================================
    # END validate keep as str ** * ** * ** * ** * ** * ** * ** * ** * ** *




    raise ValueError(_err_msg)





