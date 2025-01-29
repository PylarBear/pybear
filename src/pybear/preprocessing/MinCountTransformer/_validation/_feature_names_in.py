# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Optional, Iterable
from typing_extensions import Union

from ._n_features_in import _val_n_features_in



def _val_feature_names_in(
    _feature_names_in: Union[Iterable[str], None],
    _n_features_in: Optional[Union[int, None]]=None
) -> None:


    """
    Validate 'feature_names_in' is None or 1D list-like of strings. If
    'n_features_in' is provided and 'feature_names_in' is not None, the
    length of 'feature_names_in' must equal 'n_features_in'.


    Parameters
    ----------
    _feature_names_in:
        Union[Iterable[str], None] - if MCT was fit on a data
        container that had a header (e.g. pandas dataframe) then this is
        a list-like of those feature names. Otherwise, is None.

    _n_features_in:
        Optional[Union[int, None]], default=None - the number of features
        in the data that was fit.


    Return
    ------
    -
        None

    """

    if _n_features_in is not None:
        _val_n_features_in(_n_features_in)


    err_msg = (
        f"'_feature_names_in' must be None or a 1D list-like of strings "
        f"indicating the feature names of a data-bearing object. if "
        f"list-like and 'n_features_in' is provided, the length must "
        f"equal '_n_features_in'."
    )

    try:
        if _feature_names_in is None:
            raise UnicodeError
        iter(_feature_names_in)
        if isinstance(_feature_names_in, (str, dict)):
            raise Exception
        if not all(map(
            isinstance, _feature_names_in, (str for _ in _feature_names_in)
        )):
            raise MemoryError
    except UnicodeError:
        pass
    except MemoryError:
        raise ValueError(err_msg)
    except:
        raise TypeError(err_msg)

    del err_msg

    if _feature_names_in is not None and _n_features_in is not None:
        if len(_feature_names_in) != _n_features_in:
            raise ValueError(
                f"len(_feature_names_in) ({len(_feature_names_in)}) must "
                f"equal _n_features_in ({_n_features_in})"
            )










