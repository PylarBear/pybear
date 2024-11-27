# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from .._type_aliases import KeepType, DataFormatType
from typing import Literal
from typing_extensions import Union
import numpy.typing as npt
import numpy as np
import warnings



def _manage_keep(
    _keep: KeepType,
    _X: DataFormatType,
    constant_columns_: dict[int, any],
    _n_features_in: int,
    _feature_names_in: Union[npt.NDArray[str], None]
) -> Union[Literal['first', 'last', 'random', 'none'], dict[str, any], int]:

    """
    Before going into _make_instructions, process some of the mapping of
    'keep' to a column index and validate against constant_columns_.

    Helps to simplify _make_instructions and makes for easier testing.

    If dict[int, any], just pass through without validation.
    If callable, convert to int and verify is a constant column.
    If a feature name, convert to int and verify is a constant column.
    If a keep literal ('first', 'last', 'random'):
        if there are no constant columns, warn and set keep to 'none'
        otherwise map to a column index.
    If keep literal 'none':
        if there are no constant columns, warn, otherwise pass through.
    If keep is integer, verify is a constant column.


    FROM columns & keep VALIDATION, WE KNOW ** * ** * ** * ** * ** * **

    'feature_names_in_' could be:
        type(None),
        Iterable[str] whose len == X.shape[1]

    'keep' could be
        Literal['first', 'last', 'random', 'none'],
        dict[str, any],
        callable(X),
        int,
        a feature name

    if 'keep' is str in ('first', 'last', 'random', 'none'):
    	if 'feature_names_in_' is not None, keep literal is not in it
    if 'keep' is dict:
    	len == 1
    	key is str
    	warns if 'feature_names_in_' is not None and key is in it
        value cannot be callable, cannot be non-str iterable
    if 'keep' is callable(X):
    	output is int
    	output is not bool
    	output is in range(X.shape[1])
    if 'keep' is number:
    	is int
    	is not bool
    	is in range(X.shape[1])
    if 'keep' is str not in literals:
    	'feature_names_in_' cannot be None
    	'keep' must be in 'feature_names_in_'

    END WHAT WE KNOW FROM columns & keep VALIDATION ** * ** * ** * ** *

    RULES FOR _manage_keep:

    'feature_names_in_' is not changed

    'keep':
    --'first', 'last', 'random'         converted to int, validated--
    --'none'                            passes thru--
    --dict[str, any]                    passes thru--
    --callable(X)                       converted to int, validated--
    --int                               validated--
    --a feature name                    converted to int, validated--

    keep can only leave here as dict[int, any], int, or Literal['none']


    Parameters
    ----------
    _keep:
        Literal['first', 'last', 'random', 'none'], dict[str, any], int,
        str, Callable[[_X], int] - The strategy for handling the constant
        columns. See 'The keep Parameter' section for a lengthy
        explanation of the 'keep' parameter.
    _X:
        {array-like, scipy sparse} of shape (n_samples, n_features) -
        the data that was searched for constant columns.
    constant_columns_:
        dict[int, any] - constant column indices and their values found
        in all partial fits.
    _n_features_in:
        int - number of features in the fitted data before transform.
    _feature_names_in:
        Union[npt.NDArray[str], None] - The names of the features as seen
        during fitting. Only accessible if X is passed to :methods:
        partial_fit or fit as a pandas dataframe that has a header.


    Return
    ------
    -
        __keep:
            Union[dict[int, any], int, Literal['none']] - _keep converted
            to integer for callable, 'first', 'last', 'random', or
            feature name. __keep can only return as an integer, dict,
            or Literal['none']


    """

    # validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    # dont need to validate 'keep' this is the first thing 'keep' sees after
    # _validate in both partial_fit and transform... pizza come back and
    # revise this after inverse_transform is finalized, to account for the
    # final state of that situation.

    assert hasattr(_X, 'shape')

    assert isinstance(_n_features_in, int)
    assert _n_features_in > 0
    assert isinstance(_feature_names_in, (np.ndarray, type(None)))
    if isinstance(_feature_names_in, np.ndarray):
        assert len(_feature_names_in) == _n_features_in

    assert isinstance(constant_columns_, dict)
    if len(constant_columns_):
        assert all(map(
            isinstance, constant_columns_, (int for _ in constant_columns_)
        ))
        assert min(constant_columns_) >= 0
        assert max(constant_columns_) < _n_features_in

    # END validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *



    if isinstance(_keep, dict):
        __keep = _keep
    elif callable(_keep):
        __keep = _keep(_X)
        if __keep not in constant_columns_:
            raise ValueError(
                f"'keep' callable has returned an integer column index ({_keep}) "
                f"that is not a column of constants. \nconstant columns: "
                f"{constant_columns_}"
            )
    elif isinstance(_keep, str) and _feature_names_in is not None and \
            _keep in _feature_names_in:
        # if keep is str, convert to idx
        # validity of keep as feature str (header was passed, keep is in
        # header) should have been validated in _validation > _keep_and_columns
        __keep = int(np.arange(_n_features_in)[_feature_names_in == _keep][0])
        # this is the first place where we can validate whether the _keep
        # feature str is actually a constant column in the data
        if __keep not in constant_columns_:
            raise ValueError(
                f"'keep' was passed as '{_keep}' corresponding to column "
                f"index ({_keep}) which is not a column of constants. "
                f"\nconstant columns: {constant_columns_}"
            )
    elif _keep in ('first', 'last', 'random', 'none'):
        _sorted_constant_column_idxs = sorted(list(constant_columns_))
        if len(_sorted_constant_column_idxs) == 0:
            warnings.warn(
                f"ignoring :param: keep literal '{_keep}', there are no "
                f"constant columns"
            )
            __keep = 'none'
        elif _keep == 'first':
            __keep = int(_sorted_constant_column_idxs[0])
        elif _keep == 'last':
            __keep = int(_sorted_constant_column_idxs[-1])
        elif _keep == 'random':
            __keep = int(np.random.choice(_sorted_constant_column_idxs))
        elif _keep == 'none':
            __keep = 'none'
    elif isinstance(_keep, int):
        # this is the first place where we can validate whether the
        # _keep int is actually a constant column in the data
        __keep = _keep
        if __keep not in constant_columns_:
            raise ValueError(
                f"'keep' was passed as column index ({_keep}) "
                f"which is not a column of constants. \nconstant columns: "
                f"{constant_columns_}"
            )
    else:
        raise AssertionError(f"algorithm failure. invalid 'keep': {_keep}")


    # __keep could be dict[str, any], int, or 'none'
    return __keep












