# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Iterable
from typing_extensions import Union
from .._type_aliases import FeatureNameCombinerType

import numpy.typing as npt
import numpy as np


# pizza this is on the block for deletion



def _get_feature_names_out(
    _input_features: Union[Iterable[str], None],
    feature_names_in_: Union[npt.NDArray[str], None],
    _min_degree: int,
    _active_combos: tuple[tuple[int, ...], ...],
    n_features_in_: tuple[int, ...],
    _feature_name_combiner: FeatureNameCombinerType
) -> npt.NDArray[object]:

    """

    sklearn lingo ----> Get output feature names for transformation.


    Return the feature name vector for the transformed output. Construct
    the polynomial feature names based on :param: feature_name_combiner.
    If :param: min_degree == 1, the feature names of X are prepended to
    the polynomial feature names.

        input_features : array-like of str or None, default=None
            Input features.

            - If `input_features` is `None`, then `feature_names_in_` is
              used as feature names in. If `feature_names_in_` is not defined,
              then the following input feature names are generated:
              `["x0", "x1", ..., "x(n_features_in_ - 1)"]`.
            - If `input_features` is an array-like, then `input_features` must
              match `feature_names_in_` if `feature_names_in_` is defined.


    Parameters
    ----------
    _input_features:
        Union[Iterable[str], None] -
    feature_names_in_:
        Union[npt.NDArray[str], None] -
    _min_degree:
        int -
    _active_combos:
        tuple[tuple[int, ...], ...] -
    n_features_in_:
        tuple[int, ...] - The shape of the data passed to undergo polynomial expansion.
    _feature_name_combiner:
        # pizza finalized this and put it in the main module.
        Union[
            Callable[[Iterable[str], tuple[int, ...]], str],
            Literal['as_feature_names', 'as_indices']
        ] - Define how to name the polynomial feature names.
        Can be a user-defined callable, literal 'as_feature_names', literal 'as_indices'.
        Callable:
        User-defined function for mapping combo tuples of integers
        to polynomial feature names. Must take in two arguments.
        1) :attr: feature_names_in_
        2) a tuple of integers of variable length (min length is :param: min_degree, max length is
        :param: degree) Must return a string.
        Literal 'as_feature_names':
        Build polynomial feature names from the feature names of X. For example, if the
        feature names of X are ['x0', 'x1', ..., 'xn'] and the polynomial
        tuple is (2, 2, 4), then the default polynomial feature name is
        'x2^2_x4'.
        Literal 'as_indices':
        Build polynomial feature names from the indices comprising the polynomial.
        For example, if the polynomial tuple that constructed the feature
        is tuple((2, 2, 4)), then the polynomial feature name is tuple((2,2,4)).


    Return
    ------
    -
        _feature_names_out: npt.NDArray[object] - The feature names for
        the polynomial expansion.


    """

    # validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * 

    try:
        if isinstance(_input_features, type(None)):
            raise UnicodeError
        iter(_input_features)
        if isinstance(_input_features, (str, dict)):
            raise Exception
        if not all(map(
                isinstance, _input_features, (str for _ in _input_features)
        )):
            raise Exception
    except UnicodeError:
        pass
    except:
        raise ValueError(
            f"'_input_features' must be a vector-like containing strings, or None"
        )


    if feature_names_in_ is not None:
        assert isinstance(feature_names_in_, np.ndarray)
        assert len(feature_names_in_.shape) == 1
        assert feature_names_in_.shape[0] == n_features_in_[1]
        assert all(map(isinstance, feature_names_in_, (str for _ in feature_names_in_)))

    assert isinstance(_min_degree, int)
    assert _min_degree >= 1

    assert isinstance(_active_combos, tuple)
    assert len(_active_combos), f"empty _active_combos has gotten into _get_feature_names_out"
    for _tuple in _active_combos:
        assert isinstance(_tuple, tuple)
        assert all(map(isinstance, _tuple, (int for _ in _tuple)))

    assert isinstance(n_features_in_, int)

    assert callable(_feature_name_combiner) or _feature_name_combiner in ['as_feature_names', 'as_indices']


    # END validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *  




    return np.array(poly_feature_names, dtype=object)













