# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#




from .._type_aliases import FeatureNameCombinerType

import numpy.typing as npt
import numpy as np



def _gfno_poly(
    feature_names_in_: npt.NDArray[object],
    _active_combos: tuple[tuple[int, ...], ...],
    _feature_name_combiner: FeatureNameCombinerType
) -> npt.NDArray[object]:

    """

    sklearn lingo ----> Get output feature names for transformation.

    #   _poly_feature_names must be sorted asc len, then asc on idxs. if
    #   _active_combos is sorted correctly,
    #   then this is sorted correctly at construction.
    #   _active_combos being sorted correctly depends on self._combos
    #   being sorted correctly

    Return the feature name vector for the transformed output. Construct
    the polynomial feature names based on :param: feature_name_combiner.
    If :param: min_degree == 1, the feature names of X are prepended to
    the polynomial feature names.

        input_features : array-like of str or None, default=None
            Input features.

            - If `input_features` is `None`, then `feature_names_in_` is
              used as feature names in. If `feature_names_in_` is not defined,
              then the following input feature names are generated:
              `["x0", "x1", ..., "x(n_features_in - 1)"]`.
            - If `input_features` is an array-like, then `input_features` must
              match `feature_names_in_` if `feature_names_in_` is defined.


    Parameters
    ----------
    feature_names_in_:
        npt.NDArray[object] -
    _active_combos:
        tuple[tuple[int, ...], ...] -
    _feature_name_combiner:
        # pizza finalize this and put it in the main module.
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
        _poly_feature_names: npt.NDArray[object] - The feature names for
        the polynomial expansion.


    """

    # validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    assert isinstance(feature_names_in_, np.ndarray)
    assert feature_names_in_.dtype == object
    assert len(feature_names_in_.shape) == 1
    assert all(map(isinstance, feature_names_in_, (str for _ in feature_names_in_)))

    assert isinstance(_active_combos, tuple)
    assert len(_active_combos), f"empty _active_combos has gotten into _get_feature_names_out"
    for _tuple in _active_combos:
        assert isinstance(_tuple, tuple)
        assert all(map(isinstance, _tuple, (int for _ in _tuple)))

    # max idx in _active_combos must be in range of feature_names_in_


    assert callable(_feature_name_combiner) or _feature_name_combiner in ['as_feature_names', 'as_indices']


    # END validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    if _feature_name_combiner == "as_indices":
        _poly_feature_names = list(map(str, _active_combos))

    elif _feature_name_combiner == "as_feature_names":
        _poly_feature_names = []
        for _combo_idx, _combo in enumerate(_active_combos):

            # scan over the combo, get the powers by counting the number of
            # occurrences of X column indices
            _idx_ct_dict = {k: 0 for k in _combo}  # only gets unique idxs
            for _X_idx in _combo:
                _idx_ct_dict[_X_idx] += 1

            _poly_feature_name = ''

            for _combo_idx, _X_idx in enumerate(_combo):
                _poly_feature_name += f"{feature_names_in_[_X_idx]}^{_idx_ct_dict[_X_idx]}"
                if _combo_idx < (len(_combo) - 1):
                    _poly_feature_name += "_"

            _poly_feature_names.append(_poly_feature_name)



    elif callable(_feature_name_combiner):
        _poly_feature_names = []
        for _combo in _active_combos:
            _poly_feature_name = _feature_name_combiner(feature_names_in_, _combo)
            if not isinstance(_poly_feature_name, str):
                raise TypeError(
                    "When `feature_name_combiner` is a callable, it should return a "
                    f"Python string. Got {type(_poly_feature_name)} instead."
                )

            if _poly_feature_name in _poly_feature_names:
                raise ValueError(
                    "The `feature_name_combiner` callable has returned the same "
                    f"feature name twice. It must return unique names for each feature."
                )

            _poly_feature_names.append(_poly_feature_name)

    else:
        raise Exception()


    return np.array(_poly_feature_names, dtype=object)




















