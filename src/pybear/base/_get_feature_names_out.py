# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Iterable
from typing_extensions import Union

import numpy.typing as npt
import numpy as np

import numbers



def get_feature_names_out(
    _input_features: Union[Iterable[str], None],
    feature_names_in_: Union[npt.NDArray[object], None],
    n_features_in_: numbers.Integral
) -> npt.NDArray[object]:

    """
    Return the feature name vector for the transformed output.

    - If 'input_features' is 'None', then 'feature_names_in_' is
      used as feature names in. If 'feature_names_in_' is not defined,
      then the following input feature names are generated:
      '["x0", "x1", ..., "x(n_features_in_ - 1)"]'.
    - If 'input_features' is an array-like, then 'input_features' must
      match 'feature_names_in_' if 'feature_names_in_' is defined.


    Parameters
    ----------
    _input_features:
        Union[Iterable[str], None] - Input features.
    feature_names_in_:
        Union[npt.NDArray[object], None] - The names of the features as
        seen during fitting. Only available if X is passed to :methods:
        partial_fit or fit as a pandas dataframe that has a header.
    n_features_in_:
        numbers.Integral - The number of features in X.


    Return
    ------
    -
        _feature_names_out: npt.NDArray[object] - The feature names for
        the transformed output.

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
            f"'input_features' must be a vector-like containing strings, or None"
        )


    if feature_names_in_ is not None:
        assert isinstance(feature_names_in_, np.ndarray)
        assert len(feature_names_in_.shape) == 1
        assert feature_names_in_.shape[0] == n_features_in_
        assert all(map(
            isinstance, feature_names_in_, (str for _ in feature_names_in_)
        ))

    assert isinstance(n_features_in_, numbers.Integral)
    assert not isinstance(n_features_in_, bool)

    # END validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    # this code parallels sklearn.utils.validation _check_feature_names_in()
    if _input_features is not None:

        if len(_input_features) != n_features_in_:
            raise ValueError(
                "input_features should have length equal to number of "
                f"features ({n_features_in_}), got {len(_input_features)}"
            )

        if feature_names_in_ is not None:

            if not np.array_equal(_input_features, feature_names_in_):
                raise ValueError(
                    f"_input_features is not equal to feature_names_in_"
                )

        _X_feature_names = _input_features

    elif feature_names_in_ is not None:   # and input_features is None
        _X_feature_names = feature_names_in_

    else:  # feature_names_in_ is None and input_features not provided
        _X_feature_names = [f"x{i}" for i in range(n_features_in_)]

    # END sklearn.utils.validation _check_feature_names_in() parallel ** * ** *


    return np.array(_X_feature_names, dtype=object)

















