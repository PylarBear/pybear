# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#




from typing import Iterable

import numbers

from ._num_features import num_features
from ._num_samples import num_samples





def check_shape(
    OBJECT,
    min_features: numbers.Integral=1,
    min_samples: numbers.Integral=1,
    allowed_dimensionality: Iterable[numbers.Integral] = (1, 2)
) -> tuple[int, ...]:

    """
    Check the shape of a data-bearing object against user-defined
    criteria.
    OBJECT must have a 'shape' method.
    The number of samples in OBJECT must be greater than or equal to
    min_samples.
    The number of features in OBJECT must be greater than or equal to
    min_features.
    The dimensionality of OBJECT must be one of the allowed values in
    allowed_dimensionality.


    Parameters
    ----------
    OBJECT:
        {array-like} - The data-bearing object for which to get and
        validate the shape. Must have a 'shape' attribute.
    min_features:
        numbers.Integral - The minimum number of features required in
        OBJECT; must be greater than or equal to zero.
    min_samples:
        numbers.Integral - The minimum number of samples required in
        OBJECT; must be greater than or equal to zero.
    allowed_dimensionality:
        Iterable[numbers.Integral] - The allowed dimensionalities of
        OBJECT. All entries must be greater than zero and less than or
        equal to two.


    Return
    ------
    -
        _shape: tuple[int, ...] - the shape of OBJECT.


    """



    # validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    if not hasattr(OBJECT, 'shape'):
        raise TypeError("the passed object must have a 'shape' attribute.")

    assert isinstance(min_features,  numbers.Integral)
    assert min_features >= 0

    assert isinstance(min_samples,  numbers.Integral)
    assert min_samples >= 0

    err_msg = (f"'allowed_dimensionality' must be a vector-like iterable "
        f"of integers greater than zero and less than three.")

    try:
        if isinstance(allowed_dimensionality, numbers.Integral):
            allowed_dimensionality = (allowed_dimensionality, )
        __ = allowed_dimensionality
        iter(__)
        if isinstance(__, (str, dict)):
            raise Exception
        if not all(map(isinstance, __, (numbers.Integral for _ in __))):
            raise Exception
        if not all(map(lambda x: x > 0, __)):
            raise UnicodeError
        if not all(map(lambda x: x <= 2, __)):
            raise UnicodeError
    except UnicodeError:
        raise ValueError(err_msg)
    except:
        raise TypeError(err_msg)
    # END validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    if len(OBJECT.shape) not in allowed_dimensionality:
        raise ValueError(
            f"The dimensionality of the passed object must be in "
            f"{allowed_dimensionality}. Got {len(OBJECT.shape)}."
        )

    _samples = num_samples(OBJECT)
    _features = num_features(OBJECT)

    if _samples < min_samples:
        raise ValueError(
            f"passed object has {_samples} samples, minimum required is "
            f"{min_samples}"
        )

    if _features < min_features:
        raise ValueError(
            f"passed object has {_features} samples, minimum required is "
            f"{min_features}"
        )


    return OBJECT.shape

















