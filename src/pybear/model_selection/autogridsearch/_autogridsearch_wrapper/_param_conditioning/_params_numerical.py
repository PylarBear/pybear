# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from .._type_aliases_num import (
    InNumParamType,
    NumParamType
)

import numbers

import numpy as np



def _cond_numerical_param_value(
    _num_param_value: InNumParamType,
    _total_passes: numbers.Integral,

) -> NumParamType:


    """
    Standardize format.

    Regardless of what is entered for the first round number of points,
    that value is overwritten with the actual length of the first grid.

    COMES IN AS
    (i) a list-like of first-round grid-search points
    (ii) an int or list-like of ints indicating the number of grid points
        for each pass of autogridsearch
    (iii) a string indicating the data type and the search type

    GOES OUT AS
    (i) a list of first-round grid-search points
    (ii) a list of number of grid points for each pass of autogridsearch
    (iii) a string indicating the data type and the search type


    Parameters
    ----------
    _num_param_value:
        InNumParamType - the 'params' dict value for a numerical parameter
        to be conditioned and standardized.


    Returns
    -------
    -
        NumParamType: the conditioned numerical parameter 'params' dict
        value.


    """


    _value = _num_param_value

    _value = list(_value)

    _value[2] = _value[2].lower()

    # standardize first_grid in 0 slot ** * ** * ** * ** * ** * ** * **

    if 'integer' in _value[-1]:
        _value[0] = list(map(int, np.sort(list(_value[0]))))
    elif 'float' in _value[-1]:
        _value[0] = list(map(float, np.sort(list(_value[0]))))

    # END standardize first_grid in 0 slot ** * ** * ** * ** * ** * ** *


    # standardize points part 1 ** * ** * ** * ** * ** * ** * ** * ** * **
    try:
        iter(_value[1])  # IF IS A SINGLE NON-SEQUENCE, CONVERT TO LIST
        _value[1] = list(_value[1])
    except Exception as e:
        _value[1] = [int(_value[1]) for _ in range(_total_passes)]

    # the desired behavior is that if a user enters this [[1,2,3], 1, ...]
    # then the first points is automatically set to len grid, and all
    # passes after just run the single best value: points = [3, 1, 1, ... ]
    # simply overwrite whatever user put in 0 slot for points, without
    # notifying if original entry was erroneous
    _value[-2][0] = len(_value[0])
    # END standardize points part 1 ** * ** * ** * ** * ** * ** * ** * ** *


    return _value





