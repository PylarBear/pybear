# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from .._type_aliases_num import InNumParamType

import numbers

import numpy as np



def _val_numerical_param_value(
    _num_param_key: str,
    _num_param_value: InNumParamType,
    _total_passes: numbers.Integral
) -> None:


    """
    Validate _num_param_value.

    Integer spaces must be >= 1, float spaces must be >= 0.

    For fixed float or integer, the 'points' values must be either the
    length of the first search grid or 1 then 1 thereafter. E.g., [3,3,3,1,1].

    For any case where 1 is entered as points, all points thereafter must
    be 1.

    Logspace intervals must be integer >= 1. Meaning you cannot have
    powers that are like 10^0.5, 10^0.6, ..., but you can have powers
    like 10^2, 10^4, ...

    validate numerical_params' dict value is a list-like that contains:
    (i) a list-like of first-round grid-search values
    (ii) an int or list-like of ints indicating the number of grid points
        for each pass of autogridsearch
    (iii) a string indicating the data type and the search type


    Parameters
    ----------
    _num_param_key:
        str - the estimator parameter name to be grid-searched.
    _num_param_value:
        InNumParamType - the list-like of instructions for the multi-pass
        grid search of this numerical parameter.
    _total_passes:
        numbers.Integral - the total number of rounds of gridsearch
        entered by the user at init.


    Returns
    -------
    -
        None


    """


    _base_err_msg = f"numerical param '{str(_num_param_key)}' in :param: 'params' "


    # validate contains [first_grid] in 0 slot ** * ** * ** * ** * ** *
    _err_msg = (f"{_base_err_msg} -- "
        f"\nfirst position of the value must be a non-empty list-like that "
        f"\ncontains the first pass grid-search values. "
    )

    try:
        if any(map(isinstance, _num_param_value[0], (bool for _ in _num_param_value[0]))):
            raise Exception
        list(map(float, _num_param_value[0]))
    except:
        raise TypeError(_err_msg)

    del _err_msg

    if 'integer' in _num_param_value[2]:

        if not all(int(i) == i for i in _num_param_value[0]):
            raise ValueError(
                f"{_base_err_msg} -- \nwhen numerical is integer (soft, "
                f"hard, or fixed), \nall search values must be integers. "
                f"\ngrid = {_num_param_value[0]}"
            )

        if _num_param_value[2] in ['hard_integer', 'soft_integer'] and min(_num_param_value[0]) < 1:
            raise ValueError(
                f"{_base_err_msg} -- \nwhen numerical is hard/soft integer, "
                f"\nall search values must be >= 1. \ngrid = {_num_param_value[0]}"
            )

    elif 'float' in _num_param_value[2]:

        if _num_param_value[2] in ['hard_float', 'soft_float'] \
                and (np.array(list(_num_param_value[0])) < 0).any():
            raise ValueError(
                f"{_base_err_msg} -- \nwhen numerical is hard/soft float, "
                f"\nall search values must be >= 0. \ngrid = {_num_param_value[0]}")

    else:
        raise Exception

    # LOGSPACE
    if 'fixed' not in _num_param_value[2] and len(_num_param_value[0]) >= 3 and 0 not in _num_param_value[0]:
        log_grid = np.log10(list(_num_param_value[0]))
        log_gaps = log_grid[1:] - log_grid[:-1]
        _unq_log_gap = np.unique(np.round(log_gaps, 14))

        if len(_unq_log_gap) == 1:  # else is not a logspace
            # CURRENTLY ONLY HANDLES LOGSPACE BASE 10 OR GREATER
            if _unq_log_gap[0] < 1:
                raise ValueError(
                    f"{_base_err_msg} -- \nonly handles logspaces with "
                    f"base 10 or greater"
                )

            # 24_05_14_07_53_00 ENFORCING INTEGER FOR LOGSPACE MAKES MANAGING
            # GAPS IN DRILL SECTION A LOT EASIER
            if int(_unq_log_gap[0]) != _unq_log_gap[0]:
                raise ValueError(
                    f'{_base_err_msg} -- \nlogspaces must have integer intervals'
                )

        del log_grid, log_gaps, _unq_log_gap

    # END validate contains [first_grid] in 0 slot ** * ** * ** * ** *

    # validate points part 1 ** * ** * ** * ** * ** * ** * ** * ** * **
    err_msg = (
        f"{_base_err_msg} -- \n'points' must be "
        f"\n(i) a non-bool integer >= 1 or "
        f"\n(ii) a list-type of non-bool integers >=1 with len==passes"
        f"\ngot {_num_param_value[1]}, total_passes={_total_passes}"
    )

    # this is a helper only for easier validation! this is not returned
    if isinstance(_num_param_value[1], numbers.Number):
        _helper_list = [_num_param_value[1] for _ in range(_total_passes)]
    else:
        _helper_list = _num_param_value[1]

    try:
        iter(_helper_list)
        if isinstance(_helper_list, (dict, str)):
            raise Exception
        # NUMBER OF POINTS IN points MUST MATCH NUMBER OF PASSES
        if len(_helper_list) != _total_passes:
            raise UnicodeError
        # IF A NON-NUMERIC IS IN POINTS
        map(float, _helper_list)
        # IF A BOOL IS IN POINTS
        if any(map(isinstance, _helper_list, (bool for _ in _helper_list))):
            raise Exception
        # IF A FLOAT IS IN points
        if any(int(i) != i for i in map(float, _helper_list)):
            raise Exception
        # IF ANY INT IN points IS LESS THAN 1
        if min(_helper_list) < 1:
            raise UnicodeError
    except UnicodeError:
        raise ValueError(err_msg)
    except Exception as e:
        raise TypeError(err_msg)

    del err_msg

    _helper_list = list(map(int, _helper_list))

    # IF NUMBER OF POINTS IS EVER SET TO 1, ALL SUBSEQUENT POINTS MUST BE 1
    for idx, points in enumerate(_helper_list[:-1]):
        if points == 1 and _helper_list[idx + 1] > 1:
            raise ValueError(
                f"{_base_err_msg} -- \nonce number of points is set to 1, all "
                f"subsequent points must be 1. \ngot {_num_param_value[1]}"
            )

    if 'soft' in _num_param_value[2] and 2 in _helper_list:
        raise ValueError(
            f'{_base_err_msg} -- \nGrids of size 2 are not allowed for '
            f'"soft" numerical params'
        )

    # the desired behavior is that if a user enters this [[1,2,3], 1, ...]
    # then the first points is automatically set to len grid, and all
    # passes after just run the single best value: points = [3, 1, 1, ... ]
    # simply overwrite whatever user put in 0 slot for points, without
    # notifying if original entry was erroneous

    # fixed points in [1 or len(first grid)] (the first points will be
    # automatically set to len(_num_param_value[0]) by conditioning, so only check
    # the values in [1:]
    if 'fixed' in _num_param_value[2]:
        if any(_points not in [1, len(_num_param_value[0])] for _points in _helper_list[1:]):
            raise ValueError(
                f"{_base_err_msg} -- \nif fixed int/float, number of points "
                f"must be len(first grid) or 1 \npoints = {_num_param_value[1]}"
            )

    del _helper_list

    # END validate points part 1 ** * ** * ** * ** * ** * ** * ** * ** *









