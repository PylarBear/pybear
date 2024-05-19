# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

import numpy as np



def _numerical_param_value(
                            _numerical_param_key:str,
                            _numerical_param_value,
                            total_passes:int
    ) -> list:

    """
    Validate _numerical_param_value --- standardize format

    COMES IN AS
    [
        list [grid_value1, grid_value2, etc.],
        list [number_of_points1, number_of_points2, etc.] or integer > 0,
        str 'data type and search type'
    ]

    -- or --

    [
        str 'linspace' or 'logspace',
        float or int space interval start value,
        float or int space interval end value,
        str [number_of_points1, number_of_points2, etc.] or integer > 0,
        str data type and search type
    ]

    <end> COMES IN AS


    validate numerical_params' dict value is a list-like that contains
    either:
    (i) a list-like of first-round grid-search points
    (ii) an int or list-like of number of grid points for each pass of autogridsearch
    (iii) a string indicating the data type and the search type

    -- or --

    (i) 'linspace' or 'logspace'
    (ii) numerical lower bound of the space
    (iii) numerical upper bound of the space
    (iv) an int or list-like of number of grid points for each pass of autogridsearch
    (v) a string indicating the data type and the search type


    GOES OUT AS
    (i) a list-like of first-round grid-search points
    (ii) a list-like of number of grid points for each pass of autogridsearch
    (iii) a string indicating the data type and the search type


    """


    _ = _numerical_param_key
    __ = _numerical_param_value


    if not isinstance(_, str):
        raise TypeError(f"_numerical_param_key must be a string")


    # validate container object ** * ** * ** * ** * ** * ** * ** * ** * ** * **
    err_msg = (f"param {_} -- numerical_param_value must be a list-like of len 3 "
        f"or 5 --- \n[[*first_grid_values], [*number_of_points_for_each_grid], "
        f"soft/hard/fixed_int/float] - or - \n[search space type, start value, "
        f"end value, [*number of points for each grid], soft/hard/fixed_int/float]")


    try:
        iter(__)
    except:
        raise TypeError(err_msg)

    if isinstance(__, (set, dict, str)):
        raise TypeError(err_msg)

    __ = list(__)

    if len(__) not in [3, 5]:
        raise ValueError(err_msg)

    del err_msg
    # END validate container object ** * ** * ** * ** * ** * ** * ** * ** * **


    # validate soft/hard/fixed_int/float PART 1 ** * ** * ** * ** * ** * ** *

    try:
        __[-1] = __[-1].lower()
    except:
        raise TypeError(f"{_} -- last position must be a string indicating "
                        f"soft/hard/fixed_int/float")

    allowed = [
               "hard_integer", "soft_integer", "fixed_integer", "hard_float",
               "soft_float", "fixed_float"
    ]

    if not __[-1] in allowed:
        raise ValueError(f'{_} -- type must be in {", ".join(allowed)}')

    del allowed

    # END validate soft/hard/fixed_int/float PART 1 ** * ** * ** * ** * ** * **

    # VALIDATE POINTS ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **

    err_msg = TypeError(f'{_} -- "points" must be (i) a non-bool integer >= 1 or '
        f'(ii) a list-type of non-bool integers >=1  with len==passes'
        f'\n\npoints:\n{__[-2]}'
    )

    try:
        iter(__[-2])  # IF IS A SINGLE NON-ITERABLE, CONVERT TO LIST
        if isinstance(__[-2], (set, dict, str)):
            raise UnicodeError
    except TypeError:
        __[-2] = [__[-2] for _ in range(total_passes)]
    except UnicodeError:
        raise TypeError(err_msg)
    except Exception as e:
        raise Exception(f"'points' iterable check failed for uncontrolled "
                        f"reason --- {e}")


    # NUMBER OF POINTS IN points MUST MATCH NUMBER OF PASSES
    if len(__[-2]) != total_passes:
        raise ValueError(err_msg)

    # IF A NON-NUMERIC IS IN POINTS
    try:
        if any(map(isinstance, __[-2], (bool for _ in __[-2]))):
            raise Exception
        _float_test = list(map(float, __[-2]))
    except:
        raise TypeError(err_msg)



    # IF A FLOAT IS IN points
    if not all([int(i) == i for i in _float_test]):
        raise TypeError(err_msg)

    del _float_test

    __[-2] = list(map(int, __[-2]))

    # IF ANY NUMBER IN points IS LESS THAN 1
    if min(__[-2]) < 1:
        raise ValueError(err_msg)

    if 'soft' in __[-1] and 2 in __[-2]:
        raise ValueError(f'{_} -- Grids of size 2 are not allowed for "soft" '
                        f'numerical params')

    # IF NUMBER OF POINTS IS EVER SET TO 1, ALL SUBSEQUENT POINTS MUST BE 1
    if len(__[-2]) > 1:
        for idx in range(len(__[-2][:-1])):
            if __[-2][idx] == 1 and __[-2][idx + 1] > 1:
                raise ValueError(f"{_} -- once number of points is set to 1, all "
                     f"subsequent points must be 1 \n\npoints={__[-2]}")


    # IF FIXED int/float, NUMBER OF POINTS MUST BE SAME AS ORIGINAL len(GRID), OR 1
    if 'fixed' in __[-2]:
        for _point in __[-2]:
            if _point not in [1, __[-2][0]]:
                raise ValueError(f"{_} -- if fixed int/float, number of points "
                    f"must be same as first grid or 1 \n\npoints = {__[-2]}")

    del err_msg
    # END VALIDATE POINTS ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **


    # validate first grid ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * **

    if len(__) == 5:
        # validate list contains 'linspace/logspace', start_value, end_value ##

        err_msg = (f'{_} -- for numerical_params "5" format, first position must '
                f'be a string: "linspace" or "logspace"')

        try:
            __[0] = __[0].lower()
        except:
            raise TypeError(err_msg)

        if not __[0] in ['linspace', 'logspace']:
            raise ValueError(err_msg)

        del err_msg

        # start_value / end_value MUST BE NUMERIC
        err_msg = f"{_} -- for '5' format, start_value & end_value must be numeric"
        for idx in [1, 2]:
            try:
                if isinstance(__[idx], bool):
                    raise Exception
                np.float64(__[idx])
                if 'integer' in __[-1] and (int(__[idx]) != __[idx]):
                    raise Exception
            except:
                raise TypeError(err_msg)

        del err_msg

        # PIZZA Y THIS MUST BE INT??
        # 24_05_14_07_53_00 MANAGING GAPS IN DRILL SECTION IS A LOT EASIER
        if __[0] == 'logspace':
            for idx, posn in enumerate(['start', 'end'], 1):
                if not 'int' in str(type(__[idx])).lower():
                    raise TypeError(f'{_}: {posn}_value ({__[idx]}) must be an '
                                    f'integer for logspace')
        # END validate list contains 'linspace/logspace', start_value, end_value

        if __[0] == 'logspace':
            _grid = np.sort(np.logspace(__[1], __[2], __[3][0])).tolist()
        elif __[0] == 'linspace':
            _grid = np.sort(np.linspace(__[1], __[2], __[3][0])).tolist()

        if 'integer' in __[-1]:
            _grid = list(map(int, _grid))

        __ = [_grid, __[-2], __[-1]]
        del _grid

    # LEN MUST BE 3

    # validate list contains [first_grid], [number_of_points] in [0,1] slots

    err_msg = f"{_} -- for '3' format, first element of list must be a list-type"
    try:
        iter(__[0])
    except:
        raise TypeError(err_msg)

    if isinstance(__[0], (set, dict, str)):
        raise TypeError(err_msg)

    if len(__[0]) != __[1][0]:
        raise ValueError(f"{_} -- first number_of_points must match length of "
                         f"first grid")

    try:
        if any(map(isinstance, __[0], (bool for _ in __[0]))):
            raise Exception
        list(map(float, __[0]))
    except:
        raise TypeError(f"{_} -- search values must be numeric")

    __[0] = np.sort(__[0]).tolist()

    # END validate list contains [first_grid], [number_of_points] in [0,1] slots


    # validate soft/hard/fixed_int/float PART 2 ** * ** * ** * ** * ** * ** *
    if 'integer' in __[-1]:

        if not all([int(i) == i for i in __[0]]):
            raise TypeError(f"{_} -- when numerical is integer (soft, hard, or "
                f"fixed), all search values must be integers: \n\ngrid = {__[0]}")
        if __[-1] == 'soft_integer' and (np.array(__[0]) < 1).any():
            raise ValueError(f"{_}: when numerical is soft integer, all search "
                f"values must be >= 1: \n\ngrid = {__[0]}")

    elif 'float' in __[-1]:

        if __[-1] == 'soft_float' and (np.array(__[0]) < 0).any():
            raise ValueError(f"{_} -- when numerical is soft float, all search "
                             f"values must be >= 0: \n\ngrid = {__[0]}")

    # CURRENTLY ONLY HANDLES LOGSPACE BASE 10 OR GREATER
    if len(__[0]) >= 3:
        log_grid = np.log10(__[0])
        log_gaps = log_grid[1:] - log_grid[:-1]
        if len(np.unique(log_gaps)) == 1 and log_gaps[0] < 1:
            raise NotImplementedError(
                f"{_} -- currently only handles logspaces with base 10 or greater")
        del log_grid, log_gaps
    # END validate soft/hard/fixed_int/float PART 2 ###############

    _numerical_param_value = __

    del __

    return _numerical_param_value










