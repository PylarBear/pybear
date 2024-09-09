# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

import numpy as np

from typing import Union
from typing_extensions import TypeAlias

# see _type_aliases, these are subtypes of
# DataType, GridType, PointsType, ParamType
NumDataType: TypeAlias = Union[int, float]
InNumGridType: TypeAlias = \
    Union[list[NumDataType], tuple[NumDataType], set[NumDataType]]
InNumPointsType: TypeAlias = Union[int, Union[list[int], tuple[int]]]
InNumParamType: TypeAlias = list[InNumGridType, InNumPointsType, str]
OutNumGridType: TypeAlias = list[NumDataType]
OutNumPointsType: TypeAlias = list[int]
OutNumParamType: TypeAlias = list[OutNumGridType, OutNumPointsType, str]


def _numerical_param_value(
                            _numerical_param_key: str,
                            _numerical_param_value: InNumParamType,
                            total_passes: int
    ) -> OutNumParamType:


    """
    Validate _numerical_param_value --- standardize format

    Integer spaces must be > 1, float spaces must be > 0.

    Regardless of what is entered for the first round number of points,
    that value is overwritten with the actual length of the first grid.

    For fixed float or integer, points must be either the length of the
    first search grid or 1 then 1 thereafter.

    For any case where 1 is entered as points, all points thereafter must
    be 1.

    validate numerical_params' dict value is a list-like that contains
    either:
    (i) a list-like of first-round grid-search points
    (ii) an int or list-like of number of grid points for each pass of
        autogridsearch
    (iii) a string indicating the data type and the search type

    -- or --

    (i) 'linspace' or 'logspace'
    (ii) numerical lower bound of the space
    (iii) numerical upper bound of the space
    (iv) an int or list-like of number of grid points for each pass of
        autogridsearch
    (v) a string indicating the data type and the search type


    GOES OUT AS
    (i) a list-like of first-round grid-search points
    (ii) a list-like of number of grid points for each pass of
        autogridsearch
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
        if isinstance(__, (set, dict, str)):
            raise Exception
        __ = list(__)
    except:
        raise TypeError(err_msg)


    if len(__) not in [3, 5]:
        raise ValueError(err_msg)

    del err_msg
    # END validate container object ** * ** * ** * ** * ** * ** * ** *


    # validate soft/hard/fixed_int/float PART 1 ** * ** * ** * ** * ** *

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

    # END validate soft/hard/fixed_int/float PART 1 ** * ** * ** * ** *

    # validate points part 1 ** * ** * ** * ** * ** * ** * ** * ** * **

    err_msg = (f'{_} -- "points" must be (i) a non-bool integer >= 1 or '
        f'(ii) a list-type of non-bool integers >=1 with len==passes'
        f'\n\npoints:\n{__[-2]}'
    )

    try:
        iter(__[-2])  # IF IS A SINGLE NON-ITERABLE, CONVERT TO LIST
        if isinstance(__[-2], (dict, set, str)):
            raise UnicodeError
        __[-2] = list(__[-2])
    except TypeError:
        __[-2] = [__[-2] for _ in range(total_passes)]
    except UnicodeError:
        raise TypeError(err_msg)
    except Exception as e:
        raise Exception(f"'points' iterable check failed for uncontrolled "
                        f"reason --- {e}")


    # NUMBER OF POINTS IN points MUST MATCH NUMBER OF PASSES
    if len(__[-2]) != total_passes:
        raise ValueError(err_msg + f"\n\ntotal_passes: \n{total_passes}")

    # IF A NON-NUMERIC IS IN POINTS
    try:
        if any(map(isinstance, __[-2], (bool for _ in __[-2]))):
            raise Exception
        _float_test = list(map(float, __[-2]))
    except:
        raise TypeError(err_msg)


    # IF A FLOAT IS IN points
    if any([int(i) != i for i in _float_test]):
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

    del err_msg
    # END validate points part 1 ** * ** * ** * ** * ** * ** * ** * ** *


    # validate first grid when len == 5 ** * ** * ** * ** * ** * ** * **

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
                float(__[idx])
                if 'integer' in __[-1] and (int(__[idx]) != __[idx]):
                    raise Exception
            except:
                raise TypeError(err_msg)

        del err_msg

        # END validate list contains 'linspace/logspace', start_value, end_value

        if __[0] == 'logspace':
            _grid = np.sort(np.logspace(__[1], __[2], __[3][0])).tolist()
        elif __[0] == 'linspace':
            _grid = np.sort(np.linspace(__[1], __[2], __[3][0])).tolist()

        __ = [_grid, __[-2], __[-1]]
        del _grid

    # END validate first grid when len == 5 ** * ** * ** * ** * ** * **

    # LEN MUST BE 3

    # validate list contains [first_grid] in 0 slot ** * ** * ** * ** *

    err_msg = f"{_} -- for '3' format, first element of list must be a list-type"
    try:
        iter(__[0])
        if isinstance(__[0], (dict, str)):
            raise Exception
    except:
        raise TypeError(err_msg)

    try:
        if any(map(isinstance, __[0], (bool for _ in __[0]))):
            raise Exception
        list(map(float, __[0]))
    except:
        raise TypeError(f"{_} -- search values must be numeric")

    __[0] = list(np.sort(list(__[0])))

    # END validate list contains [first_grid] in 0 slot ** * ** * ** *

    # validate points part 2 ** * ** * ** * ** * ** * ** * ** * ** * **

    # in part 1, first grid was not built yet, but validated that points
    #  - is a list
    #  - len==(total_passes)
    #  - has integers > 0
    #  - 'soft' points always > 2
    #  - observes shrink rule

    # need to validate
    #   - fixed points in [1 or len(first grid)]

    # the desired behavior is that if a user enters this [[1,2,3], 1, ...]
    # then the first points is automatically set to len grid, and all
    # passes after just run the single best value: points = [3, 1, 1, ... ]
    # to do this, pinned into making that if all values in points are
    # equal (whether 1 or otherwise), assume that points was entered as
    # int and set all points after first to that value. after some thought
    # simply overwrite whatever user put in 0 slot for points, without
    # notifying if original entry was erroneous

    __[-2][0] = len(__[0])

    # fixed points in [1 or len(first grid)]
    if 'fixed' in __[-1]:
        if any([_points not in [1, __[-2][0]] for _points in __[-2]]):
            raise ValueError(f"{_} -- if fixed int/float, number of points "
                f"must be len(first grid) or 1 \n\npoints = {__[-2]}")

    # END validate points part 2 ** * ** * ** * ** * ** * ** * ** * ** *


    # validate soft/hard/fixed_int/float PART 2 ** * ** * ** * ** * ** * ** *
    if 'integer' in __[-1]:

        if not all([int(i) == i for i in __[0]]):
            raise TypeError(f"{_} -- when numerical is integer (soft, hard, or "
                f"fixed), all search values must be integers: \n\ngrid = {__[0]}")

        if __[-1] in ['hard_integer', 'soft_integer'] and (np.array(__[0]) < 1).any():
            raise ValueError(f"{_}: when numerical is hard/soft integer, "
                f"all search values must be >= 1: \n\ngrid = {__[0]}")

        __[0] = list(map(int, __[0]))

    elif 'float' in __[-1]:

        if __[-1] in ['hard_float', 'soft_float'] and (np.array(__[0]) < 0).any():
            raise ValueError(f"{_} -- when numerical is hard/soft float, "
                f"all search values must be >= 0: \n\ngrid = {__[0]}")

        __[0] = list(map(float, __[0]))

    # LOGSPACE
    if len(__[0]) >= 3 and 0 not in __[0]:
        log_grid = np.log10(__[0])
        log_gaps = log_grid[1:] - log_grid[:-1]
        _unq_log_gap = np.unique(np.round(log_gaps, 14))

        if len(_unq_log_gap) == 1:  # else is not a logspace
            # CURRENTLY ONLY HANDLES LOGSPACE BASE 10 OR GREATER
            err_msg = f"{_} -- only handles logspaces with base 10 or greater"
            if _unq_log_gap[0] < 1:
                raise ValueError(err_msg)
            del err_msg

            # 24_05_14_07_53_00 ENFORCING INTEGER FOR LOGSPACE MAKES MANAGING
            # GAPS IN DRILL SECTION A LOT EASIER
            if int(_unq_log_gap[0]) != _unq_log_gap[0]:
                raise ValueError(f'{_}: logspaces must have integer intervals')

        del log_grid, log_gaps, _unq_log_gap
    # END validate soft/hard/fixed_int/float PART 2 ###############

    _numerical_param_value = __

    del __

    return _numerical_param_value










