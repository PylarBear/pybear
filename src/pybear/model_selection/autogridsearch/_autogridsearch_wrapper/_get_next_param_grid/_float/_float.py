# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


from typing_extensions import Union
from typing_extensions import TypeAlias
from ._float_linspace import _float_linspace
from ._float_logspace import _float_logspace


# see _type_aliases; subtypes for DataType & GridType
FloatDataType: TypeAlias = float
FloatGridType: TypeAlias = \
    Union[list[FloatDataType], tuple[FloatDataType], set[FloatDataType]]


def _float(
            _SINGLE_GRID: FloatGridType,
            _is_logspace: Union[bool, float],
            _posn: int,
            _is_hard: bool,
            _hard_min: FloatDataType,
            _hard_max: FloatDataType,
            _points: int
    ) -> tuple[list[FloatDataType], Union[bool, float]]:

    """
    Take in a float's grid from the last round of GridSearch along with
    the index position of the best value within that grid and return a
    new grid for the upcoming (current pass') GridSearch. Important
    factors in building the next grid: hard/soft, linspace/logspace,
    number of points.

    Parameters
    ----------
    _SINGLE_GRID:
        Union[list[int], tuple[int], set[int]] - The last round's search
        grid for a single param. _SINGLE_GRID must be sorted ascending,
        and is presumed to be by _validation._numerical_params (at least
        initially).
    _is_logspace:
        Union[bool, float] - For numerical params, if the space is linear,
        or some other non-standard interval, it is False. If it is
        logspace, the 'truth' of being a logspace is represented by a
        number indicating the interval of the logspace. E.g.,
        np.logspace(-5, 5, 11) would be represented by 1.0, and
        np.logspace(-20, 20, 9) would be represented by 5.0.
    _posn:
        int - the index position in the previous round's grid where the
        best value fell
    _is_hard:
        bool - whether the parameter has hard left and right boundaries.
        This field is read from the dtype/search field in _params. If
        hard, the left and right bounds are set from the lowest and
        highest values in the first round's search grid (the grid that
        is in _params.)
    _hard_min:
        float - if hard, the minimum value in the first round's search grid.
    _hard_max:
        float - if hard, the maximum value in the first round's search grid.
    _points:
        int - the number of points for the current grid as read from _params.

    Return
    ------
    -
         _NEW_GRID:
            list[int] - new search grid for the current pass' upcoming search.

         _is_logspace:
            Union[bool, float] - current float parameter grid space is /
            is not logspace. All params leaving this module should be
            linspace and the return value should always be False.


    """

    # 24_05_17_09_16_00 do not put _validation here, keep it in the individual
    # modules to get the module names

    if not _is_logspace:

        _OUT_GRID = _float_linspace(
            _SINGLE_GRID,
            _posn,
            _is_hard,
            _hard_min,
            _hard_max,
            _points
        )



    elif _is_logspace:  # CAN ONLY HAPPEN ON FIRST PASS AFTER SHIFTER

        _OUT_GRID = _float_logspace(
            _SINGLE_GRID,
            _posn,
            _is_logspace,
            _is_hard,
            _hard_min,
            _hard_max,
            _points
        )

        # _float_logspace automatically converts logspace to linspace,
        # making _is_logspace False

        _is_logspace = False


    return _OUT_GRID, _is_logspace








