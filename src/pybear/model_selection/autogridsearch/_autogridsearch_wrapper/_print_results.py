# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

from ._type_aliases import GridsType, ResultsType


# no pytest


def _print_results(
        _GRIDS: GridsType,
        _RESULTS: ResultsType
    ) -> None:


    for _pass in _RESULTS:
        print(f'Pass {_pass + 1} results:')
        for _param in _RESULTS[_pass]:
            _grid_pad = 80 - 15 - 10
            try:
                _grid = _GRIDS[_pass][_param]
                _grid = list(map(round, _grid, (3 for _ in _grid)))
                # try to round, if except, is str, handle in exception
                print(
                    f' ' * 5 + f'{_param}:'.ljust(15) +
                    f'{str(_grid)[:_grid_pad - 5]}'.ljust(_grid_pad) +
                    f'Result = {round(_RESULTS[_pass][_param], 3)}'
                )
            except:
                print(
                    f' ' * 5 + f'{_param}:'.ljust(15) +
                    f'{str(_GRIDS[_pass][_param])[:_grid_pad - 5]}'.ljust(_grid_pad) +
                    f'Result = {_RESULTS[_pass][_param]}'
                )
        print()









