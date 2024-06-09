# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


from model_selection.autogridsearch._autogridsearch_wrapper._get_next_param_grid. \
    _get_next_param_grid import _get_next_param_grid




_good_grids =  {
                0: {
                    'a': ['x', 'y', 'z'],
                    'b': [1, 2, 3],
                    'c': [1e0, 1e1, 1e2, 1e3],
                    'd': [40, 50, 60, 70]
                }
                # 1: {
                #     'a': ['x', 'y', 'z'],
                #     'b': [1, 2, 3],
                #     'c': [1e0, 1e1, 1e2, 1e3],
                #     'd': [40, 50, 60, 70]
                # }
    }


_good_params = {
        'a': [['x', 'y', 'z'], 2, 'string'],
        'b': [[1, 2, 3], [3, 1, 1], 'fixed_integer'],
        'c': [[1e0, 1e1, 1e2, 1e3], [4, 1, 1], 'soft_integer'],
        'd': [[40, 50, 60, 70], [4, 1, 1], 'soft_float']
    }


_good_phlite = {
        # 'a': True,
        # 'b': True,
        'c': False,
        'd': False
    }


_good_is_logspace = {
        'a': False,
        'b': False,
        'c': 1.0,
        'd': False
    }


_good_best_params = {
        'a': 'x',
        'b': 1,
        'c': 1e0,
        'd': 40
    }

_good_pass = 1

_good_total_passes = 3

_good_total_passes_is_hard = False

_good_shift_ctr = 0

_good_max_shifts = 3

# _GRIDS, _params, _PHLITE, _IS_LOGSPACE, _shift_ctr, _total_passes_end = \
#     _get_next_param_grid(
#         _good_grids,
#         _good_params,
#         _good_phlite,
#         _good_is_logspace,
#         _good_best_params,
#         _good_pass,
#         _good_total_passes,
#         _good_total_passes_is_hard,
#         _good_shift_ctr,
#         _good_max_shifts
#     )

# print(f'GRIDS:')
# print(_GRIDS)
# print()
# print(f'params:')
# print(_params)
# print()
# print(f'PHLITE:')
# print(_PHLITE)
# print()
# print(f'IS_LOGSPACE:')
# print(_IS_LOGSPACE)
# print()
# print(f'shift_ctr:')
# print(_shift_ctr)




for _good_pass in range(1, _good_total_passes):

    _good_grids, _good_params, _good_phlite, _good_is_logspace, _good_shift_ctr, \
        _good_total_passes = \
            _get_next_param_grid(
                _good_grids,
                _good_params,
                _good_phlite,
                _good_is_logspace,
                _good_best_params,
                _good_pass,
                _good_total_passes,
                _good_total_passes_is_hard,
                _good_shift_ctr,
                _good_max_shifts
            )

    _good_best_params = {_param: _good_grids[_good_pass][_param][0] for _param in _good_params}

    # _good_is_logspace = {_param: False for _param in _good_params}

    print(f'pass {_good_pass} ** * ** * ** * ** * ** * ** * ** * ** * ** * ')
    print(f'GRIDS:')
    for _pass_ in _good_grids:
        print(f"pass {_pass_}:")
        for _param, _grid_ in _good_grids[_pass_].items():
            print(f"     {_param}: {_grid_}")
    print()
    print(f'best_params_:')
    for _param, _best_ in _good_best_params.items():
        print(f"{_param}: {_best_}")
    print()
    print(f'params:')
    for _param, _instr in _good_params.items():
        print(f"     {_param}: {_instr}")
    print()
    print(f'PHLITE:')
    print(_good_phlite)
    print()
    print(f'IS_LOGSPACE:')
    print(_good_is_logspace)
    print()
    print(f'shift_ctr:')
    print(_good_shift_ctr)
    print()
    print(f'good_total_passes:')
    print(_good_total_passes)
    print(f'** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ')


















