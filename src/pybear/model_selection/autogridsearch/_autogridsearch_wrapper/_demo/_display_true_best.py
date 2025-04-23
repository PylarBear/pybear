# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from .._type_aliases import ParamsType, BestParamsType


# benchmark tests only

def _display_true_best(
    _demo_cls_params: ParamsType,
    _true_best: BestParamsType
) -> None:

    """
    Display the best values in _true_best for reference against the best
    values being returned by autogridsearch.

    Parameters
    ----------
    _demo_cls_params:
        ParamsType - full set of grid-building instructions for all
        parameters
    _true_best:
        BestParamsType - True best values for estimator's hyperparameters,
        as provided by the user or generated randomly.

    Return
    ------
    -
        None


    """



    NUM_TYPES, STRING_TYPES, BOOL_TYPES = [], [], []
    for _ in _true_best:
        _type = _demo_cls_params[_][-1]
        if 'fixed_string' in _type:
            STRING_TYPES.append(_)
        elif 'fixed_bool' in _type:
            BOOL_TYPES.append(_)
        else:
            NUM_TYPES.append(_)

    _len = lambda LIST: list(map(len, LIST))
    _max_len = max(_len(STRING_TYPES) + _len(NUM_TYPES) + _len(BOOL_TYPES))
    del _len
    _pad = min(_max_len, 65)
    _print = lambda x: print(f'{x[:_pad]}:'.ljust(_pad + 5) + f'{_true_best[x]}')
    print(f'Numerical hyperparameters:')
    for x in NUM_TYPES:
        _print(x)
    print(f'\nString hyperparameters:')
    for y in STRING_TYPES:
        _print(y)
    print(f'\nBoolean hyperparameters:')
    for y in BOOL_TYPES:
        _print(y)
    print()

    del NUM_TYPES, STRING_TYPES, _max_len, _pad, _print




















