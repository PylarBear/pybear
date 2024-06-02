# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from .._type_aliases import ParamsType, BestParamsType


# benchmark test only

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
        dict[str, list[...]] - full set of grid-building instructions for
        all parameters
    _true_best:
        dict[str, [int, float, str]] - True best values for estimator's
        hyperparameters, as provided by the user or generated randomly.

    Return
    ------
    None


    """



    NUM_TYPES, STRING_TYPES = [], []
    for _ in _true_best:
        if 'string' in _demo_cls_params[_][-1]:
            STRING_TYPES.append(_)
        else:
            NUM_TYPES.append(_)

    _max_len = max(list(map(len, STRING_TYPES)) + list(map(len, NUM_TYPES)))
    _pad = min(_max_len, 65)
    _print = lambda x: print(f'{x[:_pad]}:'.ljust(_pad + 5) + f'{_true_best[x]}')
    print(f'Numerical hyperparameters:')
    for x in NUM_TYPES:
        _print(x)
    print(f'\nString hyperparameters:')
    for y in STRING_TYPES:
        _print(y)
    print()

    del NUM_TYPES, STRING_TYPES, _max_len, _pad, _print




















