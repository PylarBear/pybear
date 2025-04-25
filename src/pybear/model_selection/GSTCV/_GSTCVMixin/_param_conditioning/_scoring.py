# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from typing import Callable
from ..._type_aliases import (
    ScorerInputType,
    ScorerWIPType,
)

from ....GSTCV._master_scorer_dict import master_scorer_dict



def _cond_scoring(
    _scoring: ScorerInputType
) -> ScorerWIPType:

    """
    Condition `scoring`, the scoring metric(s) used to evaluate the
    predictions on the test (and possibly train) sets. Convert any of
    the valid input formats to an output format of dict[str, callable].
    Can come in here as str, Sequence[str], callable, dict[str, callable].


    Parameters
    ----------
    _scoring:
        Union[str, Sequence[str], callable, dict[str, callable]] - The
        scoring metric(s) used to evaluate the predictions on the test
        (and possibly train) sets.


    Return
    ------
    -
        _scoring: dict[str, callable] - dictionary of format
            {scorer_name: scorer callable}, when one or multiple metrics
            are used.

    """


    try:
        if isinstance(_scoring, Callable):
            raise Exception
        iter(_scoring)
        if isinstance(_scoring, (dict, str)):
            raise Exception
        _is_list_like = True
    except:
        _is_list_like = False


    if isinstance(_scoring, str):
        _scoring = {_scoring.lower(): master_scorer_dict[_scoring]}

    elif callable(_scoring):
        _scoring = {f'score': _scoring}

    elif _is_list_like:

        _scoring = list(_scoring)

        for _idx, _string in enumerate(_scoring):
            _scoring[_idx] = _string.lower()
        del _idx, _string

        _scoring = list(set(_scoring))

        _scoring = {k: v for k, v in master_scorer_dict.items() if k in _scoring}

    elif isinstance(_scoring, dict):

        for key in list(_scoring.keys()):
            _scoring[key.lower()] = _scoring.pop(key)

    else:
        raise Exception


    del _is_list_like


    # dict of functions - Scorer functions used on the held out data to
    # choose the best parameters for the model, in a dictionary of format
    # {scorer_name: scorer}, when one or multiple metrics are used.

    # by sklearn/dask design, name convention changes from 'scoring' to
    # 'scorer_' after conversion to dictionary


    return _scoring






