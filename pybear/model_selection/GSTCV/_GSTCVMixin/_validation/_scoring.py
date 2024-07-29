# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


from typing import Callable

import numpy as np

from model_selection.GSTCV._master_scorer_dict import master_scorer_dict

from model_selection.GSTCV._type_aliases import (
    ScorerInputType,
    ScorerWIPType,
)





def _validate_scoring(
        _scoring: ScorerInputType
    ) -> ScorerWIPType:

    """
    Validate scoring, the scoring metric(s) used to evaluate the predictions
    on the test set. Convert any of the valid input formats to an output
    format of dict[str, callable].


    Parameters
    ----------
    _scoring: string, callable, list/tuple, dict, default: 'accuracy'

        For a single scoring metric, a single string or a single callable
        is allowed.

        For evaluating multiple metrics, either give a list of (unique)
        strings or a dict with names as keys and callables as values.
        NOTE that when using custom scorers, each scorer should return a
        single value. Metric functions returning a list/array of values
        can be wrapped into multiple scorers that return one value each.

        Cannot use None, the default scorer cannot be used because the
        threshold cannot be manipulated.


    Return
    ------
    -
        _scoring: dict[str: callable] - dictionary of format
    # {scorer_name: scorer}, when one or multiple metrics are used.


    """


    def string_validation(_string: str):
        _string = _string.lower()
        if _string not in master_scorer_dict:
            if 'roc_auc' in _string or 'average_precision' in _string:
                raise ValueError(
                    f"Don't need to use GridSearchThreshold when scoring "
                    f"is roc_auc or average_precision (auc_pr). \nUse "
                    f"regular dask/sklearn GridSearch and use max(tpr-fpr) "
                    f"to find best threshold for roc_auc, \nor use max(f1) "
                    f"to find the best threshold for average_precision."
                )
            else:
                raise ValueError(
                    f"When specifying scoring by scorer name, must be "
                    f"in {', '.join(list(master_scorer_dict))} ('{_string}')"
                )

        return _string


    def check_callable_is_valid_metric(fxn_name: str, _callable: Callable):
        _truth = np.random.randint(0, 2, (100,))
        _pred = np.random.randint(0, 2, (100,))
        try:
            _value = _callable(_truth, _pred)
        except:
            raise ValueError(f"scoring function '{fxn_name}' excepted "
                             f"during validation")

        try:
            float(_value)
        except:
            raise ValueError(f"scoring function '{fxn_name}' returned a "
            f"non-numeric ({_value})")

        del _truth, _pred, _value


    err_msg = (f"scoring must be "
            f"\n1) a single metric name as string, or "
            f"\n2) a callable(y_true, y_pred) that returns a single "
                f"numeric value, or "
            f"\n3) a list-type of metric names as strings, or "
            f"\n4) a dict of: (metric name: callable(y_true, y_pred), ...)."
            f"\nCannot pass None or bool. Cannot use estimator's default "
                f"scorer."
    )

    try:
        iter(_scoring)
        if isinstance(_scoring, (dict, str)):
            raise Exception
        _is_iter = True
    except:
        _is_iter = False


    if isinstance(_scoring, str):
        _scoring = string_validation(_scoring)
        _scoring = {_scoring: master_scorer_dict[_scoring]}

    elif _is_iter:
        try:
            _scoring = list(np.array(list(_scoring)).ravel())
        except:
            raise TypeError(err_msg)

        if len(_scoring) == 0:
            raise ValueError(f'scoring is empty --- ' + err_msg)

        for idx, string_thing in enumerate(_scoring):
            if not isinstance(string_thing, str):
                raise TypeError(err_msg)
            _scoring[idx] = string_validation(string_thing)
        del idx, string_thing

        _scoring = list(set(_scoring))

        _scoring = {k: v for k, v in master_scorer_dict.items() if k in _scoring}


    elif isinstance(_scoring, dict):
        if len(_scoring) == 0:
            raise ValueError(f'scoring is empty --- ' + err_msg)

        if not all(map(isinstance, _scoring, (str for _ in _scoring))):
            raise ValueError(err_msg)

        for key in list(_scoring.keys()):
            # DONT USE string_validation() HERE, USER-DEFINED CALLABLES
            # CAN HAVE USER-DEFINED NAMES
            new_key = key.lower()
            _scoring[new_key] = _scoring.pop(key)
            check_callable_is_valid_metric(new_key, _scoring[new_key])
        del key, new_key

    elif callable(_scoring):
        check_callable_is_valid_metric(f'score', _scoring)
        _scoring = {f'score': _scoring}

    else:
        raise TypeError(err_msg)

    del err_msg, _is_iter
    del string_validation, check_callable_is_valid_metric


    # dict of functions - Scorer functions used on the held out data to
    # choose the best parameters for the model, in a dictionary of format
    # {scorer_name: scorer}, when one or multiple metrics are used.

    # by sklearn/dask design, name convention changes from 'scoring' to
    # 'scorer_' after conversion to dictionary

    return _scoring
























