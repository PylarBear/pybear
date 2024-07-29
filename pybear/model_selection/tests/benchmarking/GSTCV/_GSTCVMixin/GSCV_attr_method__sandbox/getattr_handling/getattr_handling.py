# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

import sys
import numpy as np

# Used in GSTCV__ATTR_METHOD_RESPONSES
# support functions for getting attr and method values out of a GSCV-type class.


# support access_attrs() and access_methods()
def getattr_try_handling(
    attr_or_method_output: any,
    attr_or_method_name: str,
    _round: str,
    _gscv_type: str,
    _refit: bool,
    dump_to_file: bool,
    ATTR_OR_METHOD_ARRAY_DICT: dict
) -> dict:

    """
    Function that handles successful getattr on a GSCV class. Supports
    access_attrs() and access_methods().

    If printing to screen, print immediately; if writing to file, store
    result in ATTR_OR_METHOD_ARRAY_DICT.

    """

    if not dump_to_file:  # print to screen
        print(f"{attr_or_method_name} = {attr_or_method_output}")
    else:
        _key = f"{_gscv_type}_refit_{_refit}_{_round}"

        if _key not in ATTR_OR_METHOD_ARRAY_DICT:
            raise ValueError(
                f"attempting to write to ATTR_OR_METHOD_ARRAY_DICT key "
                f"'{_key}' but it doesnt exist"
            )

        ATTR_OR_METHOD_ARRAY_DICT[_key] = np.insert(
            ATTR_OR_METHOD_ARRAY_DICT[_key],
            len(ATTR_OR_METHOD_ARRAY_DICT[_key]),
            str(attr_or_method_output),
            axis=0
        )

        del _key

    return ATTR_OR_METHOD_ARRAY_DICT


# support access_attrs() and access_methods()
def getattr_except_handling(
    exception_info: str,
    attr_or_method_name: str,
    _round: str,
    _gscv_type: str,
    _refit: bool,
    dump_to_file: bool,
    ATTR_OR_METHOD_ARRAY_DICT: dict
) -> dict:

    """
    Function that handles failed getattr on a GSCV class. Supports
    access_attrs() and access_methods().

    If printing to screen, print exception info immediately; if writing
    to file, store exception info result in ATTR_OR_METHOD_ARRAY_DICT.

    """

    if not dump_to_file:  # print to screen
        print(f"{attr_or_method_name}: {exception_info}")
    else:
        _key = f"{_gscv_type}_refit_{_refit}_{_round}"

        if _key not in ATTR_OR_METHOD_ARRAY_DICT:
            raise ValueError(
                f"attempting to write to ATTR_OR_METHOD_ARRAY_DICT key "
                f"'{_key}' but it doesnt exist"
            )

        ATTR_OR_METHOD_ARRAY_DICT[_key] = np.insert(
            ATTR_OR_METHOD_ARRAY_DICT[_key],
            len(ATTR_OR_METHOD_ARRAY_DICT[_key]),
            str(exception_info),
            axis=0
        )

        del _key

    return ATTR_OR_METHOD_ARRAY_DICT



# ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
# FUNCTION TO DISPLAY GRIDSEARCH ATTR OUTPUT B4 & AFTER CALLS TO fit() ** ** **

def access_attrs(
    xgscv_instance,
    _round: str,
    _gscv_type: str,
    _refit: bool,
    dump_to_file: bool,
    ATTR_OR_METHOD_ARRAY_DICT: dict
):

    _exc = lambda: sys.exc_info()[1]

    if not dump_to_file:
        print(f"\nStart attr round {_round} {_gscv_type} refit {_refit} " +
              f"** " * 20 + f"\n")

    # keep this one separate from the main loop below because of the slice
    try:
        xgscv_instance.cv_results_
        ATTR_OR_METHOD_ARRAY_DICT = getattr_try_handling(
            xgscv_instance.cv_results_["params"][0],
            f'cv_results_["params"][0]',
            _round,
            _gscv_type,
            _refit,
            dump_to_file,
            ATTR_OR_METHOD_ARRAY_DICT
        )
    except:
        ATTR_OR_METHOD_ARRAY_DICT = getattr_except_handling(
            _exc(),
            f"cv_results_",
            _round,
            _gscv_type,
            _refit,
            dump_to_file,
            ATTR_OR_METHOD_ARRAY_DICT
        )

    for attr in [
        f"best_estimator_", f"best_score_", f"best_params_",
        f"best_index_", f"scorer_", f"n_splits_", f"refit_time_",
        f"multimetric_", f"classes_", f"n_features_in_", 'feature_names_in_'
        ]:

        try:

            try:
                value = getattr(xgscv_instance, attr).compute()
            except:
                value = getattr(xgscv_instance, attr)

            ATTR_OR_METHOD_ARRAY_DICT = getattr_try_handling(
                value,
                attr,
                _round,
                _gscv_type,
                _refit,
                dump_to_file,
                ATTR_OR_METHOD_ARRAY_DICT
            )
        except:
            ATTR_OR_METHOD_ARRAY_DICT = getattr_except_handling(
                _exc(),
                attr,
                _round,
                _gscv_type,
                _refit,
                dump_to_file,
                ATTR_OR_METHOD_ARRAY_DICT
            )

    if not dump_to_file: print(
        f"\nEnd attr round {_round} {_gscv_type} refit {_refit} " + f"** " * 20,
        f"\n")

    return ATTR_OR_METHOD_ARRAY_DICT


# END FUNCTION TO DISPLAY GRIDSEARCH ATTR OUTPUT B4 & AFTER CALLS TO fit() ** *
# ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **


# ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
# FUNCTION TO DISPLAY GRIDSEARCH METHOD OUTPUT B4 & AFTER CALLS TO fit() ** **

# THESE METHODS NAMES ARE VERIFIED & UP-TO-DATE AS OF 24_02_19_09_26_00

# self.decision_function(X)
# self.get_metadata_routing() --- sklearn only
# self.get_params(deep:bool=True)
# self.inverse_transform(Xt)
# self.predict(X)
# self.predict_log_proba(X)
# self.predict_proba(X)
# self.score(X, y=None, **params)
# self.score_samples(X) --- sklearn only
# self.set_params(**params)
# self.transform(X)
# self.visualize(filename=None, format=None) --- dask only


def access_methods(
    xgscv_instance,
    X,
    y,
    _round: str,
    _gscv_type: str,
    _refit: str,
    dump_to_file: bool,
    METHOD_ARRAY_DICT: dict,
    **score_params
    ):

    _exc = lambda: sys.exc_info()[1]

    if not dump_to_file:
        print(f"\nStart method round {_round} {_gscv_type} refit {_refit} " +
              f"** " * 20 + f"\n")


    NAMES_ARGS_KWARGS = [
        (f"decision_function", [X], {}),
        (f"get_metadata_routing", [], {}),
        (f"get_params", [], {'deep': True}),
        (f"inverse_transform", [X], {}),
        (f"predict", [X], {}),
        (f"predict_log_proba", [X], {}),
        (f"predict_proba", [X], {}),
        (f"score", [X,y], score_params),
        (f"score_samples", [X], {}),
        (f"set_params", [], {'estimator__C': 100}),
        (f"transform", [X], {}),
        (f'visualize', [],{'filename':None, 'format':None})
    ]


    for attr, args, kwargs in NAMES_ARGS_KWARGS:

        try:
            try:
                value = getattr(xgscv_instance, attr)(*args, **kwargs).compute()
            except:
                value = getattr(xgscv_instance, attr)(*args, **kwargs)

            METHOD_ARRAY_DICT = getattr_try_handling(
                value,
                attr,
                _round,
                _gscv_type,
                _refit,
                dump_to_file,
                METHOD_ARRAY_DICT
            )
        except:
            METHOD_ARRAY_DICT = getattr_except_handling(
                _exc(),
                attr,
                _round,
                _gscv_type,
                _refit,
                dump_to_file,
                METHOD_ARRAY_DICT
            )

    if not dump_to_file: print(
        f"End method round {_round} {_gscv_type} refit {_refit} " + f"** " * 20 + f"\n")

    return METHOD_ARRAY_DICT


# END FUNCTION TO DISPLAY GRIDSEARCH METHOD OUTPUT B4 & AFTER CALLS TO fit() **
# ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **












