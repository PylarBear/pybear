# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import warnings

from model_selection.GSTCV._type_aliases import (
    RefitType,
    ScorerWIPType
)




def _validate_refit(
    _refit: RefitType,
    _scorer: ScorerWIPType
    ) -> RefitType:

    """
    https://scikit-learn.org/stable/modules/grid_search.html#multimetric-grid-search
    https://ml.dask.org/modules/generated/dask_ml.model_selection.GridSearchCV.html

    refit - bool, str, or callable, default=True

    For single metric evaluation, must be bool.

    For multiple metric evaluation, this needs to be a str denoting the
    scorer that to be used to find the best parameters.

    refit can be set to a callable(cv_results_) ->  best_index_.

    """




    err_msg = (
        f"for single scoring metric, refit must be:"
        f"\n1) bool, "
        f"\n2) None, "
        f"\n3) a single string exactly matching the scoring method in scoring, or "
        f"\n4) a callable that takes cv_results_ as input and returns an integer."
        f"\nfor multiple scoring metrics, refit must be:"
        f"\n1) boolean False"
        f"\n2) None, "
        f"\n3) a single string exactly matching any scoring method in scoring, or "
        f"\n4) a callable that takes cv_results_ as input and returns an integer."
    )


    if not callable(_refit) and not isinstance(_refit, (bool, type(None), str)):
        raise TypeError(err_msg)

    if _refit is None:
        _refit = False

    # _refit can be callable, bool, str
    if len(_scorer) > 1 and _refit is True:
        raise ValueError(err_msg)


    _is_bool = False
    _is_str = False
    _is_callable = False
    if isinstance(_refit, bool):
        _is_bool = True
    elif isinstance(_refit, str):
        _refit = _refit.lower()
        _is_str = True
    elif callable(_refit):
        # CANT VALIDATE CALLABLE OUTPUT HERE, cv_results_ IS NOT AVAILABLE
        _is_callable = True


    if _refit is False or _is_callable:

        # AS OF 24_07_08 THERE ISNT A WAY TO RETURN A best_threshold_
        # WHEN MULTIPLE SCORERS ARE PASSED TO scoring AND refit IS False
        # OR A CALLABLE (OK IF REFIT IS A STRING).  IF THIS EVER CHANGES,
        # THIS WARNING CAN COME OUT.
        if len(_scorer) > 1:
            warnings.warn(
                f"WHEN MULTIPLE SCORERS ARE USED:\n"
                f"Cannot return a best threshold if refit is False or callable.\n"
                f"If refit is False: best_index_, best_estimator_, best_score_, "
                f"and best_threshold_ are not available.\n"
                f"if refit is callable: best_score_ and best_threshold_ "
                f"are not available.\n"
                f"In either case, access score and threshold info via the "
                f"cv_results_ attribute."
            )
    else:  # refit CAN BE True OR (MATCHING A STRING IN scoring) ONLY

        refit_is_true = _refit is True
        refit_is_str = _is_str

        assert refit_is_str is not refit_is_true, \
            f"refit_is_str and refit_is_bool are both {refit_is_str}"

        # _scorer KEYS CAN ONLY BE SINGLE STRINGS: user-defined via dict,
        # 'score', or actual score method name
        if refit_is_true:  # already proved len(_scorer) == 1 when True
            _refit = 'score'
        elif refit_is_str:
            # _scorer keys should already be lower case, but ensure this
            _scorer = {k.lower(): v for k, v in _scorer.items()}
            if _refit not in _scorer:
                raise ValueError(
                    f"if refit is a string, refit must exactly match the "
                    f"(or one of the) scoring methods in scoring"
                )
            elif len(_scorer) == 1:
                _refit = 'score'

        del refit_is_true, refit_is_str

    del err_msg, _is_bool, _is_str, _is_callable

    return _refit



































