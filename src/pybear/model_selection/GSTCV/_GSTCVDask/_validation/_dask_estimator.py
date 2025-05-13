# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from ..._type_aliases import ClassifierProtocol

import sys
import warnings

from sklearn.pipeline import Pipeline



def _val_dask_estimator(
    _estimator: ClassifierProtocol
) -> None:

    """
    The GSTCVDask module is expected to most likely encounter dask_ml,
    xgboost, and lightgbm dask estimators. The estimator must be passed
    as an instance, not the class itself.

    Warn if the estimator is not a dask classifier, either from dask
    itself, or from XGBoost or LightGBM.


    Parameters
    ----------
    _estimator:
        the estimator to be validated


    Return
    ------
    -
        None

    """


    # validate estimator ** * ** * ** * ** * ** * ** * ** * ** * ** * **

    def get_inner_most_estimator(__estimator):

        try:
            if isinstance(__estimator, Pipeline):
                return get_inner_most_estimator(__estimator.steps[-1][-1])
            else:
                return get_inner_most_estimator(__estimator.estimator)
        except:
            return __estimator


    __estimator = get_inner_most_estimator(_estimator)

    try:
        _module = sys.modules[__estimator.__class__.__module__].__file__
    except:
        raise AttributeError(f"'{__estimator.__class__.__name__}' is not "
            f"a valid classifier")

    # 24_08_04_07_28_00 change raise to warn
    # to allow XGBClassifier, reference errors associated with
    # DaskXGBClassifier and dask GridSearch CV
    __ = str(_module).lower()
    if 'dask' not in __:
        warnings.warn(f"'{__estimator.__class__.__name__}' does not "
            f"appear to be a dask classifier.")
    # if 'dask' not in __ and 'conftest' not in __:  # allow pytest with
    # mock clf
        # raise TypeError(f"'{__estimator.__class__.__name__}' is not a
        #     f"dask classifier. GSTCVDask can only accept dask classifiers. "
        #     f"\nTo use non-dask classifiers, use the GSTCV package.")

    del get_inner_most_estimator, __estimator, _module

    # END validate estimator ** * ** * ** * ** * ** * ** * ** * ** * **







