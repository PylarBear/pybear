# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


"""

    estimator:
        estimator object - Must be a binary classifier that conforms to
        the sci-kit learn estimator API interface. The classifier must
        have 'fit', 'set_params', 'get_params', and 'predict_proba'
        methods. If the classifier does not have predict_proba, try to
        wrap with CalibratedClassifierCV. The classifier does not need a
        'score' method, as GSTCV never accesses the estimator score
        method because it always uses a 0.5 threshold.

        GSTCV deliberately blocks dask classifiers (including, but not
        limited to, dask_ml, xgboost, and lightGBM dask classifiers.) To
        use dask classifiers, use GSTCVDask.

"""








