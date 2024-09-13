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
        'score' method, as GSTCVDask never accesses the estimator score
        method because it always uses a 0.5 threshold.

        GSTCVDask warns when a non-dask estimator is used, but does not
        strictly prohibit them. GSTCVDask is explicitly designed for use
        with dask objects (estimators, arrays, and dataframes.) GSTCV is
        recommended for non-dask classifiers.


"""





