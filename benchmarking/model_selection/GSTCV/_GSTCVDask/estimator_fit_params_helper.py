# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import numpy as np
import dask.array as da


from pybear.model_selection.GSTCV._GSTCVDask._fit._estimator_fit_params_helper \
    import _estimator_fit_params_helper


# this file is a general functionality test that arose out of troubleshooting
# the building of dask estimator_fit_params_helper





_rows = 10

# data will always be dask array because of _validate_X_y
from dask_ml.datasets import make_classification
X, y = make_classification(n_samples=_rows, n_features=5, chunks=(100,5))

# aim for worst case, KFold and stuff in fit_params was passed with non-dask
from sklearn.model_selection import KFold as sk_KFold
from dask_ml.model_selection import KFold as dask_KFold
_sk_kfold = list(sk_KFold(n_splits=3).split(X.compute(), y.compute()))
_dask_kfold = list(dask_KFold(n_splits=3).split(X, y))



sk_fit_params = {
    'sample_weight': np.random.uniform(0, 1, _rows),
    'fake_sample_weight': np.random.uniform(0, 1, _rows // 2),
    'made_up_param_1':  'something_else',
    'made_up_param_2': False,
    'some_other_param_1': {'abc': 123}
}

dask_fit_params = {
    'sample_weight': da.random.uniform(0, 1, _rows),
    'fake_sample_weight': da.random.uniform(0, 1, _rows // 2),
    'made_up_param_1':  'something_else',
    'made_up_param_2': False,
    'some_other_param_1': {'abc': 123}
}


for _fit_params_name, _fit_params in zip(('dask_fit_params', 'sk_fit_params'), (dask_fit_params, sk_fit_params)):
    for _kfold_name, _kfold in zip(('_dask_kfold', '_sk_kfold'), (_dask_kfold, _sk_kfold)):

        out = _estimator_fit_params_helper(
            _rows,
            _fit_params,
            _kfold
        )

        print(f'** * ' * 20)
        print(f'fit_params = {_fit_params_name}, kfold = {_kfold_name}')
        print()
        for cv_idx, pass_fit_params in out.items():
            print(f'cv_idx = {cv_idx}')
            for fit_param, param_value in pass_fit_params.items():
                print(f'    {fit_param}: {param_value}')

        print()
        print(f'END fit_params = {_fit_params_name}, kfold = {_kfold_name}')
        print(f'** * ' * 20)

















