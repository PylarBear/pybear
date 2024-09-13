# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

import numpy as np
import pandas as pd

from model_selection.GSTCV._GSTCVDask.GSTCVDask import GSTCVDask

from dask_ml.preprocessing import OneHotEncoder as dask_OneHotEncoder
from dask_ml.linear_model import LogisticRegression as dask_LogisticRegression
from dask_ml.model_selection import GridSearchCV as dask_GridSearchCV
from sklearn.pipeline import Pipeline



deep = False


gstcv = GSTCVDask(
    Pipeline(
        steps=[
            ('onehot', dask_OneHotEncoder()),
            ('logistic', dask_LogisticRegression())
        ]
    ),
    param_grid={'onehot__min_frequency': [3,4,5], 'logistic__C': [.0001, .001, .01]}
)

gstcv_params = gstcv.get_params(deep=deep)



gstcv = dask_GridSearchCV(
    Pipeline(
        steps=[
            ('onehot', dask_OneHotEncoder()),
            ('logistic', dask_LogisticRegression())
        ]
    ),
    param_grid={'onehot__min_frequency': [3,4,5], 'logistic__C': [.0001, .001, .01]}
)

skgscv_params = gstcv.get_params(deep=deep)


max_len = max(len(gstcv_params), len(skgscv_params))


DF = pd.DataFrame(index=np.arange(max_len), columns=['skgscv_params', 'gstcv_params'])

for idx in range(max_len):
    try:
        DF.loc[idx, 'gstcv_params'] = list(gstcv_params.keys())[idx]
    except:
        pass

    try:
        DF.loc[idx, 'skgscv_params'] = list(skgscv_params.keys())[idx]
    except:
        pass


print(DF)













