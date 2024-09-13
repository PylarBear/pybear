# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

import numpy as np
import pandas as pd

from model_selection.GSTCV._GSTCV.GSTCV import GSTCV

from sklearn.preprocessing import OneHotEncoder as sk_OneHotEncoder
from sklearn.linear_model import LogisticRegression as sk_LogisticRegression
from sklearn.model_selection import GridSearchCV as sk_GridSearchCV
from sklearn.pipeline import Pipeline



deep = False


gstcv = GSTCV(
    Pipeline(
        steps=[
            ('onehot', sk_OneHotEncoder()),
            ('logistic', sk_LogisticRegression())
        ]
    ),
    param_grid={'onehot__min_frequency': [3,4,5], 'logistic__C': [.0001, .001, .01]}
)

gstcv_params = gstcv.get_params(deep=deep)



gstcv = sk_GridSearchCV(
    Pipeline(
        steps=[
            ('onehot', sk_OneHotEncoder()),
            ('logistic', sk_LogisticRegression())
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













