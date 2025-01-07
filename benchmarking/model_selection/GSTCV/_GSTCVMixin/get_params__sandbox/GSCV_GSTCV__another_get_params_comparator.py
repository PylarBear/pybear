# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import numpy as np
import pandas as pd

from pybear.model_selection.GSTCV._GSTCV.GSTCV import GSTCV

from sklearn.preprocessing import OneHotEncoder as sk_OneHotEncoder
from sklearn.linear_model import LogisticRegression as sk_LogisticRegression
from sklearn.model_selection import GridSearchCV as sk_GridSearchCV
from sklearn.pipeline import Pipeline



# THIS MODULE BUILDS SK GSCV AND PYBEAR GSTCV WITH PIPELINES.
# ACCESS get_params() WITH deep == True OR False FOR BOTH
# COMPARE THE get_params() KEYS OUTPUT HEAD TO HEAD IN A DF
# THIS POPULATES THE DF IN THE ORDER THAT THE PARAMS ARE STORED IN THE
# RESPECTIVE paramsdict FOR EACH.


"""
deep = False
        skgscv_params        gstcv_params
0                  cv                  cv
1         error_score         error_score
2           estimator           estimator
3              n_jobs              n_jobs
4          param_grid          param_grid
5        pre_dispatch                   -
6               refit               refit
7  return_train_score  return_train_score          
8             scoring             scoring
9                   -          thresholds
10            verbose             verbose

deep = True
                               skgscv_params                              gstcv_params
0                                         cv                                        cv
1                                error_score                               error_score
2                          estimator__memory                         estimator__memory
3                           estimator__steps                          estimator__steps
4                         estimator__verbose                        estimator__verbose
5                          estimator__onehot                         estimator__onehot
6                        estimator__logistic                       estimator__logistic
7              estimator__onehot__categories             estimator__onehot__categories
8                    estimator__onehot__drop                   estimator__onehot__drop
9                   estimator__onehot__dtype                  estimator__onehot__dtype
10  estimator__onehot__feature_name_combiner  estimator__onehot__feature_name_combiner
11         estimator__onehot__handle_unknown         estimator__onehot__handle_unknown
12         estimator__onehot__max_categories         estimator__onehot__max_categories
13          estimator__onehot__min_frequency          estimator__onehot__min_frequency
14          estimator__onehot__sparse_output          estimator__onehot__sparse_output
15                    estimator__logistic__C                    estimator__logistic__C
16         estimator__logistic__class_weight         estimator__logistic__class_weight
17                 estimator__logistic__dual                 estimator__logistic__dual
18        estimator__logistic__fit_intercept        estimator__logistic__fit_intercept
19    estimator__logistic__intercept_scaling    estimator__logistic__intercept_scaling
20             estimator__logistic__l1_ratio             estimator__logistic__l1_ratio
21             estimator__logistic__max_iter             estimator__logistic__max_iter
22          estimator__logistic__multi_class          estimator__logistic__multi_class
23               estimator__logistic__n_jobs               estimator__logistic__n_jobs
24              estimator__logistic__penalty              estimator__logistic__penalty
25         estimator__logistic__random_state         estimator__logistic__random_state
26               estimator__logistic__solver               estimator__logistic__solver
27                  estimator__logistic__tol                  estimator__logistic__tol
28              estimator__logistic__verbose              estimator__logistic__verbose
29           estimator__logistic__warm_start           estimator__logistic__warm_start
30                                 estimator                                 estimator
31                                    n_jobs                                    n_jobs
32                                param_grid                                param_grid
33                              pre_dispatch                                     refit
34                                     refit                        return_train_score
35                        return_train_score                                   scoring
36                                   scoring                                thresholds
37                                   verbose                                   verbose

"""



deep = True


gstcv = GSTCV(
    Pipeline(
        steps=[
            ('onehot', sk_OneHotEncoder()),
            ('logistic', sk_LogisticRegression())
        ]
    ),
    param_grid={
        'onehot__min_frequency': [3,4,5],
        'logistic__C': [.0001, .001, .01]
    }
)

gstcv_params = gstcv.get_params(deep=deep)



gscv = sk_GridSearchCV(
    Pipeline(
        steps=[
            ('onehot', sk_OneHotEncoder()),
            ('logistic', sk_LogisticRegression())
        ]
    ),
    param_grid={
        'onehot__min_frequency': [3,4,5],
        'logistic__C': [.0001, .001, .01]
    }
)

skgscv_params = gscv.get_params(deep=deep)


max_len = max(len(gstcv_params), len(skgscv_params))


DF = pd.DataFrame(
    index=np.arange(max_len),
    columns=['skgscv_params', 'gstcv_params']
)

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













