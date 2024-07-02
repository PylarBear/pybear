# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import numpy as np
import pandas as pd
import string

from sklearn.model_selection import train_test_split as sklearn_train_test_split
from sklearn.datasets import make_classification as sklearn_make_classification
from sklearn.model_selection import GridSearchCV as sklearn_GridSearchCV
from sklearn.linear_model import LogisticRegression as sklearn_Logistic
from sklearn.metrics import balanced_accuracy_score

import dask.array as da
import dask.dataframe as ddf
from dask_ml.model_selection import train_test_split as dask_train_test_split
from dask_ml.datasets import make_classification as dask_make_classification
from dask_ml.model_selection import GridSearchCV as dask_GridSearchCV
from dask_ml.linear_model import LogisticRegression as dask_Logistic









%%time

_rows = 2_000_000
_columns = 10
COLUMNS = list(string.ascii_lowercase[:_columns])

sk_np_X = np.random.randint(0,10,(_rows,_columns))
sk_np_y = np.random.randint(0,2,(_rows,))

sk_df_X = pd.DataFrame(data=sk_np_X, columns=COLUMNS)
sk_df_y = pd.DataFrame(data=sk_np_y, columns=['Y'])

da_da_X = da.random.randint(0,10,(_rows,_columns)).rechunk((_rows//10, _columns))
da_da_y = da.random.randint(0,2,(_rows,)).rechunk((_rows//10,))

da_df_X = ddf.from_array(da_da_X, columns=COLUMNS, chunksize=(_rows//10,))
da_df_y = ddf.from_array(da_da_y, columns=['Y'], chunksize=(_rows//10,))



# BEARS LOOK AT NP ARRAYS ######################################################

%%time
sk_np_X1, sk_np_X_test, sk_np_y1, sk_np_y_test = sklearn_train_test_split(sk_np_X, sk_np_y, test_size=0.2)
sk_np_X_train, sk_np_X_val, sk_np_y_train, sk_np_y_val = sklearn_train_test_split(sk_np_X1, sk_np_y1, test_size=0.25)

%%time
sk_logistic_np = sklearn_Logistic(max_iter=10_000, tol=1e-6)

# VERIFY DTYPES
sk_np_X_train

sk_np_y_train

%%time
sk_logistic_np.fit(sk_np_X_train, sk_np_y_train)

# END BEARS LOOK AT NP ARRAYS ######################################################



### BEAR TRIES TO SPEED UP DASK ARRAYS ###############################################################################

%%time
da_da_X1, da_da_X_test, da_da_y1, da_y_test = dask_train_test_split(da_da_X, da_da_y, test_size=0.2)
da_da_X_train, da_da_X_val, da_da_y_train, da_da_y_val = dask_train_test_split(da_da_X1, da_da_y1, test_size=0.25)

%%time
da_da_X_train = da_da_X_train.rechunk(da_da_X_train.shape)
da_da_y_train = da_da_y_train.rechunk(da_da_y_train.shape)



%%time
da_logistic_da = dask_Logistic(max_iter=10_000, tol=1e-6)

# VERIFY DTYPES
da_da_X_train

da_da_y_train

%%time
da_logistic_da.fit(da_da_X_train, da_da_y_train)

### END BEAR TRIES TO SPEED UP DASK ARRAYS ###########################################################################







# BEARS LOOK AT SK DATAFRAMES ######################################################################################

sk_df_X = pd.DataFrame(data=np.random.randint(0,10,(_rows,_columns)), columns=COLUMNS)
sk_df_y = pd.DataFrame(data=np.random.randint(0,2,(_rows,)), columns=['Y'])

%%time
sk_df_X1, sk_df_X_test, sk_df_y1, sk_df_y_test = sklearn_train_test_split(sk_df_X, sk_df_y, test_size=0.2)
sk_df_X_train, sk_df_X_val, sk_df_y_train, sk_df_y_val = sklearn_train_test_split(sk_df_X1, sk_df_y1, test_size=0.25)

%%time
sk_logistic_df = sklearn_Logistic(max_iter=10_000, tol=1e-6)

# VERIFY DTYPES
sk_df_X_train

sk_df_y_train

%%time
sk_logistic_df.fit(sk_df_X_train, sk_df_y_train)

# BEARS LOOK AT SK DATAFRAMES ######################################################################################



### BEAR TRIES TO SPEED UP DASK DATAFRAMES ###############################################################################

# dask_Logistic CANT TAKE DDFs

### END BEAR TRIES TO SPEED UP DASK DATAFRAMES ###############################################################################





















X = np.random.randint(0,10,(500,5))
y = np.random.randint(0,2,(500,))

tater = sklearn_GridSearchCV(
                                estimator=sklearn_Logistic(),
                                param_grid={'C':[100]},
                                scoring=['balanced_accuracy','accuracy'],
                                refit='balanced_accuracy',
                                return_train_score=True
)

tater.fit(X,y)

tater.predict_proba(X)

tater.score(X, y)

# SCORE BY balanced_accuracy_score
balanced_accuracy_score(y, tater.predict(X))

DUM = pd.DataFrame(tater.cv_results_)
for _ in DUM:
    print(f"{_}".ljust(30) + f"{DUM[_].to_frame().to_numpy()[0][0]}")

  data=[[0.5 , 0.5 ],
        [0.25, 0.25],
        [0.5 , 0.5 ],
        [0.25, 0.25],
        [0.25, 0.25]],



new_tater = tater.set_params(estimator__C=10)

new_tater.predict_proba(X)

new_tater.score(X, y)



from GridSearchThresholdCV import GridSearchThresholdCV

test_gstcv = GridSearchThresholdCV(
                                    estimator=sklearn_Logistic(),
                                    param_grid={'C':[100]},
                                    scoring=['balanced_accuracy','accuracy'],
                                    refit='balanced_accuracy',
                                    thresholds=np.linspace(0,1,21),
                                    return_train_score=True
)

test_gstcv.fit(X,y)

test_gstcv.predict_proba(X)

test_gstcv.score(X,y)

test_gstcv.best_index_

test_gstcv.best_threshold_

# DUMP_DF = pd.DataFrame(test_gstcv.cv_results_)
# DUMP_DF.to_csv(r'/home/bear/Desktop/GSTCV_TEST_CV_RESULTS.ods')

tftssm = test_gstcv._TEST_FOLD_x_THRESHOLD_x_SCORER__SCORE_MATRIX
tftssm[:, 7, :]

tftssm.mean(axis=0)





new_test_gstcv = test_gstcv.set_params(estimator__C=10)

# for _thresh in np.linspace(0,1,21):
#     new_test_gstcv.best_threshold_ = _thresh
#     print(f'{_thresh}: {new_test_gstcv.score(X,y)}')

new_test_gstcv.best_threshold_




