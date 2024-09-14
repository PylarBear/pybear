# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


from pybear.preprocessing import MinCountTransformer
import dask.array as da
from dask_ml.wrappers import Incremental, ParallelPostFit
import numpy as np
from sklearn.cluster import MiniBatchKMeans

_thresh = 10
x_rows = 100
y_rows = 100
x_cols = 6
y_cols = 2


FLOAT_COL = np.random.randint(1, 4, size=(x_rows - (_thresh - 1), x_cols)).astype(np.float64)
# for _col in range(x_cols):
#     MASK = np.random.choice(list(range(x_rows - (_thresh - 1))), _thresh-1, replace=False)
#     FLOAT_COL[MASK, _col] = np.nan
# del MASK

FLOAT_COL = np.vstack((FLOAT_COL, np.full((_thresh-1, x_cols), 2.5)))



_mct = MinCountTransformer(
    count_threshold=_thresh,
    ignore_float_columns=False,
    ignore_non_binary_integer_columns=True,
    ignore_columns=None,
    ignore_nan=False,
    handle_as_bool=None,
    delete_axis_0=False,
    reject_unseen_values=False,
    max_recursions=1,
    n_jobs=None
)

_test_cls = ParallelPostFit(
    Incremental(
        estimator=_mct   #MiniBatchKMeans()
    )
)



# _test_cls.fit(FLOAT_COL) #, y)
# _test_cls.transform(FLOAT_COL) #, y)

# quit()



DA_X = da.from_array(FLOAT_COL, chunks=(x_rows, x_cols))


y = np.random.randint(0,2, (y_rows, y_cols))

DA_Y = da.from_array(y, chunks=(y_rows, y_cols))

tron = DA_Y.compute()

print(tron)

_test_cls.partial_fit(DA_X, y=DA_Y)   #, classes=None)#, y=None)#, classes=None)#, DA_Y)
_test_cls.fit(DA_X, y=DA_Y)
_test_cls.transform(DA_X)#, DA_Y)

# _mct.fit(DA_X)#, DA_Y)
# _mct.transform(DA_X)#, DA_Y)













