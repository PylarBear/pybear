import numpy as np
from debug import time_memory_tester as tmt
import sparse_dict as sd
from general_data_ops import create_random_sparse_numpy as crsn

'''
##################################################################################################################
# NUMPY ##########################################################################################################
def make_XTX_from_X_as_column(X):
    outer_len = len(X)
    XTX = np.empty((outer_len, outer_len), dtype=np.float64)
    for outer_idx1 in range(outer_len):
        for outer_idx2 in range(outer_idx1+1):
            dot = np.matmul(X[outer_idx1], X[outer_idx2])
            XTX[outer_idx1][outer_idx2] = dot
            XTX[outer_idx2][outer_idx1] = dot
    print(f'make_XTX_from_X_as_column=\n{XTX}')
    return XTX

def make_XTX_from_matmul_XT_and_X(X, X_T):
    _ = np.matmul(X_T, X)
    print(f'make_XTX_from_matmul_XT_and_X=\n{_}')
    return _

def make_XTX_from_X_as_row(X):
    inner_len = len(X[0])
    XTX = np.empty((inner_len, inner_len), dtype=np.float64)
    for inner_idx1 in range(inner_len):
        for inner_idx2 in range(inner_idx1+1):
            dot = np.matmul(X[:, inner_idx1], X[:, inner_idx2])
            XTX[inner_idx1][inner_idx2] = dot
            XTX[inner_idx2][inner_idx1] = dot
    print(f'make_XTX_from_X_as_row=\n{XTX}')
    return XTX



# FUNCTIONALITY TEST  WAS USED FOR NP, THEN CONVERTED OVER FOR SD, SO CHANGE BACK TO TEST NP
fxn = crsn.create_random_sparse_numpy
ctr = 0
for OBJ in (fxn(0,10,(3,10),50,np.int32), fxn(-10,10,(10,3),50,np.float64), fxn(-5,6,(100,100),50,np.int32)):
    ctr+=1
    print(f'Running test {ctr}...')
    OBJ = sd.zip_list_as_py_float(OBJ)

    XTX_1 = sd_make_XTX_from_X_as_column(sd.sparse_transpose(OBJ))
    XTX_2 = sd_make_XTX_from_X_as_row(OBJ)
    XTX_3 = sd_make_XTX_from_matmul_XT_and_X(OBJ)
    
    if not sd.core_sparse_equiv(XTX_1, XTX_2) or not sd.core_sparse_equiv(XTX_1, XTX_3):
        print(f'1 = ')
        print(XTX_1)
        print()
        print(f'2 = ')
        print(XTX_2)
        print()
        print(f'3 = ')
        print(XTX_3)
        print()

        raise Exception(f'ERROR')


# TIME TEST
X = crsn.create_random_sparse_numpy(-10,10,(10000, 300), 50, np.int32)
X_T = X.transpose()

tmt.time_memory_tester(
                         (f'make_XTX_from_X_as_column', make_XTX_from_X_as_column, [X_T], {}),
                         (f'make_XTX_from_X_as_row', make_XTX_from_X_as_row, [X], {}),
                         (f'make_XTX_from_matmul_XT_and_X', make_XTX_from_matmul_XT_and_X, [X, X_T], {}),
                         number_of_trials=5,
                         rest_time=2
                         )


# m,n = ?,? (MORE COLUMNS THAN ROWS)
# make_XTX_from_X_as_column                         average, sdev: time = 3.352 sec, 0.010; mem = 7.000, 0.000
# make_XTX_from_X_as_row                            average, sdev: time = 3.605 sec, 0.019; mem = 7.000, 0.000
# make_XTX_from_matmul_XT_and_X                     average, sdev: time = 0.306 sec, 0.002; mem = 3.000, 0.000

# m,n = 1000,1000  (EXAMPLES = COLUMNS)
# make_XTX_from_X_as_column                         average, sdev: time = 10.230 sec, 0.028; mem = 8.000, 0.000
# make_XTX_from_X_as_row                            average, sdev: time = 10.598 sec, 0.020; mem = 8.000, 0.000
# make_XTX_from_matmul_XT_and_X                     average, sdev: time = 11.925 sec, 0.035; mem = 4.000, 0.000

# m,n = 10000, 300 (MORE EXAMPLES THAN COLUMNS)
# make_XTX_from_X_as_column                         average, sdev: time = 3.145 sec, 0.025; mem = 0.000, 0.000
# make_XTX_from_X_as_row                            average, sdev: time = 3.144 sec, 0.007; mem = 0.333, 0.471
# make_XTX_from_matmul_XT_and_X                     average, sdev: time = 5.219 sec, 0.021; mem = 0.000, 0.000
# END NUMPY ######################################################################################################
##################################################################################################################
'''




'''
##################################################################################################################
# SPARSE DICT ######################################################################################################

def sd_unzip_make_XTX_from_X_as_row(X):
    # 2) UNZIP TO NUMPY, DO SYMMETRIC
    inner_len = sd.inner_len_quick(X)
    X = sd.unzip_to_ndarray_float64(X)[0]
    XTX = {int(_):{} for _ in range(inner_len)}
    for inner_idx1 in range(inner_len):
        for inner_idx2 in range(inner_idx1+1):
            dot = np.matmul(X[:, inner_idx1], X[:, inner_idx2])
            if dot != 0:
                XTX[int(inner_idx1)][int(inner_idx2)] = dot
                XTX[int(inner_idx2)][int(inner_idx1)] = dot

    X = sd.zip_list_as_py_float(X)
    for outer_idx in XTX:
        if inner_len - 1 not in XTX[outer_idx]:
            XTX[int(outer_idx)][int(inner_len-1)] = 0

    return XTX

def sd_make_XTX_from_matmul_XT_and_X(X):
    # 3) BUILD A TOSS-AWAY TRANSPOSE AND DO core_symmetric_matmul
    return sd.core_symmetric_matmul(sd.sparse_transpose(X), X, DICT2_TRANSPOSE=sd.sparse_transpose(X))

def sd_make_XTX_from_X_as_column(X):
    # 1) TRANSPOSE, SYMMETRIC, TRANSPOSE
    X = sd.sparse_transpose(X)

    outer_len = sd.outer_len(X)
    XTX = {int(_): {} for _ in range(outer_len)}
    for outer_idx1 in range(outer_len):
        for outer_idx2 in range(outer_idx1+1):
            dot = sd.core_dot({0: X[outer_idx1]}, {0: X[outer_idx2]})
            if dot != 0:
                XTX[int(outer_idx1)][int(outer_idx2)] = dot
                XTX[int(outer_idx2)][int(outer_idx1)] = dot

    for outer_idx in XTX:
        if outer_len - 1 not in XTX[outer_idx]:
            XTX[int(outer_idx)][int(outer_len - 1)] = 0

    X = sd.sparse_transpose(X)   # PUT IT BACK LIKE IF IN MLObject CLASS
    return XTX


fxn = crsn.create_random_sparse_numpy
ctr = 0
for OBJ in (fxn(0,10,(3,10),50,np.int32), fxn(-10,10,(10,3),50,np.float64), fxn(-5,6,(100,100),50,np.int32)):
    ctr+=1
    print(f'Running test {ctr}...')
    OBJ = sd.zip_list_as_py_float(OBJ)

    XTX_1 = sd_make_XTX_from_X_as_column(OBJ)
    XTX_2 = sd_unzip_make_XTX_from_X_as_row(OBJ)
    XTX_3 = sd_make_XTX_from_matmul_XT_and_X(OBJ)

    if not sd.core_sparse_equiv(XTX_1, XTX_2) or not sd.core_sparse_equiv(XTX_1, XTX_3):
        print(f'1 = ')
        print(XTX_1)
        print()
        print(f'2 = ')
        print(XTX_2)
        print()
        print(f'3 = ')
        print(XTX_3)
        print()

        raise Exception(f'ERROR')

else:
    print(f'ALL GOOD')


X = crsn.create_random_sparse_numpy(-10, 10, (50, 5000), 90, np.int32)
X = sd.zip_list_as_py_float(X)

tmt.time_memory_tester(
                          (f'sd_make_XTX_from_X_as_column', sd_make_XTX_from_X_as_column, [X], {}),
                          (f'sd_unzip_make_XTX_from_X_as_row', sd_unzip_make_XTX_from_X_as_row, [X], {}),
                          (f'sd_make_XTX_from_matmul_XT_and_X', sd_make_XTX_from_matmul_XT_and_X, [X], {}),
                          number_of_trials=5,
                          rest_time=2
                          )


# FOR X GIVEN AS NP AS ROW OR COLUMN:
# SYMMETRIC BUILD USING [:, idx] AND [idx, :] IS ALWAYS FASTER THAN matmul(X.transpose(), X) AND matmul(X, X.transpose())

# FOR X GIVEN AS SD AS ROW OR COLUMN: 
# THE SUMMARY SEEMS TO BE THAT THERE IS NOT AN OUTRIGHT WINNER, THERE ARE SOME TRADE-OFFS DEPENDING ON SPARSITY,
# SIZE, & SHAPE. IT SEEMS THAT UNZIP/SYMMETRIC BUILD XTX/ZIP IS MORE OFTEN FASTER AND IS ALWAYS COMPETITIVE,
# COMPARED W core_symmetric_matmul APPROACHES.

# m, n = (50, 5000), 90% SPARSITY (SHOULD NOT EVER BE DOING XTX ON MORE COLUMNS THAN ROWS THO!)
# sd_make_XTX_from_X_as_column                      average, sdev: time = 52.780 sec, 2.430; mem = 622.000, 48.792
# sd_unzip_make_XTX_from_X_as_row                   average, sdev: time = 89.646 sec, 3.841; mem = 566.000, 128.696
# sd_make_XTX_from_matmul_XT_and_X                  average, sdev: time = 37.775 sec, 2.475; mem = 623.333, 48.321

# m, n = (500, 500), 90% SPARSITY
# sd_make_XTX_from_X_as_column                      average, sdev: time = 1.833 sec, 0.341; mem = 12.333, 0.471
# sd_unzip_make_XTX_from_X_as_row                   average, sdev: time = 1.781 sec, 0.045; mem = 13.333, 0.471
# sd_make_XTX_from_matmul_XT_and_X                  average, sdev: time = 1.384 sec, 0.046; mem = 11.333, 0.471

# m,n = (25000, 250), 90% SPARSE
# sd_make_XTX_from_X_as_column                      average, sdev: time = 19.120 sec, 0.474; mem = 5.333, 0.471
# sd_unzip_make_XTX_from_X_as_row                   average, sdev: time = 13.852 sec, 0.363; mem = 3.667, 1.700
# sd_make_XTX_from_matmul_XT_and_X                  average, sdev: time = 18.110 sec, 0.470; mem = -1.000, 0.816

# m,n = (5000, 50) 90% SPARSE
# sd_make_XTX_from_X_as_column                      average, sdev: time = 0.214 sec, 0.011; mem = 0.000, 0.000
# sd_unzip_make_XTX_from_X_as_row                   average, sdev: time = 0.456 sec, 0.018; mem = 0.000, 0.000
# sd_make_XTX_from_matmul_XT_and_X                  average, sdev: time = 0.231 sec, 0.012; mem = 0.000, 0.000

# m,n = (50, 5000) 50% SPARSE TAKING FOREVER, STOPPED SHORT.
# IN THE REAL WORLD SHOULDNT BE TRYING TO MAKE XTX IN THIS SITUATION ANYWAY

# m,n = (5000, 50)  50% SPARSE
# sd_make_XTX_from_X_as_column                      average, sdev: time = 1.410 sec, 0.016; mem = 1.000, 0.000
# sd_unzip_make_XTX_from_X_as_row                   average, sdev: time = 0.505 sec, 0.005; mem = 0.333, 0.471
# sd_make_XTX_from_matmul_XT_and_X                  average, sdev: time = 1.436 sec, 0.019; mem = 0.000, 0.000

# m,n = (500, 500), 50% SPARSE
# sd_make_XTX_from_X_as_column                      average, sdev: time = 11.062 sec, 0.024; mem = 12.667, 0.471
# sd_unzip_make_XTX_from_X_as_row                   average, sdev: time = 1.809 sec, 0.040; mem = 12.667, 0.943
# sd_make_XTX_from_matmul_XT_and_X                  average, sdev: time = 12.395 sec, 0.194; mem = 11.333, 0.471

# END SPARSE DICT ######################################################################################################
########################################################################################################################


'''








if __name__ == '__main__':
    pass























