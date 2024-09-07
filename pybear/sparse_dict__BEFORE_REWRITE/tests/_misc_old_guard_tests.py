import numpy as np
import time
from debug import timer



def t_break(start_time, t_cutoff, alg_name):
    if time.time() - start_time > t_cutoff:
        print(f'{alg_name} timed out after {time.time() - start_time} seconds.')
        return True
    else: return False

@timer.timer     # #rows, cols, sparsity
def random_create(rows, cols, sparsity):
    RANDOM_SPARSE = create_random(rows,cols,sparsity)
    return RANDOM_SPARSE

@timer.timer
def list_to_sparsedict(LIST):
    SPARSEDICT = zip_list(LIST)
    return SPARSEDICT

@timer.timer
def to_list(SPARSE_DICT):
    LIST = unzip_to_list(SPARSE_DICT)[0]
    return LIST

@timer.timer
def create_y():
    return [1 if np.random.randint(0, 2) == 1 else -1 for _ in range(rows)]

@timer.timer
def create_alphas():
    return [np.abs(np.random.randn()) for _ in range(rows)]

@timer.timer
def by_matmul(DATA, d, exp):
    K1 = np.matmul(DATA, DATA.transpose())
    K1 = (K1 + d) ** exp
    return K1

@timer.timer
def individual_matmuls(DATA, rows, mode, d, exp):
    K = np.zeros([rows, rows])
    t1 = time.time()
    if mode == 'S':
        for idx1 in range(rows):
            if idx1 % 10 == 0:
                if t_break(t1, t_cutoff, 'CREATE_DATA'): break
            for idx2 in range(idx1 + 1):
                DOT = (np.matmul(DATA[idx1], DATA[idx2]) + d) ** exp
                K[idx1][idx2] = DOT
                K[idx2][idx1] = DOT
    if mode != 'S':
        for idx1 in range(rows):
            if idx1 % 10 == 0:
                if t_break(t1, t_cutoff, 'CREATE_DATA'): break
            for idx2 in range(rows):
                DOT = (np.matmul(DATA[idx1], DATA[idx2]) + d) ** exp
                K[idx1][idx2] = DOT

    return K

@timer.timer
def by_core_dot(SPARSEDICT, d, exp, mode):

    rows = outer_len(SPARSEDICT)

    K = np.zeros([rows, rows])
    if mode == 'S':
        for idx1 in range(rows):
            for idx2 in range(idx1 + 1):
                DOT = (core_dot(DICT1={0: SPARSEDICT[idx1]}, DICT2={0: SPARSEDICT[idx2]}) + d) ** exp
                K[idx1][idx2] = DOT
                K[idx2][idx1] = DOT
    if mode != 'S':
        for idx1 in range(rows):
            for idx2 in range(rows):
                DOT = (core_dot(DICT1={0: SPARSEDICT[idx1]}, DICT2={0: SPARSEDICT[idx2]}) + d) ** exp
                K[idx1][idx2] = DOT

    return K


@timer.timer
def manual_sparse_dots(SPARSEDICT, d, exp, mode):

    rows = outer_len(SPARSEDICT)

    K = np.zeros([rows, rows])

    if mode == 'S':
        for idx1 in range(rows):
            for idx2 in range(idx1 + 1):
                DOT = 0
                for inner_key in SPARSEDICT[idx1]:
                    if inner_key in SPARSEDICT[idx2]:
                        DOT += SPARSEDICT[idx1][inner_key] * SPARSEDICT[idx2][inner_key]
                DOT = (DOT + d) ** exp
                K[idx1][idx2] = DOT
                K[idx2][idx1] = DOT
    if mode != 'S':
        for idx1 in range(rows):
            for idx2 in range(rows):
                DOT = 0
                for inner_key in SPARSEDICT[idx1]:
                    if inner_key in SPARSEDICT[idx2]:
                        DOT += SPARSEDICT[idx1][inner_key] * SPARSEDICT[idx2][inner_key]
                DOT = (DOT + d) ** exp
                K[idx1][idx2] = DOT

    return K

@timer.timer
def by_sparse_AAT(SPARSEDICT, d, exp):
    K = sparse_AAT(SPARSEDICT)
    K = unzip_to_list(K)[0]
    K = (np.array(K) + d)**exp

    return K

@timer.timer
def calc_W(K5, ALPHAS, Y):

    return np.sum(ALPHAS) + np.sum(np.outer(Y, Y) * np.outer(ALPHAS, ALPHAS) * K5)



cols = vui.validate_user_int(f'\nEnter number of columns > ', min=1)
rows = vui.validate_user_int(f'Enter number of rows > ', min=1)
t_cutoff = 6000
sparsity_ = vui.validate_user_int(f'Enter sparsity > ', min=0, max=100)
mode = {'S':'S', 'N': 'NS'}[vui.validate_user_str(f'Enter mode symmetric (s) or not symmetric (n) > ', 'NS')]
# K_poly PARAMETERS
d = vui.validate_user_float(f'Enter d > ')
exp = vui.validate_user_float(f'Enter exp > ')


#### DATA FEED #####################################################################################################
print(f'\nCREATE DATA BY create_random() AND UNZIP TO LIST BY unzip_to_list()')
SPARSE_DICT = random_create(rows, cols, sparsity_)
DATA = to_list(SPARSE_DICT)
DATA = nparray(DATA, dtype=object)
####################################################################################################################

#### SPARSE DICT FEED ##############################################################################################
print(f'\nCREATE SPARSEDICT BY ZIPPING DATA BACK WITH zip_list()')
SPARSEDICT = list_to_sparsedict(DATA)
####################################################################################################################

#### Y FEED ########################################################################################################
print(f'\nCREATE Y WITH LIST COMPREHENSION')
Y = create_y()
#### END Y FEED ########################################################################################################

#### ALPHA FEED ########################################################################################################
print(f'\nCREATE ALPHAS WITH LIST COMPREHENSION')
ALPHAS = create_alphas()
#### END ALPHA FEED ##################################################################################################

#### K BY MATMUL ##################################################################################################
print(f'\nCREATE K_poly BY MATMUL (K1)')
K1 = by_matmul(DATA, d, exp)

del K1
#### END K BY MATMUL ##################################################################################################

'''
#### K BY INDIVIDUAL MATMULS ##################################################################################################
print(f'\nCREATE K_poly BY INDIVIDUAL MATMULS (K2)')
K2 = individual_matmuls(DATA, rows, mode, d, exp)

del K2
#### END K BY INDIVIDUAL MATMULS ##################################################################################################
'''

#### K BY core_dot #######################################################################################################
print(f'\nCREATE K_poly BY core_dot (K3)')
K3 = by_core_dot(SPARSEDICT, d, exp, mode)

del K3
#### END K BY core_dot ##################################################################################################

#### K BY manual_sparse_dots ##################################################################################################
print(f'\nCREATE K_poly BY manual_sparse_dots (K4)')
K4 = manual_sparse_dots(SPARSEDICT, d, exp, mode)

del K4
#### END K BY manual_sparse_dots ##################################################################################################

#### K BY sparse_AAT ####################################################################################################
print(f'\nCREATE K_poly BY sparse_AAT (K5)')

K5 = by_sparse_AAT(SPARSEDICT, d, exp)

#### END K BY sparse_AAT ####################################################################################################

'''
K_LIST = [K1, K2, K3, K4, K5]
for k1 in range(len(K_LIST)):
    for k2 in range(k1+1):
        if not nparray_equiv(K_LIST[k1], K_LIST[k2]):
            raise AssertionError(f'K{k1+1} and K{k2+1} ARE NOT EQUAL :(')
print(f'\nALL Ks ARE EQUAL!!!!\n')
'''

# CALCULATE W
print(f'\nCALCULATE W')
W = calc_W(K5, ALPHAS, Y)
del K5
print('\nW = ', W)


