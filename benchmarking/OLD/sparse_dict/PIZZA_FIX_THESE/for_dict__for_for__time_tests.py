from general_data_ops.create_random_sparse_numpy import create_random_sparse_numpy as crsn, \
    build_serialized_posns as bspv
from pybear.data_validation import validate_user_input as vui
import numpy as np, pandas as pd
import sparse_dict as sd
from debug import time_memory_tester as tmt
from read_write_file.generate_full_filename import base_path_select as bps


# THIS MAKES ALL DATAFRAME HEADERS AND INDEXES "SPARSE"
pd.set_option('display.multi_sparse', True, 'display.colheader_justify', 'center')
pd.set_option('display.max_columns', None, 'display.width', 150)
pd.options.display.float_format = '{:,.3f}'.format


# PIZZA NOTES 11/24/22  THIS GOT BUTCHERED AFTER CHANGING serialized FROM RETURNING IDXS AND VALUES TO JUST RETURNING
# IDXS. WILL NEED A LOT OF WORK.  ALSO, ITERATING BY for/for IS ALWAYS WAAYYY SLOWER THAN EXTRACTING FULLY SIZED NP
# ROW BY ROW TO SPARSE_DICT, THAT FUNCTIONALITY IS DEFUNCT AND THIS WILL NEED MORE WORK. ESSENTIALLY NOW ALL FORMS OF
# create_random_sparse_dict USE A FULLY SIZED NUMPY ARRAY WITH MASK APPLIED, AND MASK IS CREATED BY NUMBER FILTER,
# _random_.choice ON [0,1] W p GIVEN BY SPARSITY, OR BY SERIALIZED IDXS MAPPED TO A NDARRAY


def serialized_idxs_by_for_for(RAND_SERIALIZED_DENSE_POSNS, RAND_SERIALIZED_VALUES,
                               _len_outer, _len_inner, _dtype, as_py_or_np):
    # METHOD 1 --- IF INNER DICTS ARE SHORTER THAN OUTER LEN, FILL BY FOR LOOPS (dict(()) METHOD IS MUCH SLOWER HERE)

    RAND_SERIALIZED_DENSE_POSNS = RAND_SERIALIZED_DENSE_POSNS.tolist()

    if as_py_or_np == 'py':
        RAND_SERIALIZED_VALUES = RAND_SERIALIZED_VALUES.tolist()
        if 'int' in str(_dtype): return_type = int
        elif 'float' in str(_dtype): return_type = float
    elif as_py_or_np == 'np':
        return_type = _dtype

    SPARSE_DICT = {int(_): {} for _ in range(_len_outer)}
    for ser_loc_idx in range(len(RAND_SERIALIZED_DENSE_POSNS)):
        SPARSE_DICT[int(RAND_SERIALIZED_DENSE_POSNS[ser_loc_idx] // _len_inner)][
            int(RAND_SERIALIZED_DENSE_POSNS[ser_loc_idx] % _len_inner)] = \
            return_type(RAND_SERIALIZED_VALUES[ser_loc_idx])  # APPLY USER DEFINED VALUE DTYPE

    del RAND_SERIALIZED_DENSE_POSNS, RAND_SERIALIZED_VALUES

    # ADD PLACEHOLDERS  --- CORRECT ORDERING IS GUARANTEED BY SORTED SERIAL LOCATION
    for outer_key in SPARSE_DICT:
        if _len_inner - 1 not in SPARSE_DICT[outer_key]:
            SPARSE_DICT[int(outer_key)][int(_len_inner - 1)] = return_type(0)

    return SPARSE_DICT


def serialized_idxs_and_values_by_dict(RAND_SERIALIZED_DENSE_POSNS, RAND_SERIALIZED_VALUES,
                                       _len_outer, _len_inner, _dtype, as_py_or_np):
    # METHOD 2 --- IF INNER DICTS ARE LONGER THAN OVERALL LEN, FILL BY dict(()) METHOD (MUCH FASTER THAN for LOOPS)

    OUTER_KEYS = RAND_SERIALIZED_DENSE_POSNS // _len_inner
    INNER_KEYS = RAND_SERIALIZED_DENSE_POSNS % _len_inner
    del RAND_SERIALIZED_DENSE_POSNS

    SPARSE_DICT = {}
    for outer_key in range(_len_outer):
        ACTIVE_OUTER_KEYS = np.argwhere(OUTER_KEYS == outer_key).reshape((1,-1))[0]
        ACTIVE_VALUES = RAND_SERIALIZED_VALUES[ACTIVE_OUTER_KEYS]    # DONT NEED TO SET dtype, SHOULD HAVE BEEN SET AT CREATION

        # IF OUTPUT IS TO BE NP, KEEP AS NDARRAY, VALUES ARE PUT IN DICT AS NP VALUES. OTHERWISE tolist FORCES
        # np ints AND np floats TO py ints AND py floats.  ALWAYS FORCE KEYS TO py int!!!
        if as_py_or_np == 'py': ACTIVE_VALUES = ACTIVE_VALUES.tolist()

        SPARSE_DICT[int(outer_key)] = dict((
            zip(INNER_KEYS[ACTIVE_OUTER_KEYS].astype(np.int32).tolist(),
                ACTIVE_VALUES)
        ))

    del RAND_SERIALIZED_VALUES, OUTER_KEYS, INNER_KEYS, ACTIVE_OUTER_KEYS, ACTIVE_VALUES

    # ADD PLACEHOLDERS  --- CORRECT ORDERING IS GUARANTEED BY SORTED SERIAL LOCATION
    if as_py_or_np == 'py':
        if 'int' in str(_dtype): return_type = int
        elif 'float' in str(_dtype): return_type = float
    elif as_py_or_np == 'np':
        return_type = _dtype

    for outer_key in SPARSE_DICT:
        if _len_inner - 1 not in SPARSE_DICT[outer_key]:
            SPARSE_DICT[int(outer_key)][int(_len_inner - 1)] = return_type(0)

    return SPARSE_DICT


def fully_sized_numpy_by_for_for(VALUES, _len_outer, _len_inner, _dtype, as_py_or_np):
    # METHOD 1 --- dict(()) IF INNER DICTS ARE SHORTER THAN OVERALL LEN, FILL BY FOR LOOPS (dict(()) METHOD IS MUCH SLOWER HERE)

    if as_py_or_np == 'py':
        if 'int' in str(_dtype):
            return_type = int
        elif 'float' in str(_dtype):
            return_type = float
    elif as_py_or_np == 'np':
        return_type = _dtype

    SPARSE_DICT = {}
    for outer_key in range(_len_outer):
        SPARSE_DICT[int(outer_key)] = {}
        VALUES_AS_LIST = VALUES[outer_key].tolist()  # TO ALLOW FOR py INT OR FLOAT TO BE APPLIED (AS np WONT ALLOW)
        for inner_key in np.nonzero(VALUES[outer_key])[-1].tolist():  # TO ENSURE DICT KEYS ARE py INTS, NOT np INTS
            SPARSE_DICT[int(outer_key)][int(inner_key)] = return_type(VALUES_AS_LIST[inner_key])  # APPLY NUMBER FORMAT

    del VALUES_AS_LIST, VALUES

    # ADD PLACEHOLDERS
    for outer_key in SPARSE_DICT:
        if _len_inner - 1 not in SPARSE_DICT[outer_key]:
            SPARSE_DICT[int(outer_key)][int(_len_inner - 1)] = return_type(0)

    return SPARSE_DICT


def fully_sized_numpy_by_dict(VALUES, _len_outer, _len_inner, _dtype, as_py_or_np):
    # METHOD 2 --- IF INNER DICTS ARE LONGER THAN OVERALL LEN, FILL BY dict(()) METHOD (FASTER HERE THAN for LOOPS)

    SPARSE_DICT = {}
    for outer_key in range(_len_outer):
        NON_ZERO_KEYS = np.nonzero(VALUES[outer_key])[-1]
        NON_ZERO_VALUES = VALUES[outer_key][NON_ZERO_KEYS]

        if as_py_or_np != 'np': NON_ZERO_VALUES = NON_ZERO_VALUES.tolist()

        SPARSE_DICT[int(outer_key)] = dict((zip(NON_ZERO_KEYS.tolist(), NON_ZERO_VALUES)))

    del VALUES, NON_ZERO_VALUES, NON_ZERO_KEYS

    # ADD PLACEHOLDERS
    if as_py_or_np == 'py':
        if 'int' in str(_dtype):
            return_type = int
        elif 'float' in str(_dtype):
            return_type = float
    elif as_py_or_np == 'np':
        return_type = _dtype
    
    for outer_key in SPARSE_DICT:
        if _len_inner - 1 not in SPARSE_DICT[outer_key]:
            SPARSE_DICT[int(outer_key)][int(_len_inner - 1)] = return_type(0)

    return SPARSE_DICT




def print_example(SPARSE_DICT):
    [print(f'{_}: {[(key,value) for key,value in list(SPARSE_DICT[_].items())[:5]]}') for _ in range(5)]
    print(f'outer key types = {(np.unique([str(type(_)) for _ in SPARSE_DICT]))}')
    print(f'inner key types = {(np.unique([str(type(_)) for _ in SPARSE_DICT[0]]))}')


def exception_handle_test_fail(fxn, attr, constraint, act_value, expected_value):
    print(f'\n{fxn} wants to Except for actual {attr} ({act_value}) not {constraint} expected value of {expected_value}.')
    if vui.validate_user_str(f'Quit(q) continue(c) > ', 'QC') == 'Q':
        raise Exception(f'\n{fxn}: actual {attr} ({act_value}) not {constraint} expected value of {expected_value}.')


def exception_handling(fxn, jargon):
    raise Exception(f'\n*** {fxn}: {jargon} ***\n')


def test_shape_dtype_min_max_spar(fxn, SPARSE_DICT, outer_size, inner_size, _dtype, as_py_or_np, _min, _max, _sparsity):
    # TEST SHAPE #######################################################################################################
    if sd.outer_len(SPARSE_DICT) != outer_size:
        exception_handle_test_fail(fxn, f'outer_len', 'equal to', sd.outer_len(SPARSE_DICT), outer_size)
    if sd.inner_len(SPARSE_DICT) != inner_size:
        exception_handle_test_fail(fxn, f'inner_len', 'equal to', sd.inner_len(SPARSE_DICT), inner_size)
    # END TEST SHAPE #######################################################################################################

    # TEST DTYPES #######################################################################################################
    # GET / VERIFY OUTER KEY DTYPE   (MUST ALWAYS BE py_int)
    RAW_OUTER_KEY_DTYPES = np.unique(np.fromiter((str(type(_)) for _ in SPARSE_DICT.keys()), dtype='<U100'))
    number_of_outer_key_dtypes = len(RAW_OUTER_KEY_DTYPES)
    if number_of_outer_key_dtypes > 1:
        exception_handle_test_fail(fxn, 'number of outer key dtypes', 'equal to', number_of_outer_key_dtypes, 1)
    else:
        raw_outer_key_dtype = RAW_OUTER_KEY_DTYPES[0]
        if raw_outer_key_dtype != "<class 'int'>":
            exception_handle_test_fail(fxn, 'outer key dtype', 'equal to', raw_outer_key_dtype, "<class 'int'>")
        else:
            del RAW_OUTER_KEY_DTYPES, number_of_outer_key_dtypes

    # GET / VERIFY INNER KEY DTYPE   (MUST ALWAYS BE py_int)
    RAW_INNER_KEY_DTYPES = []
    for outer_key in SPARSE_DICT:
        RAW_INNER_KEY_DTYPES += [str(type(_)) for _ in SPARSE_DICT[outer_key].keys()]
    RAW_INNER_KEY_DTYPES = np.unique(RAW_INNER_KEY_DTYPES)
    number_of_inner_key_dtypes = len(RAW_INNER_KEY_DTYPES)
    if number_of_inner_key_dtypes > 1:
        exception_handle_test_fail(fxn, 'number of inner key dtypes', 'equal to', number_of_inner_key_dtypes, 1)
    else:
        raw_key_dtype = RAW_INNER_KEY_DTYPES[0]
        if raw_key_dtype != "<class 'int'>":
            exception_handle_test_fail(fxn, 'inner key dtype', 'equal to', raw_key_dtype, "<class 'int'>")
        else:
            del RAW_INNER_KEY_DTYPES, number_of_inner_key_dtypes

    # GET INNER VALUE DTYPE
    RAW_INNER_VALUE_DTYPES = []
    for outer_key in SPARSE_DICT:
        RAW_INNER_VALUE_DTYPES += [str(type(SPARSE_DICT[outer_key][inner_key])) for inner_key in SPARSE_DICT[outer_key].keys()]
    RAW_INNER_VALUE_DTYPES = np.unique(RAW_INNER_VALUE_DTYPES)
    number_of_inner_value_dtypes = len(RAW_INNER_VALUE_DTYPES)
    if number_of_inner_value_dtypes > 1:
        exception_handle_test_fail(fxn, 'number of inner value dtypes', 'equal to', number_of_inner_value_dtypes, 1)
    else:
        raw_value_dtype = RAW_INNER_VALUE_DTYPES[0]
        if 'numpy.int8' in raw_value_dtype: act_value_dtype = 'np_int8'
        elif 'numpy.int32' in raw_value_dtype: act_value_dtype = 'np_int32'
        elif 'numpy.float64' in raw_value_dtype: act_value_dtype = 'np_float64'
        elif raw_value_dtype == "<class 'int'>": act_value_dtype = 'py_int'
        elif raw_value_dtype == "<class 'float'>": act_value_dtype = 'py_float'
        else: exception_handling(fxn, f'act_value_dtype LOGIC IS FAILING')

        del RAW_INNER_VALUE_DTYPES, number_of_inner_value_dtypes

    # GET EXPECTED INNER VALUE DTYPE
    if as_py_or_np == 'py':
        if 'int' in str(_dtype): exp_value_dtype = "py_int"
        elif 'float' in str(_dtype): exp_value_dtype = "py_float"
        else: exception_handling(f'exp_value_dtype LOGIC IS FAILING FOR as_py_or_np == py')
    elif as_py_or_np == 'np':
        if 'int8' in str(_dtype): exp_value_dtype = "np_int8"
        elif 'int32' in str(_dtype): exp_value_dtype = "np_int32"
        elif 'float64' in str(_dtype): exp_value_dtype = "np_float64"
        else: exception_handling(f'exp_value_dtype LOGIC IS FAILING FOR as_py_or_np == np')

    # VERIFY INNER VALUE DTYPE
    if act_value_dtype != exp_value_dtype:
        exception_handle_test_fail(fxn, 'inner value dtype', 'equal to', act_value_dtype, exp_value_dtype)

    # END TEST DTYPES #######################################################################################################

    # TEST MIN/MAX ########################################################################################################
    # TEST MIN/MAX INTEGER ################EEEEEEEEEEEEEEEEEE################################################################
    act_min = sd.min_(SPARSE_DICT)
    act_max = sd.max_(SPARSE_DICT)

    if 'bin' in exp_value_dtype: exp_min, exp_max = 0, 1
    elif 'int' in exp_value_dtype: exp_min, exp_max = _min, _max - 1
    elif 'float' in exp_value_dtype: exp_min, exp_max = _min, _max
    if exp_value_dtype == 'create_random': exp_min, exp_max = 0, 9
    if _sparsity == 0:
        if 'bin' in exp_value_dtype: exp_min = 1
        elif exp_value_dtype == 'create_random': exp_min = 1
        elif 'int' in exp_value_dtype: exp_min = _min if _min != 0 else 1
        elif 'float' in exp_value_dtype: exp_min = _min
        else: raise Exception(f'\n*** LOGIC FOR SETTING MIN WHEN SPARSITY IS ZERO IS FAILING ***\n')
    elif _sparsity > 0 and _sparsity < 100: exp_min = 0
    elif _sparsity == 100: exp_min, exp_max = 0, 0

    if True in map(lambda x: x in exp_value_dtype, ('int', 'bin')):
        if act_min != exp_min: exception_handle_test_fail(fxn, '_min', 'equal to', act_min, exp_min)
        if act_max != exp_max: exception_handle_test_fail(fxn, '_max', 'equal to', act_max, exp_max)
    # END TEST MIN/MAX INTEGER #############EEEEEEEEEEEEEEEEEe###################################################################

    # TEST MIN/MAX FLOAT ######EEEEEEEEEEEEEEEE#############################################################################
    elif 'float' in exp_value_dtype:
        min_max_window = .1
        if act_min < exp_min or act_min > exp_min + min_max_window: exception_handling('_min', 'within window of', act_min, exp_min)
        if act_max > exp_max or act_max < exp_max - min_max_window: exception_handling('_max', 'within window of', act_max, exp_max)
    # END TEST MIN/MAX FLOAT ##################EEEEEEEEEEEEE#################################################################
    else:
        raise Exception(f'\n*** LOGIC FOR TESTING exp_min/exp_max VS act_min/act_max IS FAILING ***/n')
    # END TEST MIN/MAX ########################################################################################################

    # TEST SPARSITY #####################################################################################################
    act_sparsity = sd.sparsity(SPARSE_DICT)
    sparsity_window = 1
    if act_sparsity > _sparsity + sparsity_window or act_sparsity < _sparsity - sparsity_window:
        exception_handle_test_fail('sparsity', 'within range of', act_sparsity, _sparsity)
    # END TEST SPARSITY #####################################################################################################





HEADER = pd.MultiIndex(
            levels=[ ['np_int8', 'np_int32', 'np_float64'],
                     ['np', 'py'] ],
            codes=[[0, 0, 1, 1, 2, 2], [0, 1, 0, 1, 0, 1]] )

INDEX = pd.MultiIndex(
            levels=[ ['1,000,000', '100,000,000'],
                     ['100x10000', '1000x1000', '10000x100', '1000x100000', '10000x10000', '100000x1000'],
                     ['1','2','3','4']],
            codes=[[0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1],
                   [0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5],
                   [0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3]]
)

TIME_RESULTS = pd.DataFrame(columns=HEADER, index=INDEX).fillna('-')
MEM_RESULTS = TIME_RESULTS.copy()





_min, _max = 0, 2    # JUST USE BIN FOR THIS TEST, KEEP IT SIMPLE
_sparsity = 90
ctr = 0
total_trials = 2*3*3*2
for x in (1,100): #(1,10,100):

    for _dtype in (np.int8, np.int32, np.float64):
        # if ctr > 12: continue    # PIZZA
        if _dtype == np.int8: header_level_1 = 'np_int8'
        elif _dtype == np.int32: header_level_1 = 'np_int32'
        elif _dtype == np.float64: header_level_1 = 'np_float64'
        else: exception_handling('main tests___PIZZA_FIX_THESE module', 'LOGIC TO GET header_level_1 FROM _dtype IS FAILING')

        total_size = 1000000 * x

        if total_size == 1000000: index_level_1 = '1,000,000'
        elif total_size == 100000000: index_level_1 = '100,000,000'
        else: exception_handling('main tests___PIZZA_FIX_THESE module', 'LOGIC TO GET index_level_1 FROM total_size IS FAILING')

        # JUST MAKE THESE 1 x total_size.... JUST RESHAPE THE np array, WONT HAVE T0 CHANGE SERIALIZED
        print(f'\nBUILDING VALUES FOR full_numpy_w_mask...')
        VALUES = crsn(_min, _max, (1, total_size), _sparsity, _dtype)
        print(f'Done.')

        print(f'\nBUILDING RAND_SERIALIZED POSNS & VALUES FOR serial...')
        RAND_SERIALIZED_DENSE_POSNS = bspv(0, 2, (1, total_size), _dtype)
        print(f'Done.')

        for outer_size in (np.sqrt(total_size)/10, np.sqrt(total_size), np.sqrt(total_size)*10):    #(10*x, 25*x, 40*x, 50*x, 100*x, 250*x, 400*x, 500*x, 1000*x):

            outer_size = int(outer_size)   # BLOWING UP THAT np CANT RESHAPE WHEN SHAPE IS FLOAT

            if total_size == 1000000 and outer_size == 100: index_level_2 = '100x10000'
            elif total_size == 1000000 and outer_size == 1000: index_level_2 = '1000x1000'
            elif total_size == 1000000 and outer_size == 10000: index_level_2 = '10000x100'
            elif total_size == 100000000 and outer_size == 1000: index_level_2 = '1000x100000'
            elif total_size == 100000000 and outer_size == 10000: index_level_2 = '10000x10000'
            elif total_size == 100000000 and outer_size == 100000: index_level_2 = '100000x1000'
            else: exception_handling('main tests___PIZZA_FIX_THESE module', 'LOGIC TO GET index_level_2 FROM outer_size IS FAILING')


            for as_py_or_np in ('py', 'np'):
                ctr += 1

                # if ctr > 12: continue     # PIZZA

                header_level_2 = as_py_or_np

                inner_size = int(total_size/outer_size)   # BLOWING UP THAT np CANT RESHAPE WHEN SHAPE IS FLOAT

                print(f'Running trial {ctr} of {total_trials}...')
                print(f'Produce sparse dict from np {_dtype}, return as {as_py_or_np}')
                print(f'total_size = {total_size:,.0f}, outer_size = {outer_size:,.0f}, inner_size = {inner_size: ,.0f}')

                '''
                ##########################################################################################################
                #### TEST FOR CORRECTNESS OF SPARSE_DICT AND EQUALITY OF THE METHODS #####################################
                #### DO THIS BEFORE RUNNING SPEED / MEM TESTS ############################################################
                SPARSE_DICT1 = serialized_idxs_by_for_for(RAND_SERIALIZED_DENSE_POSNS, RAND_SERIALIZED_VALUES,
                                               outer_size, inner_size, _dtype, as_py_or_np)
                test_shape_dtype_min_max_spar('serialized_idxs_by_for_for', SPARSE_DICT1, outer_size, inner_size,
                                              _dtype, as_py_or_np, _min, _max, _sparsity)

                SPARSE_DICT2 = serialized_idxs_and_values_by_dict(RAND_SERIALIZED_DENSE_POSNS, RAND_SERIALIZED_VALUES,
                                                       outer_size, inner_size, _dtype, as_py_or_np)

                test_shape_dtype_min_max_spar('serialized_idxs_and_values_by_dict', SPARSE_DICT2, outer_size, inner_size,
                                              _dtype, as_py_or_np, _min, _max, _sparsity)

                SPARSE_DICT3 = fully_sized_numpy_by_for_for(
                    VALUES.reshape((outer_size, inner_size)), outer_size, inner_size, _dtype, as_py_or_np)

                test_shape_dtype_min_max_spar('fully_sized_numpy_by_for_for', SPARSE_DICT3, outer_size, inner_size,
                                              _dtype, as_py_or_np, _min, _max, _sparsity)

                SPARSE_DICT4 = fully_sized_numpy_by_dict(
                    VALUES.reshape((outer_size, inner_size)), outer_size, inner_size, _dtype, as_py_or_np)

                test_shape_dtype_min_max_spar('fully_sized_numpy_by_dict', SPARSE_DICT4, outer_size, inner_size,
                                              _dtype, as_py_or_np, _min, _max, _sparsity)

                # METHODS BY SERIALIZED SHOULD BE EQUAL
                if not sd.core_sparse_equiv(SPARSE_DICT1, SPARSE_DICT2):
                    print(f'\n*** DISASTER.  METHOD1 != METHOD2 ***\n')
                # AND METHODS BY FULLY SIZED NUMPY SHOULD BE EQUAL
                if not sd.core_sparse_equiv(SPARSE_DICT3, SPARSE_DICT4):
                    print(f'\n*** DISASTER.  METHOD3 != METHOD4 ***\n')

                del SPARSE_DICT1, SPARSE_DICT2, SPARSE_DICT3, SPARSE_DICT4

                #### END TEST FOR CORRECTNESS OF SPARSE_DICT AND EQUALITY OF THE METHODS #################################
                ##########################################################################################################
                ##########################################################################################################
                '''

                ##########################################################################################################
                ##########################################################################################################
                # SPEED / MEM TEST #######################################################################################
                TIME_MEM = \
                    tmt.time_memory_tester(
                        ('serialized_idxs_by_for_for',
                         serialized_idxs_by_for_for,
                         [RAND_SERIALIZED_DENSE_POSNS, RAND_SERIALIZED_VALUES, outer_size, inner_size, _dtype, as_py_or_np],
                         {}),
                        ('serialized_idxs_and_values_by_dict',
                        serialized_idxs_and_values_by_dict,
                        [RAND_SERIALIZED_DENSE_POSNS, RAND_SERIALIZED_VALUES, outer_size, inner_size, _dtype, as_py_or_np],
                        {}),
                        ('fully_sized_numpy_by_dict',
                         fully_sized_numpy_by_dict,
                         [VALUES.reshape((outer_size, inner_size)), outer_size, inner_size, _dtype, as_py_or_np],
                         {}),
                        ('fully_sized_numpy_by_dict',
                         fully_sized_numpy_by_dict,
                         [VALUES.reshape((outer_size, inner_size)), outer_size, inner_size, _dtype, as_py_or_np],
                         {}),
                         number_of_trials=3,
                         rest_time=3
                         )

                TIMES = TIME_MEM[0].mean(axis=1).ravel()
                MEMS = TIME_MEM[1].mean(axis=1).ravel()

                for idx, index_level_3 in enumerate(['1','2','3','4']):
                    TIME_RESULTS.loc[(index_level_1, index_level_2, index_level_3),
                                                                    (header_level_1, header_level_2)] = TIMES[idx]

                    MEM_RESULTS.loc[(index_level_1, index_level_2, index_level_3),
                                                                    (header_level_1, header_level_2)] = MEMS[idx]

                # END SPEED / MEM TEST ###################################################################################
                ##########################################################################################################
                ##########################################################################################################


print(TIME_RESULTS)

print()

print(MEM_RESULTS)

base_path = bps.base_path_select()
full_path = base_path +  r'time_results.xlsx'


try:
    with pd.ExcelWriter(full_path) as writer:
        # index must be True, NotImplementedError: Writing to Excel with MultiIndex columns and no index ('index'=False) is not yet implemented.
        TIME_RESULTS.style.set_properties(**{'text-align': 'center'}).to_excel(excel_writer=writer,
                             sheet_name='TIME RESULTS', float_format='%.3f', startrow=1, startcol=1, merge_cells=False,
                             index=True, na_rep='NaN')

        MEM_RESULTS.style.set_properties(**{'text-align': 'center'}).to_excel(excel_writer=writer,
                             sheet_name='MEM RESULTS', float_format='%.3f', startrow=1, startcol=1, merge_cells=False,
                             index=True, na_rep='NaN')

except:
    with pd.ExcelWriter(full_path) as writer:
        # index must be True, NotImplementedError: Writing to Excel with MultiIndex columns and no index ('index'=False) is not yet implemented.
        TIME_RESULTS.style.set_properties(**{'text-align': 'center'}).to_excel(excel_writer=writer,
                sheet_name='TIME RESULTS', float_format='%.3f', startrow=1, startcol=1, merge_cells=False,
                index=True, na_rep='NaN')

        MEM_RESULTS.style.set_properties(**{'text-align': 'center'}).to_excel(excel_writer=writer,
                sheet_name='MEM RESULTS', float_format='%.3f', startrow=1, startcol=1, merge_cells=False,
                index=True, na_rep='NaN')





