# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

import numpy as np
import sparse_dict as sd
from data_validation import validate_user_input as vui



# ***** TEST MODULE FOR ALL create_random FUNCTIONS *****

# BEAR 24_04_24_10_26_00 NEED:
# sd.min_(RANDOM_SD)
# sd.max_(RANDOM_SD)
# sd.outer_len(RANDOM_SD)
# sd.inner_len(RANDOM_SD)
# sd.sparsity(RANDOM_SD)






# TEST MODULE FOR UPDATED sparse_dict.create_random FXNS       CIRCA NOV 2022

min_max_window = 0.01
sparsity_window = .5

FXNS = [sd.create_random, sd.create_random_py_bin, sd.create_random_py_int, sd.create_random_py_float,
        sd.create_random_np_bin, sd.create_random_np_int32, sd.create_random_np_float64]
MIN = [2 ,0 ,4 ,0]
MAX = [10 ,10 ,5 ,2]
OUTER = [80, 1200]
INNER = [1200, 80]
SPARSITY = [0, 25, 50, 75, 100]

FXN_NAMES = ['create_random', 'create_random_py_bin', 'create_random_py_int', 'create_random_py_float',
             'create_random_np_bin', 'create_random_np_int32', 'create_random_np_float64']
EXPECTED_KEY_TYPE = ['py_int', 'py_int', 'py_int', 'py_int', 'py_int', 'py_int', 'py_int']
EXPECTED_VALUE_TYPE = ['py_int', 'py_int', 'py_int', 'py_float', 'np_int8', 'np_int32', 'np_float64']

EXPECTED_KEY_DICT = dict((zip(FXN_NAMES, EXPECTED_KEY_TYPE)))
EXPECTED_VALUE_DICT = dict((zip(FXN_NAMES, EXPECTED_VALUE_TYPE)))



def exception_handle(attr, constraint, act_value, expected_value):
    print \
        (f'\nWants to Except for actual {attr} ({act_value}) not {constraint} expected value of {expected_value}.')
    if vui.validate_user_str(f'Quit(q) continue(c) > ', 'QC') == 'Q':
        raise Exception \
            (f'\nactual {attr} ({act_value}) not {constraint} expected value of {expected_value}.')


ctr = 0
for itr, FXN in enumerate(FXNS):
    # ASSIGN A FUNCTION NAME FOR EACH PASS
    __ = str(FXN)
    if ' create_random_py_bin ' in __: function_name = 'create_random_py_bin'
    elif ' create_random_py_int ' in __: function_name = 'create_random_py_int'
    elif ' create_random_py_float ' in __: function_name = 'create_random_py_float'
    elif ' create_random_np_bin ' in __: function_name = 'create_random_np_bin'
    elif ' create_random_np_int32 ' in __: function_name = 'create_random_np_int32'
    elif ' create_random_np_float64 ' in __: function_name = 'create_random_np_float64'
    elif ' create_random ' in __: function_name = 'create_random'
    else: raise Exception \
            (f'\n*** GETTING function_name FROM str(FXN) LOGIC IS FAILING ***\n')
    # END ASSIGN A FUNCTION NAME FOR EACH PASS
    for _min, _max in zip(MIN, MAX):
        for len_outer, len_inner in zip(OUTER, INNER):
            for _sparsity in SPARSITY:
                ctr += 1
                print \
                    ('\n****************************************************************************************')
                print \
                    (f'Running trial {ctr} of {np.product(list(map(len, [FXNS, MIN, SPARSITY, OUTER])))}... '
                      f'function name = {function_name}, _min={_min}, _max={_max}, len_outer={len_outer}, len_inner={len_inner}, sparsity={_sparsity}')

                if FXN is sd.create_random:                  # ' create_random ' in str(FXN):
                    RANDOM_SD = FXN((len_outer, len_inner), _sparsity)
                else:
                    RANDOM_SD = FXN(_min, _max, (len_outer, len_inner), _sparsity)

                # GET ACTUALS ################################################################################################3
                act_fxn_name = function_name
                act_min = sd.min_(RANDOM_SD)
                act_max = sd.max_(RANDOM_SD)
                act_outer_len = sd.outer_len(RANDOM_SD)
                act_inner_len = sd.inner_len(RANDOM_SD)
                act_sparsity = sd.sparsity(RANDOM_SD)

                # GET / VERIFY OUTER KEY DTYPE   (MUST ALWAYS BE py_int)
                RAW_OUTER_KEY_DTYPES = np.unique \
                    (np.fromiter((str(type(_)) for _ in RANDOM_SD.keys()), dtype='<U100'))
                number_of_outer_key_dtypes = len(RAW_OUTER_KEY_DTYPES)
                if number_of_outer_key_dtypes > 1:
                    exception_handle('number of outer key dtypes', 'equal to', number_of_outer_key_dtypes, 1)
                else:
                    raw_outer_key_dtype = RAW_OUTER_KEY_DTYPES[0]
                    if raw_outer_key_dtype != "<class 'int'>":
                        exception_handle('outer key dtype', 'equal to', raw_outer_key_dtype, "<class 'int'>")
                    else:
                        del RAW_OUTER_KEY_DTYPES, number_of_outer_key_dtypes

                # GET / VERIFY INNER KEY DTYPE   (MUST ALWAYS BE py_int)
                RAW_INNER_KEY_DTYPES = []
                for outer_key in RANDOM_SD:
                    RAW_INNER_KEY_DTYPES += [str(type(_)) for _ in RANDOM_SD[outer_key].keys()]
                RAW_INNER_KEY_DTYPES = np.unique(RAW_INNER_KEY_DTYPES)
                number_of_inner_key_dtypes = len(RAW_INNER_KEY_DTYPES)
                if number_of_inner_key_dtypes > 1:
                    exception_handle('number of inner key dtypes', 'equal to', number_of_inner_key_dtypes, 1)
                else:
                    raw_key_dtype = RAW_INNER_KEY_DTYPES[0]
                    if 'numpy.int8' in raw_key_dtype: act_key_dtype = 'np_int8'
                    elif 'numpy.int32' in raw_key_dtype: act_key_dtype = 'np_int32'
                    elif 'numpy.float64' in raw_key_dtype: act_key_dtype = 'np_float64'
                    elif 'int' in raw_key_dtype: act_key_dtype = 'py_int'
                    elif 'float' in raw_key_dtype: act_key_dtype = 'py_float'
                    else: raise Exception \
                            (f'\n*** act_key_dtype LOGIC IS FAILING')
                    del RAW_INNER_KEY_DTYPES, number_of_inner_key_dtypes


                # GET INNER VALUE DTYPE
                RAW_INNER_VALUE_DTYPES = []
                for outer_key in RANDOM_SD:
                    RAW_INNER_VALUE_DTYPES += \
                        [str(type(RANDOM_SD[outer_key][inner_key])) for
                        inner_key in
                        RANDOM_SD[outer_key].keys()]
                RAW_INNER_VALUE_DTYPES = np.unique(RAW_INNER_VALUE_DTYPES)
                number_of_inner_value_dtypes = len(RAW_INNER_VALUE_DTYPES)
                if number_of_inner_value_dtypes > 1:
                    exception_handle('number of inner value dtypes',
                                     'equal to', number_of_inner_value_dtypes,
                                     1)
                else:
                    raw_value_dtype = RAW_INNER_VALUE_DTYPES[0]
                    if 'numpy.int8' in raw_value_dtype:
                        act_value_dtype = 'np_int8'
                    elif 'numpy.int32' in raw_value_dtype:
                        act_value_dtype = 'np_int32'
                    elif 'numpy.float64' in raw_value_dtype:
                        act_value_dtype = 'np_float64'
                    elif raw_value_dtype == "<class 'int'>":
                        act_value_dtype = 'py_int'
                    elif raw_value_dtype == "<class 'float'>":
                        act_value_dtype = 'py_float'
                    else:
                        raise Exception(
                            f'\n*** act_value_dtype LOGIC IS FAILING ***\n')

                    del RAW_INNER_VALUE_DTYPES, number_of_inner_value_dtypes
                # END GET ACTUALS ################################################################################################

                # GET EXPECTEDS ###################################################################################################
                if 'bin' in function_name:
                    exp_min, exp_max = 0, 1
                elif 'int' in function_name:
                    exp_min, exp_max = _min, _max - 1
                elif 'float' in function_name:
                    exp_min, exp_max = _min, _max
                if function_name == 'create_random': exp_min, exp_max = 0, 9
                if _sparsity == 0:
                    if 'bin' in function_name:
                        exp_min = 1
                    elif function_name == 'create_random':
                        exp_min = 1
                    elif 'int' in function_name:
                        exp_min = _min if _min != 0 else 1
                    elif 'float' in function_name:
                        exp_min = _min
                    else:
                        raise Exception(
                            f'\n*** LOGIC FOR SETTING MIN WHEN SPARSITY IS ZERO IS FAILING ***\n')
                elif _sparsity > 0 and _sparsity < 100:
                    exp_min = 0
                elif _sparsity == 100:
                    exp_min, exp_max = 0, 0

                # exp len_outer, exp len_inner set by iterator
                # exp sparsity set by iterator
                exp_key_dtype = EXPECTED_KEY_DICT[function_name]
                exp_value_dtype = EXPECTED_VALUE_DICT[function_name]
                # if _sparsity == 100: exp_value_dtype = 'py_int'
                # GET EXPECTEDS ###################################################################################################

                print()
                # print(RANDOM_SD)
                print()
                print(f'FXN = {function_name}')
                print(
                    f'initial min/max = {_min}/{_max}, expected min/max = {exp_min}/{exp_max}, actual min/max = {act_min}/{act_max}')
                print(
                    f'expected shape = {len_outer}/{len_inner}, actual shape = {act_outer_len}/{act_inner_len}')
                print(
                    f'expected sparsity = {_sparsity}, actual sparsity = {act_sparsity}')
                print(
                    f'expected key dtype = {exp_key_dtype}, actual key dtype = {act_key_dtype}')
                print(
                    f'expected value dtype = {exp_value_dtype}, actual value dtype = {act_value_dtype}')

                # TEST MIN/MAX INTEGER ################################################################################
                if 'int' in EXPECTED_VALUE_DICT[function_name] or 'bin' in \
                        EXPECTED_VALUE_DICT[function_name]:
                    if act_min != exp_min: exception_handle('_min', 'equal to',
                                                            act_min, exp_min)
                    if act_max != exp_max: exception_handle('_max', 'equal to',
                                                            act_max, exp_max)
                # END TEST MIN/MAX INTEGER ################################################################################

                # TEST MIN/MAX FLOAT ###################################################################################
                elif 'float' in EXPECTED_VALUE_DICT[function_name]:
                    if act_min < exp_min or act_min > exp_min + min_max_window:
                        exception_handle('_min', 'within window of', act_min,
                                         exp_min)
                    if act_max > exp_max or act_max < exp_max - min_max_window:
                        exception_handle('_max', 'within window of', act_max,
                                         exp_max)
                # END TEST MIN/MAX FLOAT ###################################################################################
                else:
                    raise Exception(
                        f'\n*** LOGIC FOR TESTING exp_min/exp_max VS act_min/act_max IS FAILING ***/n')

                if len_outer != act_outer_len: exception_handle('len_outer',
                                                                'equal to',
                                                                act_outer_len,
                                                                len_outer)
                if len_inner != act_inner_len: exception_handle('len_inner',
                                                                'equal to',
                                                                act_inner_len,
                                                                len_inner)
                if act_sparsity > _sparsity + sparsity_window or act_sparsity < _sparsity - sparsity_window:
                    exception_handle('sparsity', 'within range of',
                                     act_sparsity, _sparsity)
                if act_key_dtype != exp_key_dtype: exception_handle(
                    'key_dtype', 'equal to', act_key_dtype, exp_key_dtype)
                if act_value_dtype != exp_value_dtype: exception_handle(
                    'value_dtype', 'equal to', act_value_dtype,
                    exp_value_dtype)

                # ____ = vui.validate_user_str(f'type f > ', 'FA')

print(f'\nDone.')



