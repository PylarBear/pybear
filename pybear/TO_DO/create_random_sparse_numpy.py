import numpy as np
import sys, inspect, time
from general_sound import winlinsound as wls
from general_data_ops import list_sparsity as lsp, new_np_random_choice as nnrc
from data_validation import validate_user_input as vui


# SPEED & MEM TESTS 22_11_21.  FOR EQUAL SIZED ARRAYS, CALL IT SIZE "S", serialized ALWAYS TAKES LONGER THAN
# filter AND choice (MEM IS THE SAME FOR ALL 3) REGARDLESS OF WHETHER "S" IS IN THE HUNDREDS OR HUNDREDS OF
# MILLIONS AND REGARDLESS OF SPARSITY. (filter AND choice HAVE NO DISCERNIBLE DIFFERENCE, THEREFORE ARE ESSENTIALLY
# EQUAL.) FOR ARRAY OF SIZE "S", THERE IS SOME VARIATION IN TIME BASED ON SPARSITY.
# FOR SPARSITY = 50%, serial IS 2.25 X (choice OR filter); FOR SPARSITY = 90%, serial IS 2.0 X (choice OR filter);
# FOR SPARSITY = 99%, serial IS 1.6 X (choice OR filter).  IT APPEARS THAT JUST HAVING np.random.choice SIMPLY LOOK INTO A
# POOL OF SIZE "S" TAKES MORE TIME THAN THE choice OR filter METHODS DO TO 1) CREATE A RANDOM ARRAY 2) CREATE A MASK 3) APPLY
# THE MASK. THE VARIATION FROM SPARSITY INDICATES THAT INCREASING SIZE SELECTED OUT OF THE POOL OF SERIALIZED INDICES OF
# SIZE "S" BY serial INCREASES THE TIME EVEN FURTHER. 11/23/22, BUILT new_np_random_choice; IN TESTS HEAD TO HEAD VS
# np.random.choice, IS ALMOST 2X FASTER (WHEN PULLING 1000 OUT OF POOL OF 50e7), BUT HERE IS ABOUT 20-30X WORSE. THINKING
# ITS BECAUSE new_np_random_choice IS DOING MANY LOOPS TO CLEAN OUT THE DUPLICATES GIVEN THE SPARSITIES BEING USED HERE.
#
# GIVEN THAT TIME AND MEMORY HAVE NO MEASURABLE DIFFERENCE FOR choice OR filter, GOING TO USE choice AS THE DEFAULT
# create_random_sparse_numpy, BECAUSE ITS ACCURACY IN SPARSITY IS BETTER THAN filter.

###################################################################################################################################
# GENERAL FUNCTIONS
# exception_handling
# zero_cleaner                                 Replace zeros in a randomly generated array with non-zero random values.
# build_serialized_posns                       Return a list of serialized indices to map to an array.
# create_random_sparse_numpy                   Winner of RAM/speed trials of the three methods.
###################################################################################################################################
# METHOD 1 - BUILD FULL-SIZED MASK WITH DENSE LOCATIONS DETERMINED BY random.choice ON [0,1], WITH p ACHIEVING SPARSITY ###########
# APPLY MASK TO A FULL SIZED NP ARRAY FILLED AS DESIRED
# build_serialized_posns                       Return a list of serialized indices to map to an array
###################################################################################################################################
# METHOD 2 - BUILD FULL-SIZED ARRAY ON RANGE [0-30000] WITH DENSE LOCATIONS DETERMINED BY FILTERING THE RANGE BY A CUTOFF ###########
# create_sparse_numpy_by_number_filter         Generate an array filled randomly from [1,30000], create sparse mask by applying
#                                              number filter, then mask another randomly generated array.
###################################################################################################################################
# METHOD 3 - BUILD SERIALIZED DENSE LOCATIONS USING random.choice AND AN EQUAL-SIZED VECTOR OF VALUES, MAP TO A FULL SIZED ARRAY ##
# create_sparse_numpy_by_dense_serialized      Generate a serialized list of indices and random values and map into a fully-sized
#                                              array of zeros.
###################################################################################################################################

# GENERAL FUNCTIONS ###############################################################################################################
def exception_handling(fxn, jargon):
    raise Exception(f'\n*** {fxn}: {jargon} ***\n')


def zero_cleaner(_min, _max, tuple_of_shape, _dtype, _function):
    ''' Replace zeros in a randomly generated array with non-zero random values. '''

    # fxn = inspect.stack()[0][3]

    ARGS = (_min, _max, tuple_of_shape, _dtype)
    NON_ZERO_VALUES = _function(*ARGS if 'int' in str(_function) else ARGS[:-1])
    NON_ZERO_VALUES = NON_ZERO_VALUES.astype(_dtype)
    # DONT LOOP THE ZEROES OUT OF THE BIG ARRAY "NON_ZERO_VALUES"!  LOOP THEM OUT OF THE PATCH!
    if 0 in NON_ZERO_VALUES:
        MASK = np.array(NON_ZERO_VALUES == 0, dtype=bool)
        mask_len = np.sum(MASK.astype(np.int8))
        PATCH = _function(_min, _max, (1, mask_len))[0]
        if 0 in PATCH:
            while 0 in PATCH:
                PATCH_MASK = np.array(PATCH == 0, dtype=bool)
                patch_mask_len = np.sum(PATCH_MASK)
                PATCH[PATCH_MASK] = _function(_min, _max, (1, patch_mask_len))[0]
            del PATCH_MASK, patch_mask_len
        NON_ZERO_VALUES[MASK] = PATCH
        del MASK, mask_len, PATCH
    del _function, tuple_of_shape, ARGS

    return NON_ZERO_VALUES


def build_serialized_posns(shape_as_tuple, _sparsity, _dtype):
    ''' Return a list of serialized indices and corresponding values to map to an array. '''

    fxn = inspect.stack()[0][3]

    __ = str(_dtype).upper()

    total_size = np.product(shape_as_tuple).astype(np.int32)
    dense_size = total_size * (100 - _sparsity) / 100

    # IF SPARSITY DOESNT GO EVENLY INTO NUM ELEMENTS, PROBABILISTICALLY ADJUST NUMBER OF DENSE TO COMPENSATE
    dense_size = int(dense_size // 1) + np.random.choice((0, 1), 1, p=(1 - dense_size % 1, dense_size % 1))[0]

    sparse_size = int(total_size - dense_size)

    # CREATE A SINGLE SERIALIZED VECTOR CONTAINING THE DENSE POSITIONS IN FINAL SPARSE DICT #############################
    # ALLOW new_np_random_choice TO SELECT FROM THE SMALLER OF dense_size OR sparse_size, SAVES MEMORY & TIME
    if sparse_size >= dense_size:  # DENSE IS SMALLER (HOPEFULLY THE USUAL CASE)
        # 11/23/22 TRIED new_np_random_choice IN PLACE OF np.random.choice
        RAND_SERIALIZED_DENSE_POSNS = nnrc.new_np_random_choice(
            np.fromiter(range(total_size), dtype=np.int32), (1, dense_size), replace=False).astype(np.int32)[0]
        RAND_SERIALIZED_DENSE_POSNS.sort()  # GUARANTEES OUTER & INNER KEYS ARE ORDERED CORRECTLY AFTER CONSTRUCTION

    elif sparse_size < dense_size:
        # 11/23/22 TRIED new_np_random_choice IN PLACE OF np.random.choice
        RAND_SERIALIZED_SPARSE_POSNS = nnrc.new_np_random_choice(
            np.fromiter(range(total_size), dtype=np.int32), (1, sparse_size), replace=False).astype(np.int32)[0]
        RAND_SERIALIZED_SPARSE_POSNS.sort()  # GUARANTEES OUTER & INNER KEYS ARE ORDERED CORRECTLY AFTER CONSTRUCTION
        # GET OPPOSITE SPACE OF SPARSE_POSNS
        MASK = np.ones((1, total_size), dtype=bool)[0]
        MASK[RAND_SERIALIZED_SPARSE_POSNS] = False
        # END GET OPPOSITE SPACE OF SPARSE_POSNS
        RAND_SERIALIZED_DENSE_POSNS = np.fromiter(range(total_size), dtype=np.int32)[MASK]  # SORTED BY NATURE
        del MASK, RAND_SERIALIZED_SPARSE_POSNS, sparse_size, dense_size
    # END CREATE A SINGLE SERIALIZED VECTOR CONTAINING THE DENSE POSITIONS IN FINAL SPARSE DICT #########################

    '''
    # CREATE RANDOM VALUES MATCHING THE DENSE SIZE ##########################################################################
    # LOOP OVER AND OVER UNTIL ALL ZEROS ARE OUT OF IT
    if 'INT' in __: RAND_SERIALIZED_VALUES = zero_cleaner(_min, _max, (1,len(RAND_SERIALIZED_DENSE_POSNS)), _dtype, np.random.randint)[0]
    elif 'FLOAT' in __: RAND_SERIALIZED_VALUES = zero_cleaner(_min, _max, (1, len(RAND_SERIALIZED_DENSE_POSNS)), _dtype, np.random.uniform)[0]
    else: exception_handling(fxn, f'LOGIC OF SELECTING randint OR uniform FOR zero_cleaner BY "INT" OR "FLOAT" IN str(_dtype) IS FAILING')

    # END CREATE RANDOM VALUES BASED ON USER WANT int/float, MATCHING THE DENSE SIZE ########################################
    '''

    return RAND_SERIALIZED_DENSE_POSNS


def create_random_sparse_numpy(_min, _max, shape_as_tuple, _sparsity, _dtype):
    '''Winner of RAM/speed trials of the three methods.'''

    fxn = inspect.stack()[0][3]
    # USE serialized TO GET ACCURACY IN SPARSITY FOR SMALL ARRAYS
    # IF total_size IS ABOVE X, MAKE BY FULLY SIZED NUMPY, IS 2X FASTER THAN SERIALZED AND LAW OF AVERAGES SHOULD
    # GET SPARSITY CLOSE ENOUGH
    # BUT WHEN SIZE IS SMALL, "FILTER" AND "CHOICE" HAVE A HARD TIME GETTING SPARSITY RIGHT, SO USE SERIALIZED
    if np.product(shape_as_tuple) >= 1e6:
        return create_sparse_numpy_by_random_choice_mask(_min, _max, shape_as_tuple, _sparsity, _dtype)
    else:
        return create_sparse_numpy_by_dense_serialized(_min, _max, shape_as_tuple, _sparsity, _dtype)


# END GENERAL FUNCTIONS ############################################################################################################

###################################################################################################################################
# METHOD 1 - BUILD FULL-SIZED MASK WITH DENSE LOCATIONS DETERMINED BY random.choice ON [0,1], WITH p ACHIEVING SPARSITY ###########
# APPLY MASK TO A FULL SIZED NP ARRAY FILLED AS DESIRED

def create_sparse_numpy_by_random_choice_mask(_min, _max, shape_as_tuple, sparsity, _dtype):
    ''' Apply a mask of bools generated by random.choice to a filled base array of int or floats to achieve sparsity.'''

    fxn = inspect.stack()[0][3]

    __ = str(_dtype).upper()

    # USE zero-cleaner TO GET A NON ZERO BASE ARRAY
    if 'INT' in __: _function = np.random.randint
    elif 'FLOAT' in __: _function = np.random.uniform
    else: exception_handling(fxn, f'LOGIC OF SELECTING randint OR uniform BY "INT" OR "FLOAT" IN str(_dtype) IS FAILING')

    BASE_ARRAY = zero_cleaner(_min, _max, shape_as_tuple, _dtype, _function)

    MASK = np.random.choice([1, 0], np.product(shape_as_tuple), replace=True,
                    p=(sparsity / 100, (100 - sparsity) / 100)).reshape(shape_as_tuple).astype(bool)

    # REMEMBER! THE MASK IS GOING TO BECOME A BOOL TO REPRESENT PLACES IN THE INCOMING BASE ARRAY THAT WILL GO TO ZERO!  THAT
    # MEANS THAT THE PLACES THAT WILL BE ZERO MUST BE A ONE IN THE MASK ABOVE, AND ZERO IF NOT GOING TO BE ZERO! MAKE SENSE?

    BASE_ARRAY[MASK] = 0
    del MASK

    return BASE_ARRAY

# END METHOD 1 - BUILD FULL-SIZED MASK WITH DENSE LOCATIONS DETERMINED BY random.choice ON [0,1], WITH p ACHIEVING SPARSITY #######
###################################################################################################################################
###################################################################################################################################


###################################################################################################################################
###################################################################################################################################
# METHOD 2 - BUILD FULL-SIZED ARRAY ON RANGE [0,30000] WITH DENSE LOCATIONS DETERMINED BY FILTERING THE RANGE BY A CUTOFF ###########

def create_sparse_numpy_by_number_filter(_min, _max, shape_as_tuple, _sparsity, _dtype):
    '''Generate an array filled randomly from [1,30000], create sparse mask by applying number filter, then mask another randomly generated array.'''

    fxn = inspect.stack()[0][3]

    __ = str(_dtype).upper()

    MASK = np.random.randint(0, 30000, shape_as_tuple, dtype=np.int16)   # #30000 SO DONT HAVE TO GO TO int32

    # GET cutoff TO BUILD MASK AND FOR CALCULATING FINAL VALUES OF NONZEROS
    cutoff = (1 - _sparsity / 100) * 30000
    # GET MASK FIRST BUT DONT APPLY YET, APPLY AFTER MATH! (IF DO MASK FIRST, MATH WILL CHANGE THE ZEROS)
    MASK = np.array(MASK < cutoff, dtype=np.int8)

    # CREATE THE ARRAY TO WHICH THE MASK WILL BE APPLIES, AND APPLY MASK
    # IF _min IS LESS THAN ZEROS, UNWANTED ZEROS COULD SHOW UP, SO RUN 'INT' & 'FLOAT' THRU THE zero_cleaner CYCLE
    if 'BIN' in __:
        SPARSE_ARRAY = np.ones(shape_as_tuple, dtype=_dtype)
        SPARSE_ARRAY = np.array(SPARSE_ARRAY * MASK, dtype=_dtype)
    elif 'INT' in __:
        SPARSE_ARRAY = np.random.randint(_min, _max, shape_as_tuple).astype(_dtype)
        SPARSE_ARRAY = np.array(SPARSE_ARRAY * MASK, dtype=_dtype)
        SPARSE_ARRAY = zero_cleaner(_min, _max, shape_as_tuple, _dtype, np.random.randint)
    elif 'FLOAT' in __:
        SPARSE_ARRAY = np.random.uniform(_min, _max, shape_as_tuple).astype(_dtype)
        SPARSE_ARRAY = np.array(SPARSE_ARRAY * MASK, dtype=_dtype)
        SPARSE_ARRAY = zero_cleaner(_min, _max, shape_as_tuple, _dtype, np.random.uniform)
    else: exception_handling(fxn, f'LOGIC OF SELECTING randint OR uniform BY "INT" OR "FLOAT" IN str(_dtype) IS FAILING')

    SPARSE_ARRAY = np.array(SPARSE_ARRAY * MASK, dtype=_dtype)

    del MASK, cutoff

    return SPARSE_ARRAY

# END METHOD 2 - END BUILD FULL-SIZED ARRAY OF RANGE 0-10000 WITH DENSE LOCATIONS DETERMINED BY FILTERING THE RANGE BY A CUTOFF ###
###################################################################################################################################
###################################################################################################################################


###################################################################################################################################
###################################################################################################################################
# METHOD 3 - BUILD SERIALIZED DENSE LOCATIONS USING random.choice AND AN EQUAL-SIZED VECTOR OF VALUES, MAP TO A FULL SIZED ARRAY ##

def create_sparse_numpy_by_dense_serialized(_min, _max, shape_as_tuple, sparsity, _dtype):
    ''' Generate a serialized list of indices and random values and map into a fully-sized array of zeros.'''

    fxn = inspect.stack()[0][3]

    _inner_len = int(shape_as_tuple[1])

    RAND_SERIALIZED_DENSE_POSNS = build_serialized_posns(shape_as_tuple, sparsity, _dtype)

    MASK = np.ones(shape_as_tuple, dtype=bool)
    # REMEMBER RAND_SERIALIZED FOUND THE DENSE POSITIONS, SO HAVE TO INVERT TO FIND THE SPARSE POSITIONS, THEN SET THOSE
    # POSN IN SPARSE_NUMPY_ARRAY TO 0
    MASK[RAND_SERIALIZED_DENSE_POSNS // _inner_len, RAND_SERIALIZED_DENSE_POSNS % _inner_len] = False
    del RAND_SERIALIZED_DENSE_POSNS

    if 'INT' in str(_dtype).upper():
        # LOOP UNTIL ALL THE ZEROS ARE OUT OF IT
        SPARSE_NUMPY_ARRAY = zero_cleaner(_min, _max, shape_as_tuple, _dtype, np.random.randint)
    elif 'FLOAT' in str(_dtype).upper():
        # LOOP UNTIL ALL THE ZEROS ARE OUT OF IT
        SPARSE_NUMPY_ARRAY = zero_cleaner(_min, _max, shape_as_tuple, _dtype, np.random.uniform)

    SPARSE_NUMPY_ARRAY[MASK] = 0

    return SPARSE_NUMPY_ARRAY

# END METHOD 3 - BUILD SERIALIZED DENSE LOCATIONS USING random.choice AND AN EQUAL-SIZED VECTOR OF VALUES, MAP TO A FULL SIZED ARRAY ##
#######################################################################################################################################
#######################################################################################################################################















if __name__ == '__main__':

    # TEST MODULE FOR RANDOM SPARSE NUMPYS ACCURACY (ANOTHER MODULE FOR SPEED / MEM)

    fxn = 'TEST IN GUARD'

    # VERIFICATION OF NEW SD create_random FXNS

    min_max_window = .1
    sparsity_window = .1

    FXNS = [create_sparse_numpy_by_random_choice_mask,
            create_sparse_numpy_by_number_filter,
            create_sparse_numpy_by_dense_serialized]
    MIN = [2, 0, 5, 0]
    MAX = [10, 10, 6, 2]
    OUTER = [400, 8000]
    INNER = [8000, 400]
    SPARSITY = [0, 25, 50, 75, 100]
    DTYPES = [np.int8, np.int16, np.int32, np.float64]

    print(f'Total trials = {np.product(list(map(len, [FXNS, MIN, OUTER, SPARSITY, DTYPES])))}')

    FXN_NAMES = ['create_sparse_numpy_by_random_choice_mask',
                    'create_sparse_numpy_by_number_filter',
                    'create_sparse_numpy_by_dense_serialized']


    def error_handle(attr, constraint, act_value, expected_value):
        print(f'\nWants to Except for actual {attr} ({act_value}) not {constraint} expected value of {expected_value}.')
        if vui.validate_user_str(f'Quit(q) continue(c) > ', 'QC') == 'Q':
            wls.winlinsound(444,500)
            raise Exception(f'\nactual {attr} ({act_value}) not {constraint} expected value of {expected_value}.')


    ctr = 0
    for itr, FXN in enumerate(FXNS):
        # ASSIGN A FUNCTION NAME FOR EACH PASS
        __ = str(FXN)

        if 'random_choice_mask' in __: function_name = 'random_choice_mask'
        elif 'number_filter' in __: function_name = 'number_filter'
        elif 'dense_serialized' in __: function_name = 'dense_serialized'
        else: exception_handling(fxn, f'GETTING function_name FROM str(FXN) LOGIC IS FAILING')

        # END ASSIGN A FUNCTION NAME FOR EACH PASS
        for _min, _max in zip(MIN, MAX):
            for len_outer, len_inner in zip(OUTER, INNER):
                for _sparsity in SPARSITY:
                    for _dtype in DTYPES:
                        ctr += 1
                        print('\n****************************************************************************************')
                        print(f'Running trial {ctr} of {np.product(list(map(len, [FXNS, MIN, OUTER, SPARSITY, DTYPES])))}... '
                              f'function name = {function_name}, _min={_min}, _max={_max}, len_outer={len_outer}, '
                              f'len_inner={len_inner}, sparsity={_sparsity}, dtype={_dtype}')

                        RANDOM_SD = FXN(_min, _max, (len_outer, len_inner), _sparsity, _dtype)

                        # GET ACTUALS ################################################################################################3
                        act_fxn_name = function_name
                        act_min = np.min(RANDOM_SD)
                        act_max = np.max(RANDOM_SD)
                        act_outer_len = len(RANDOM_SD)
                        act_inner_len = len(RANDOM_SD[0])
                        act_sparsity = lsp.list_sparsity(RANDOM_SD)
                        raw_act_value_dtype = RANDOM_SD[0][0].dtype
                        if 'int8' in str(raw_act_value_dtype): act_value_dtype = 'np_int8'
                        elif 'int16' in str(raw_act_value_dtype): act_value_dtype = 'np_int16'
                        elif 'int32' in str(raw_act_value_dtype): act_value_dtype = 'np_int32'
                        elif 'float64' in str(raw_act_value_dtype): act_value_dtype = 'np_float64'
                        else: exception_handling(fxn, f'act_value_dtype LOGIC IS FAILING')
                        # END GET ACTUALS ################################################################################################

                        # GET EXPECTEDS ###################################################################################################
                        raw_exp_value_dtype = str(_dtype)
                        exp_min, exp_max = _min, _max
                        if 'int' in raw_exp_value_dtype: exp_min, exp_max = _min, _max - 1
                        elif 'float' in raw_exp_value_dtype: exp_min, exp_max = _min, _max
                        else: exception_handling(fxn, f'exp_min/exp_max INITIAL SET LOGIC IS FAILING')
                        if _sparsity == 0:
                            exp_min = _min if _min != 0 else _min+1 if 'int' in raw_exp_value_dtype else _min
                        elif _sparsity > 0 and _sparsity < 100:
                            exp_min = 0 if _min >= 0 else _min
                            exp_max = 0 if _max <= 0 else _max-1 if 'int' in raw_exp_value_dtype else _max
                        elif _sparsity == 100:
                            exp_min, exp_max = 0, 0
                        # exp len_outer, exp len_inner set by iterator
                        # exp sparsity set by iterator
                        if 'int8' in str(raw_exp_value_dtype): exp_value_dtype = 'np_int8'
                        elif 'int16' in str(raw_exp_value_dtype): exp_value_dtype = 'np_int16'
                        elif 'int32' in str(raw_exp_value_dtype): exp_value_dtype = 'np_int32'
                        elif 'float64' in str(raw_exp_value_dtype): exp_value_dtype = 'np_float64'
                        else: exception_handling(fxn, f'act_value_dtype LOGIC IS FAILING')
                        # GET EXPECTEDS ###################################################################################################

                        print()
                        # print(RANDOM_SD)
                        print()
                        print(f'FXN = {function_name}')
                        print(
                            f'initial min/max = {_min}/{_max}, expected min/max = {exp_min}/{exp_max}, actual min/max = {act_min}/{act_max}')
                        print(f'expected shape = {len_outer}/{len_inner}, actual shape = {act_outer_len}/{act_inner_len}')
                        print(f'expected sparsity = {_sparsity}, actual sparsity = {act_sparsity}')
                        print(f'expected value dtype = {exp_value_dtype}, actual value dtype = {act_value_dtype}')

                        # TEST MIN/MAX INTEGER ################################################################################
                        if 'int' in exp_value_dtype or 'bin' in exp_value_dtype:
                            if act_min != exp_min:
                                attr, constraint, act_value, expected_value = '_min', 'equal to', act_min, exp_min
                                error_handle(attr, constraint, act_value, expected_value)
                            if act_max != exp_max:
                                attr, constraint, act_value, expected_value = '_max', 'equal to', act_max, exp_max
                                error_handle(attr, constraint, act_value, expected_value)
                        # END TEST MIN/MAX INTEGER ################################################################################

                        # TEST MIN/MAX FLOAT ###################################################################################
                        elif 'float' in exp_value_dtype:
                            if act_min < exp_min or act_min > exp_min + min_max_window:
                                attr, constraint, act_value, expected_value = '_min', 'within window of', act_min, exp_min
                                error_handle(attr, constraint, act_value, expected_value)
                            if act_max > exp_max or act_max < exp_max - min_max_window:
                                attr, constraint, act_value, expected_value = '_max', 'within window of', act_max, exp_max
                                error_handle(attr, constraint, act_value, expected_value)
                        # END TEST MIN/MAX FLOAT ###################################################################################
                        else:
                            exception_handling(f'LOGIC FOR TESTING exp_min/exp_max VS act_min/act_max IS FAILING')

                        if len_outer != act_outer_len:
                            error_handle('len_outer', 'equal to', act_outer_len, len_outer)
                        if len_inner != act_inner_len:
                            error_handle('len_inner', 'equal to', act_inner_len, len_inner)
                        if act_sparsity > _sparsity + sparsity_window or act_sparsity < _sparsity - sparsity_window:
                            error_handle('sparsity', 'within range of', act_sparsity, _sparsity)
                        if act_value_dtype != exp_value_dtype:
                            error_handle('value_dtype', 'equal to', act_value_dtype, exp_value_dtype)
                        # ____ = vui.validate_user_str(f'type f > ', 'FA')

    print(f'\nDone. All create_random_sparse_numpy functions passed all tests for accuracy.')
    wls.winlinsound(444, 500)









