# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import sys, inspect, time
from functools import wraps

from pybear import new_numpy
from pybear.sparse_dict import _validation as val



"""
decorator_for_create_random     Create a positive-valued random sparse matrix of with user-given type, min, max, dimensions and sparsity.
create_random                   Create a random sparse matrix of python integers from 0 to 9 with user-given dimensions and sparsity.
create_random_py_bin            Create a random sparse matrix of python integers from 0 to 1 with user-given dimensions and sparsity.
create_random_py_int            Create a positive-valued random sparse matrix of python integers with user-given min, max, dimensions and sparsity.
create_random_py_float          Create a positive-valued random sparse matrix of python floats with user-given min, max, dimensions and sparsity.
create_random_np_bin            Create a random sparse matrix of np.int8s from 0 to 1 with user-given dimensions and sparsity.
create_random_np_int32          Create a positive-valued random sparse matrix of np.int32s from with user-given min, max, dimensions and sparsity.
create_random_np_float64        Create a positive-valued random sparse matrix of np.float64s with user-given min, max, dimensions and sparsity.
"""






def decorator_for_create_random(orig_func):

    # orig_func sets the dtypes of the values inside the sparse dict

    def timer(_time, _time2, words):
        del_t = time.time() - _time
        _total_time = time.time() - _time2
        # print(f"Time for {words} = {round(del_t, 2)} sec, total elapsed time = {round(_total_time, 2)} sec")
        return time.time()

    @wraps(orig_func)
    def create_random_x(_min, _max, shape_as_tuple, _sparsity):
        '''Create a positive-valued random sparse matrix with user-given type, min, max, dimensions and sparsity.'''

        # 11/23/22 MAPPING THE DENSE VALUES OF A SPARSE ARRAY TO A SPARSE
        # DICT IS UNDER ALL CIRCUMSTANCES FASTEST WHEN USING np.nonzero TO GET
        # POSNS AND VALUES, THEN USING dict(()) TO MAP.

        # SUMMARY OF OPERATIONS
        # 1) VALIDATE PARAMS
        # 2) BUILD SPARSE NDARRAY OBJECT BY pybear.new_numpy.random.sparse
        # 3) MAKE A SPARSE DICT USING dict(()) TO BUILD INNER DICTS BY GETTING
        #       np.nonzero AND CORRESPONDING VALUES FOR EACH ARRAY IN fully_sized_array (FASTEST METHOD)
        # 4) FIX PLACEHOLDERS

        t0 = time.time()
        t1 = time.time()

        _len_outer, _len_inner = shape_as_tuple

        total_size = int(_len_outer * _len_inner)

        val.is_int(_len_outer)
        val.is_int(_len_inner)

        __ = str(orig_func).upper()

        # INPUT VERIFICATION ################################################################################################
        while True:
            if _sparsity > 100 or _sparsity < 0:
                print(f'{fxn}() sparsity is {_sparsity}, must be 0 to 100.')
                _sparsity = vui.validate_user_float(f'Enter sparsity (0 to 100) > ', min=0, max=100)

            if _len_outer == 0:
                print(f'{fxn}() outer len is {_len_outer}, must be greater than 0.')
                _len_outer = vui.validate_user_int(f'Enter outer len > ', min=1)

            if _len_inner == 0:
                print(f'{fxn}() inner len is {_len_inner}, must be greater than 0.')
                _len_inner = vui.validate_user_int(f'Enter inner len > ', min=1)

            if _sparsity >= 0 and _sparsity <= 100 and _len_outer > 0 and _len_inner > 0:
                break
        t0 = timer(t0, t1, 'input verification')
        # END INPUT VERIFICATION ############################################################################################

        size_warn_cutoff = 400e6

        while True:

            if total_size >= size_warn_cutoff and _sparsity < 100:
                print(
                f'\nTotal size of sparse data is ~{round(total_size, -6):,}.  This will take forever.')
                _ = vui.validate_user_str(f'Enter new size and sparsity(n), proceed anyway(p), quit(q) > ', 'NPQ')
                if _ == 'P': break
                elif _ == 'Q': sys.exit(f'\n*** TERMINATED BY USER. ***\n')
                elif _ == 'N':
                    while True:
                        _len_outer = vui.validate_user_int(f'\nEnter new outer size > ', min=1, max=1e9)
                        _len_inner = vui.validate_user_int(f'\nEnter new inner size > ', min=1, max=1e9)
                        _sparsity = vui.validate_user_int(f'\nEnter new sparsity > ', min=0, max=100)

                        if vui.validate_user_str(
                            f'User selected outer size = {_len_outer}, inner_size = {_len_inner}, sparsity = {_sparsity}. Accept? (y/n) > ', 'YN') == 'Y':
                            break

                    continue
            break
        t0 = timer(t0, t1, 'size handling')
        # END SIZE HANDLING ######################################################################################################

        ##########################################################################################################################
        # IF SPARISTY==100, JUST BUILD HERE, BYPASSING EVERYTHING BELOW ##########################################################
        if _sparsity == 100:
            SPARSE_DICT = {int(outer_idx):{int(_len_inner-1): orig_func(0)} for outer_idx in range(_len_outer)}
        # END IF SPARISTY==100 ###################################################################################################
        ##########################################################################################################################

        # CREATING LIST OF DENSE IDXS AND VALUES USING serialized_dense_locations HAS TWO LIMITATIONS:
        # 1) THE np.random.choice FUNCTION WITHIN CANT HANDLE LOOKING IN SIZE > 100M
        # 2) FOR ALL SIZES AND SPARSITIES, PULLING THE LIST OF DENSE IDXS AND VALUES USING np.random.choice TAKES AT BEST 1.6X
        #   LONGER THAN use_fully_sized_ndarray_w_mask DOES TO GENERATE A FULLY SIZED DENSE np array, GENERATE A FULLY SIZED
        #   RANDOM MASK OF 1s AND 0s, AND APPLY THE MASK.
        # THE ONLY THING THAT COULD REDEEM serialized IS IF MAPPING THE DENSE IDXS AND VALUES (THAT TAKE LONGER TO
        # MAKE) TO A SPARSE DICT GOES MUCH FASTER THAN ZIPPING A FULLY SIZED np array TO SPARSE DICT. (IT DOESNT, SLOWER TOO).

        ''' 11/23/22 A RELIC OF THE PAST.  PROVED TO BE SLOWER IN BOTH CONSTRUCTING DENSE AND MAPPING TO SPARSE DICT THAN ARRAYS.
        ##########################################################################################################################
        # DENSE SERIALIZED LOCATIONS #############################################################################################
        elif use_serialized_dense_locations:  # THE NEW WAY, CREATE AN EXACTLY SIZED RANDOMLY SELECTED SERIALIZED LIST
            # OF DENSE LOCATIONS & RESPECTIVE VALUES  ( :( np.random.choice IS SLOWER THAN full_np_array_w_mask METHOD

            # CREATE A SINGLE SERIALIZED VECTOR CONTAINING THE DENSE POSITIONS IN FINAL SPARSE DICT #############################
            # CREATE RANDOM VALUES BASED ON USER WANT int/float, MATCHING THE DENSE SIZE ########################################
            # CREATE AS NUMPY, THEN CONDITIONALLY CONVERT TO py int, py float, np int, np float

            RAND_SERIALIZED_DENSE_POSNS, RAND_SERIALIZED_VALUES = bspv(
                                                    1 if 'BIN' in __ else _min,
                                                    2 if 'BIN' in __ else _max,
                                                    shape_as_tuple,
                                                    _sparsity,
                                                    np.int8 if 'BIN' in __ else np.int32 if 'INT' in __ else np.float64)

            t0 = timer(t0, t1, 'select and sort random dense locations, generate rand values for each dense location')
            # END CREATE A SINGLE SERIALIZED VECTOR CONTAINING THE DENSE POSITIONS IN FINAL SPARSE DICT #########################
            # END CREATE RANDOM VALUES BASED ON USER WANT int/float, MATCHING THE DENSE SIZE ########################################


            # FOR DICTS WHERE inner_len >> outer_len, for/dict(()) IS 0.25*TIME OF for/for LOOP
            # BUT WHEN inner_len << outer_len, for/dict(()) IS IMPOSSIBLY SLOW.  USING BOTH CONDITIONALLY

            # METHOD 1 --- IF INNER DICTS ARE SHORTER THAN OUTER LEN, FILL BY FOR LOOPS (dict(()) METHOD IS MUCH SLOWER HERE)
            if _len_inner < _len_outer:
                SPARSE_DICT = {int(_): {} for _ in range(_len_outer)}
                for ser_loc_idx in range(len(RAND_SERIALIZED_DENSE_POSNS)):
                    SPARSE_DICT[int(RAND_SERIALIZED_DENSE_POSNS[ser_loc_idx] // _len_inner)][
                                int(RAND_SERIALIZED_DENSE_POSNS[ser_loc_idx] % _len_inner)] = \
                                    orig_func(RAND_SERIALIZED_VALUES[ser_loc_idx])  # APPLY USER DEFINED VALUE DTYPE

                del RAND_SERIALIZED_DENSE_POSNS, RAND_SERIALIZED_VALUES

                t0 = timer(t0, t1, 'build sparse dict by for loop')

            # METHOD 2 --- IF INNER DICTS ARE LONGER THAN OVERALL LEN, FILL BY dict(()) METHOD (MUCH FASTER THAN for LOOPS)
            elif _len_inner >= _len_outer:
                OUTER_KEYS = RAND_SERIALIZED_DENSE_POSNS // _len_inner
                INNER_KEYS = RAND_SERIALIZED_DENSE_POSNS % _len_inner
                del RAND_SERIALIZED_DENSE_POSNS

                SPARSE_DICT = {}
                for outer_key in range(_len_outer):
                    ACTIVE_OUTER_KEYS = np.argwhere(OUTER_KEYS==outer_key).transpose()[0]
                    ACTIVE_VALUES = RAND_SERIALIZED_VALUES[ACTIVE_OUTER_KEYS]

                    # IF OUTPUT IS TO BE NP, KEEP AS NDARRAY, VALUES ARE PUT IN DICT AS NP VALUES. OTHERWISE tolist FORCES
                    # np ints AND np floats TO py ints AND py floats.  ALWAYS FORCE KEYS TO py int!!!
                    if 'NP_' not in __: ACTIVE_VALUES = ACTIVE_VALUES.tolist()

                    SPARSE_DICT[int(outer_key)] = dict((
                                                        zip(INNER_KEYS[ACTIVE_OUTER_KEYS].tolist(),
                                                        ACTIVE_VALUES)
                    ))

                del RAND_SERIALIZED_VALUES, OUTER_KEYS, INNER_KEYS, ACTIVE_OUTER_KEYS, ACTIVE_VALUES

                t0 = timer(t0, t1, 'build sparse dict by dict(())')

            # ADD PLACEHOLDERS  --- CORRECT ORDERING IS GUARANTEED BY SORTED SERIAL LOCATION
            for outer_key in SPARSE_DICT:
                if _len_inner - 1 not in SPARSE_DICT[outer_key]:
                    SPARSE_DICT[int(outer_key)][int(_len_inner - 1)] = orig_func(0)
            t0 = timer(t0, t1, 'placeholders')
        # END DENSE SERIALIZED LOCATIONS #############################################################################################
        ##########################################################################################################################
        '''

        ##########################################################################################################################
        # FULLY SIZED NDARRAY W MASK #############################################################################################
        # THE OLD WAY, FILL A FULLY-SIZED NDARRAY WITH RANDOM NUMBERS ON [_min, _max] BASED
        # ON USER WANT bin/int OR float. CREATE A RANDOMLY FILLED MASK OF ONES AND ZEROS USING np.random.choice ON [0,1] WITH
        # LIKELIHOODS GIVEN BY _sparsity AND 1-_sparsity TO INDICATE A DENSE (STAYING) POSN IN THE FIRST ARRAY.
        # 11/21/22 FINALIZED A FUNCTION TO HANDLE THIS, general_data_ops.create_random_sparse_numpy.create_random_sparse_numpy.
        # CREATE AS NUMPY ints AND floats, THEN COVERT TO py LATER IF NEEDED.

        # VALUES = new_numpy.random.sparse(
        #                         1 if 'BIN' in __ else _min,
        #                         2 if 'BIN' in __ else _max,
        #                         shape_as_tuple,
        #                         _sparsity,
        #                         dtype = np.int8 if 'BIN' in __ else np.int32 if 'INT' in __ else np.float64
        # )

        t0 = timer(t0, t1, '"fully-sized ndarray w mask method" rand values')

        # CANT USE sd.zip_ndarray HERE, MUST DO PLUG N CHUG TO APPLY/DISAPPLY .tolist() ON VALUES
        SPARSE_DICT = {}
        for outer_key in range(_len_outer):
            NON_ZERO_KEYS = np.nonzero(VALUES[outer_key])[-1]
            NON_ZERO_VALUES = VALUES[outer_key][NON_ZERO_KEYS]

            if 'NP_' not in __: NON_ZERO_VALUES = NON_ZERO_VALUES.tolist()

            SPARSE_DICT[int(outer_key)] = dict((zip(NON_ZERO_KEYS.tolist(), NON_ZERO_VALUES)))

        del VALUES, NON_ZERO_VALUES, NON_ZERO_KEYS

        t0 = timer(t0, t1, '"fully-sized ndarray w mask method" build by dict(())')

        # ADD PLACEHOLDERS
        for outer_key in SPARSE_DICT:
            if _len_inner - 1 not in SPARSE_DICT[outer_key]:
                SPARSE_DICT[int(outer_key)][int(_len_inner - 1)] = orig_func(0)

        t0 = timer(t0, t1, 'placeholders')
        # END FULLY SIZED NDARRAY W MASK #########################################################################################
        ##########################################################################################################################

        return SPARSE_DICT

    return create_random_x


def create_random(shape_as_tuple, sparsity):
    '''Legacy. Create a random sparse matrix of python integers from 0 to 9 with user-given dimensions and sparsity.'''
    return create_random_py_int(0, 10, shape_as_tuple, sparsity)

@decorator_for_create_random
def create_random_py_bin(value):
    '''Create a random sparse matrix of python integers from 0 to 1 with user-given dimensions and sparsity.'''
    return int(value)

@decorator_for_create_random
def create_random_py_int(value):
    '''Create a positive-valued random sparse matrix of python integers with user-given min, max, dimensions and sparsity.'''
    return int(value)

@decorator_for_create_random
def create_random_py_float(value):
    '''Create a positive-valued random sparse matrix of python floats with user-given min, max, dimensions and sparsity.'''
    return float(value)

@decorator_for_create_random
def create_random_np_bin(value):
    '''Create a random sparse matrix of np.int8s from 0 to 1 with user-given dimensions and sparsity.'''
    return np.int8(value)

@decorator_for_create_random
def create_random_np_int32(value):
    '''Create a positive-valued random sparse matrix of np.int32s from with user-given min, max, dimensions and sparsity.'''
    return np.int32(value)

@decorator_for_create_random
def create_random_np_float64(value):
    '''Create a positive-valued random sparse matrix of np.float64s with user-given min, max, dimensions and sparsity.'''
    return np.float64(value)



















