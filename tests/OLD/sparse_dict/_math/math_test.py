# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

import pytest

pytest.skip(reason=f'24_09_07_06_55_00 needs a lot of work', allow_module_level=True)


'''

# SPARSE MATRIX MATH ####################################################################################################################
vector_sum                      Vector sum of user-specified outer dictionaries, with outer keys given by set.
sparse_matrix_math              Function called by decorators of specific matrix _math functions.
matrix_add                      Element-wise addition of two sparse dictionaires representing identically sized matrices.
matrix_subtract                 Element-wise subtraction of two sparse dictionaires representing identically sized matrices.
matrix_multiply                 Element-wise multiplication of two sparse dictionaires representing identically sized matrices.
matrix_divide                   Element-wise division of two sparse dictionaires representing identically sized matrices.
# END SPARSE MATRIX MATH #################################################################################################################
# SPARSE SCALAR MATH #####################################################################################################################
sparse_scalar_math              Function called by decorators of specific scalar _math functions.
scalar_add                      Element-wise addition of a scalar to a sparse dictionary representing a matrix.
scalar_subtract                 Element-wise subraction of a scalar from a sparse dictionary representing a matrix.
scalar_multiply                 Element-wise multiplication of a sparse dictionary representing a matrix by a scalar.
scalar_divide                   Element-wise division of a sparse dictionary representing a matrix by a scalar.
scalar_power                    Raises every element of a sparse dictionary representing a matrix by a scalar.
scalar_exponentiate             Exponentiates a scalar by elements of a sparse dictionary representing a matrix.
# END SPARSE SCALAR MATH #################################################################################################################
# SPARSE FUNCTIONS #######################################################################################################################
sparse_functions                Function called by decorators of specific miscellaneous functions.
exp                             Exponentiation of e by elements of a sparse dictionary representing a matrix.
ln                              Element-wise natural logarithm of a sparse dictionary representing a matrix.
sin                             Element-wise sine of a sparse dictionary representing a matrix.
cos                             Element-wise cosine of a sparse dictionary representing a matrix.
tan                             Element-wise tangent of a sparse dictionary representing a matrix.
tanh                            Element-wise hyperbolic tangent of a sparse dictionary representing a matrix.
logit                           Element-wise logistic transformation of a sparse dictionary representing a matrix.
relu                            Element-wise linear rectification of a sparse dictionary representing a matrix.
none                            Element-wise linear pass-through of a sparse dictionary representing a matrix (no change).
abs_                            Element-wise absolute value of a sparse dictionary.
# END  SPARSE FUNCTIONS ###################################################################################################################
'''


#########################################################################################################################################
#########################################################################################################################################
# GENERAL MATH ##########################################################################################################################

# SPARSE MATRIX MATH ####################################################################################################################

def vector_sum(DICT1, OUTER_KEYS_AS_LIST=None, WEIGHTS_AS_LIST=None):
    '''Vector sum of user-specified outer dictionaries, with outer keys given by set. '''

    fxn = inspect.stack()[0][3]
    DICT1 = dict_init(DICT1, fxn)

    # PREPARE AND CHECK OUTER_KEYS_AS_LIST
    if OUTER_KEYS_AS_LIST == None:
        OUTER_KEYS_AS_LIST = np.fromiter(range(outer_len(DICT1)), dtype=np.int32)

    exception_text = f'{module_name()}.{fxn}() Outer dict key in OUTER_KEYS_AS_LIST out of range. ' + \
                        f'Must be between 0 and len(DICT)-1, with value '
    if np.max(OUTER_KEYS_AS_LIST) > max(list(DICT1.keys())):
        raise Exception(exception_text + f'{max(OUTER_KEYS_AS_LIST)}.')
    elif np.min(OUTER_KEYS_AS_LIST) < 0:
        raise Exception(exception_text + f'{min(OUTER_KEYS_AS_LIST)}.')

    # PREPARE AND CHECK WEIGHTS_AS_LIST
    if WEIGHTS_AS_LIST == None:
        WEIGHTS_AS_LIST = np.fromiter((1 for _ in range(outer_len(DICT1))), dtype=np.int8)

    if len(WEIGHTS_AS_LIST) != len(OUTER_KEYS_AS_LIST):
        raise Exception(f'{module_name()}.{fxn}() Number of weights in WEIGHTS_AS_LIST ({len(WEIGHTS_AS_LIST)}) must equal number of keys in '
                        f'OUTER_KEYS_AS_SET ({len(OUTER_KEYS_AS_LIST)})')

    _inner_len = inner_len(DICT1)
    VECTOR_SUM = {int(0): {int(_): 0 for _ in range(_inner_len)}}
    for outer_key in OUTER_KEYS_AS_LIST:
        for inner_key in DICT1[outer_key]:
            if not WEIGHTS_AS_LIST is None:
                VECTOR_SUM[0][inner_key] += DICT1[outer_key][inner_key] * \
                                            WEIGHTS_AS_LIST[OUTER_KEYS_AS_LIST.index(outer_key)]
            else:
                VECTOR_SUM[0][inner_key] += DICT1[outer_key][inner_key]

    # ANY SUMMED TO ZERO, REMOVE FROM SPARSE DICT
    [VECTOR_SUM[0].pop(__) for __ in VECTOR_SUM[0] if VECTOR_SUM[0][__] == 0]

    # ENFORCE PLACEHOLDER RULE
    if _inner_len - 1 not in VECTOR_SUM[0]:
        VECTOR_SUM[0][_inner_len-1] = 0

    return VECTOR_SUM


def sparse_matrix_math(orig_func):
    '''Function called by decorators of specific matrix _math functions.'''
    # CRAZY WAY TO GET FUNCTION NAME TO FEED INTO operation, SINCE @ IS PASSING THE FUNCTION IN HERE
    FUNCTIONS = ['add', 'subtract', 'multiply', 'divide']
    operation = [__ for __ in range(len(FUNCTIONS)) if FUNCTIONS[__] in str(orig_func)][0]
    operation = FUNCTIONS[operation]

    @wraps(orig_func)
    def core_matrix_math(DICT1, DICT2):

        DICT1 = dict_init(DICT1, f'matrix_' + f'{operation}')
        DICT2 = dict_init(DICT2, f'matrix_' + f'{operation}')

        insufficient_dict_args_2(DICT1, DICT2, f'matrix_' + f'{operation}')
        matrix_shape_check(DICT1, DICT2, f'matrix_' + f'{operation}')

        if operation == 'divide' and True in [0 in list(DICT2[_].values()) for _ in DICT2]: # ONLY CATCHES PLACEHOLDER 0
            raise Exception(f'{module_name()}.{str(orig_func)}() 0/0 division error.')

        # MUST BUILD FROM SCRATCH TO BUILD CORRECT ORDERING IN INNER DICTS, CANT MAINTAIN ORDER IF DOING | ON COPYS OF DICT1 & DICT2
        FINAL_DICT = dict()
        _inner_len = inner_len(DICT1)
        for outer_key in DICT1:   # outer len of DICT1 & DICT2 already been checked, must be equal & contiguous
            '''FOR ADD, SUBTRACT, MULTIPLY, & DIVIDE, ALL MATCHING ZEROS BETWEEN DICT1 AND DICT2 STAY AT ZERO, SO OK TO SKIP
                OVER ALL LOCATIONS WHERE DOUBLE ZEROS (THIS ALLOWS FOR PLACES THAT WOULD BE 0/0 TO GET THRU!)'''

            # ASSEMBLAGE OF NON-ZERO INNER KEYS FROM DICT1 & DICT2 ASCENDING**************************************************************
            '''IF USE | OR ** ON {0:a, 2:b}, {1:c, 3:d} GET {0:a, 2:b, 1:c, 3:d} SO HAVE TO DO SOMETHING THAT SORTS '''
            INNER_DICT_SORTED_KEYS = set(DICT1[outer_key]).union(DICT2[outer_key])
            # ****************************************************************************************************************************
            FINAL_DICT[int(outer_key)] = {}
            for inner_key in INNER_DICT_SORTED_KEYS:  # 9-23-22 WAS reversed(list(INNER_DICT_SORTED_KEYS)) SUPPOSEDLY "# NOTHING ELSE IS KEEPING INNER DICTS ORDERED :("

                result = orig_func(DICT1, DICT2, outer_key, inner_key)

                if result != 0: FINAL_DICT[int(outer_key)][int(inner_key)] = result   # IF NON-ZERO, UPDATE KEY W NEW VALUE
                # IF == 0 AND ON LAST LOCATION, ENSURE PLACEHOLDER RULES ARE FOLLOWED
                elif inner_key == _inner_len-1 and result == 0: FINAL_DICT[outer_key][inner_key] = 0

        return FINAL_DICT

    return core_matrix_math


@sparse_matrix_math
def matrix_add(DICT1, DICT2, outer_key, inner_key):
    '''Element-wise addition of two sparse dictionaires representing identically sized matrices.'''
    return DICT1[outer_key].get(inner_key, 0) + DICT2[outer_key].get(inner_key, 0)

@sparse_matrix_math
def matrix_subtract(DICT1, DICT2, outer_key, inner_key):
    '''Element-wise subtraction of two sparse dictionaires representing identically sized matrices.'''
    return DICT1[outer_key].get(inner_key, 0) - DICT2[outer_key].get(inner_key, 0)

@sparse_matrix_math
def matrix_multiply(DICT1, DICT2, outer_key, inner_key):
    '''Element-wise multiplication of two sparse dictionaires representing identically sized matrices.'''
    return DICT1[outer_key].get(inner_key, 0) * DICT2[outer_key].get(inner_key, 0)

@sparse_matrix_math
def matrix_divide(DICT1, DICT2, outer_key, inner_key):
    '''Element-wise division of two sparse dictionaires representing identically sized matrices.'''
    return DICT1[outer_key].get(inner_key, 0) / DICT2[outer_key].get(inner_key, 0)
# END SPARSE MATRIX MATH #################################################################################################################

# SPARSE SCALAR MATH #####################################################################################################################
def sparse_scalar_math(orig_func):
    '''Function called by decorators of specific scalar _math functions.'''
    # CRAZY WAY TO GET FUNCTION NAME TO FEED INTO operation, SINCE @ IS PASSING THE FUNCTION IN HERE
    FUNCTIONS = ['add', 'subtract', 'multiply', 'divide', 'power', 'exponentiate']
    operation = [__ for __ in range(len(FUNCTIONS)) if FUNCTIONS[__] in str(orig_func)][0]
    operation = FUNCTIONS[operation]

    def core_scalar_math(DICT1, scalar):

        DICT1 = dict_init(DICT1, f'scalar_' + f'{operation}')
        insufficient_dict_args_1(DICT1, f'scalar_' + f'{operation}')

        TO_DELETE_HOLDER = []
        _inner_len = inner_len(DICT1)
        HOLDER_DICT = {_:{} for _ in range(outer_len(DICT1))}   # DOING THIS BECAUSE THE NATURE OF THE OPERATION CAUSES KEYS TO GO OUT OF ORDER
        for outer_key in DICT1:

            for inner_key in range(_inner_len):   # MUST HIT ALL POSITIONS
                result = orig_func(DICT1, outer_key, inner_key, scalar)

                if result != 0: HOLDER_DICT[int(outer_key)][int(inner_key)] = result   # IF NON-ZERO, UPDATE (OR CREATE) KEY W NEW VALUE
                # IF == 0 AND NOT ON LAST LOCATION, DEL LOCATION
                elif result == 0 and inner_key != _inner_len-1 and inner_key in DICT1[outer_key]:
                    TO_DELETE_HOLDER.append((outer_key, inner_key))
                # IF == 0 AND ON LAST LOCATION, ENSURE PLACEHOLDER RULES ARE FOLLOWED
                elif result == 0 and inner_key == _inner_len-1: HOLDER_DICT[outer_key][inner_key] = 0

        # Expection WHEN TRYING TO DELETE FROM DICTIONARY ON THE FLY, SO QUEUE DELETIONS UNTIL END
        for outer_key, inner_key in TO_DELETE_HOLDER:
            del DICT1[outer_key][inner_key]

        del TO_DELETE_HOLDER

        return HOLDER_DICT

    return core_scalar_math


@sparse_scalar_math
def scalar_add(DICT1, outer_key, inner_key, scalar):
    '''Element-wise addition of a scalar to a sparse dictionary representing a matrix.'''
    return DICT1[outer_key].get(inner_key,0) + scalar

@sparse_scalar_math
def scalar_subtract(DICT1, outer_key, inner_key, scalar):
    '''Element-wise subraction of a scalar from a sparse dictionary representing a matrix.'''
    return DICT1[outer_key].get(inner_key,0) - scalar

@sparse_scalar_math
def scalar_multiply(DICT1, outer_key, inner_key, scalar):
    '''Element-wise multiplication of a sparse dictionary representing a matrix by a scalar.'''
    return DICT1[outer_key].get(inner_key,0) * scalar

@sparse_scalar_math
def scalar_divide(DICT1, outer_key, inner_key, scalar):
    '''Element-wise division of a sparse dictionary representing a matrix by a scalar.'''
    return DICT1[outer_key].get(inner_key,0) / scalar

@sparse_scalar_math
def scalar_power(DICT1, outer_key, inner_key, scalar):
    '''Raises every element of a sparse dictionary representing a matrix by a scalar.'''
    return DICT1[outer_key].get(inner_key,0) ** scalar

@sparse_scalar_math
def scalar_exponentiate(DICT1, outer_key, inner_key, scalar):
    '''Exponentiates a scalar by elements of a sparse dictionary representing a matrix.'''
    return scalar ** DICT1[outer_key].get(inner_key,0)
# END SPARSE SCALAR MATH #################################################################################################################

# SPARSE FUNCTIONS #######################################################################################################################
def sparse_functions(orig_func):
    '''Function called by decorators of specific miscellaneous functions.'''
    # CRAZY WAY TO GET FUNCTION NAME TO FEED INTO operation, SINCE @ IS PASSING THE FUNCTION IN HERE
    FUNCTIONS = ['exp', 'ln', 'sin', 'cos', 'tan', 'tanh', 'logit', 'relu', 'none', 'abs_']
    operation = [__ for __ in range(len(FUNCTIONS)) if FUNCTIONS[__] in str(orig_func)][0]
    operation = FUNCTIONS[operation]

    def core_sparse_functions(DICT1):

        DICT1 = dict_init(DICT1, f'{operation}')
        insufficient_dict_args_1(DICT1, f'{operation}')

        _inner_len = inner_len(DICT1)
        for outer_key in deepcopy(DICT1):
            for inner_key in reversed(range(_inner_len)):    # MUST HIT ALL POSITIONS BECAUSE FOR MANY OF THESE FXNS f(0) != 0
                result = orig_func(DICT1, outer_key, inner_key)

                if result != 0: DICT1[int(outer_key)][int(inner_key)] = result  # IF NON-ZERO, UPDATE (OR CREATE) KEY W NEW VALUE
                # IF == 0 AND NOT ON LAST LOCATION, DEL LOCATION
                if result == 0 and inner_key != _inner_len-1 and inner_key in DICT1[outer_key]:
                    del DICT1[outer_key][inner_key]
                # IF == 0 AND ON LAST LOCATION, ENSURE PLACEHOLDER RULES ARE FOLLOWED
                if result == 0 and inner_key == _inner_len-1: DICT1[outer_key][inner_key] = 0

        return DICT1

    return core_sparse_functions

@sparse_functions
def exp(DICT1, outer_key, inner_key):
    '''Exponentiation of e by elements of a sparse dictionary representing a matrix.'''
    return np.exp(DICT1[outer_key].get(inner_key, 0))

@sparse_functions
def ln(DICT1, outer_key, inner_key):
    '''Element-wise natural logarithm of a sparse dictionary representing a matrix.'''
    return np.log(DICT1[outer_key].get(inner_key, 0))

@sparse_functions
def sin(DICT1, outer_key, inner_key):
    '''Element-wise sine of a sparse dictionary representing a matrix.'''
    return np.sin(DICT1[outer_key].get(inner_key, 0))

@sparse_functions
def cos(DICT1, outer_key, inner_key):
    '''Element-wise cosine of a sparse dictionary representing a matrix.'''
    return np.cos(DICT1[outer_key].get(inner_key, 0))

@sparse_functions
def tan(DICT1, outer_key, inner_key):
    '''Element-wise tangent of a sparse dictionary representing a matrix.'''
    return np.tan(DICT1[outer_key].get(inner_key, 0))

@sparse_functions
def tanh(DICT1, outer_key, inner_key):
    '''Element-wise hyperbolic tangent of a sparse dictionary representing a matrix.'''
    return np.tanh(DICT1[outer_key].get(inner_key, 0))

@sparse_functions
def logit(DICT1, outer_key, inner_key):
    '''Element-wise logistic transformation of a sparse dictionary representing a matrix.'''
    return 1 / (1 + np.exp(-DICT1[outer_key].get(inner_key, 0)))

@sparse_functions
def relu(DICT1, outer_key, inner_key):
    '''Element-wise linear rectification of a sparse dictionary representing a matrix.'''
    return max(0, DICT1[outer_key].get(inner_key, 0))

@sparse_functions
def none(DICT1, outer_key, inner_key):
    '''Element-wise linear pass-through of a sparse dictionary representing a matrix (no change).'''
    return DICT1[outer_key].get(inner_key, 0)

@sparse_functions
def abs_(DICT1, outer_key, inner_key):
    '''Element-wise absolute value of a sparse dictionary.'''
    return abs(DICT1[outer_key].get(inner_key,0))
# END SPARSE FUNCTIONS ##################################################################################################################

# END GENERAL MATH ######################################################################################################################
#########################################################################################################################################
#########################################################################################################################################