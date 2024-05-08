# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import numpy as np
from functools import wraps

from _validation import (
                            _dict_init,
                            _insufficient_dict_args_2

)

from _utils import (
                    outer_len,
                    inner_len,
                    shape_,
                    sparsity
)

from _linalg_validation import (
                                _matrix_shape_check
)

"""

# PIZZA PROBABLY BEST TO LEAVE THESE HERE FOR REFERENCE UNTIL ALL sparse_dict FUNCTIONS HAVE A HOME
# SPARSE MATRIX MATH ####################################################################################################################
vector_sum                      Vector sum of user-specified outer dictionaries, with outer keys given by set.
sparse_matrix_math              Function called by decorators of specific matrix math functions.
matrix_add                      Element-wise addition of two sparse dictionaires representing identically sized matrices.
matrix_subtract                 Element-wise subtraction of two sparse dictionaires representing identically sized matrices.
matrix_multiply                 Element-wise multiplication of two sparse dictionaires representing identically sized matrices.
matrix_divide                   Element-wise division of two sparse dictionaires representing identically sized matrices.
# END SPARSE MATRIX MATH #################################################################################################################
# SPARSE SCALAR MATH #####################################################################################################################
sparse_scalar_math              Function called by decorators of specific scalar math functions.
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




# 24_05_07 LAST MATH STUFF IN sparse_dict MAIN FILE
sum_over_outer_key              Sum all the values in an inner dict, as given by outer dict key.
sum_over_inner_key              Sum over all inner dicts the values that are keyed with the user-entered inner key.
summary_stats                   Function called by decorators of specific summary statistics functions.
sum_                            Sum of all values of a sparse dictionary, across all inner dictionaries.
median_                         Median of all values of a sparse dictionary, across all inner dictionaries.
average_                        Average of all values of a sparse dictionary, across all inner dictionaries.
min_                            Minimum value in a sparse dictionary, across all inner dictionaries.
max_                            Maximum value in a sparse dictionary, across all inner dictionaries.
min_max_                        Returns minimum and maximum value in a sparse dictionary, across all inner dictionaries.
centroid_                       Centroid of a sparse dictionary.
variance_                       Variance of a one sparse dictionary column.
r_                              R of two sparsedict vectors of equal inner length.
rsq_                            RSQ of two sparsedict vectors of equal inner length.

"""



###############################################################################
###############################################################################
# GENERAL MATH ################################################################

# SPARSE MATRIX MATH ##########################################################

def vector_sum(DICT1:dict, OUTER_KEYS_AS_LIST=None, WEIGHTS_AS_LIST=None):

    """Vector sum of specified outer dictionaries.

    Parameters
    ----------
    DICT1:
        dict - sparse dictionary object containing vector(s) to be summed

    OUTER_KEYS_AS_LIST:
        array-like, default = None - outer key(s) of the sparse dictionary on
        which to perform vector addition

    WEIGHTS_AS_LIST:
        array-like, default = None - multiplicative weights applied to the
        respective vectors called out in OUTER_KEYS_AS_LIST. If provided,
        WEIGHTS_AS_LIST must be the same length as OUTER_KEYS_AS_LIST. If not
        provided, all the vectors are assigned unit weight.

    Returns
    -------
    VECTOR_SUM:
        dict - the weighted vector sum of the selected vectors

    See Also
    --------

    Notes
    -----


    Examples
    --------
    >>> from pybear.sparse_dict import vector_sum
    >>> SD = {0: {0:-1, 2:1}, 1: {0: 2, 1: 1, 2: -2}, 2: {1: 2, 2: -1}}
    # >>> OUTER_KEYS = []
    >>> out = vector_sum(SD, OUTER_KEYS_AS_LIST=OUTER_KEYS)
    >>> out


    """

    DICT1 = _dict_init(DICT1)

    _outer_len, _inner_len = shape_(DICT1)

    # PREPARE AND CHECK OUTER_KEYS_AS_LIST
    OUTER_KEYS_AS_LIST = OUTER_KEYS_AS_LIST or \
                                np.arange(_outer_len, dtype=np.int32)

    exception_text = lambda x, y: (f"Outer dict key '{x}' in OUTER_KEYS_AS_LIST "
            f"out of range. Must be between 0 and len(DICT)-1 ({y}) inclusive")

    if (_min := min(OUTER_KEYS_AS_LIST)) < 0:
        raise ValueError(exception_text(_min, _outer_len))

    if (_max := max(OUTER_KEYS_AS_LIST)) > max(DICT1):
        raise ValueError(exception_text(_max, _outer_len))

    del exception_text, _min, _max


    # PREPARE AND CHECK WEIGHTS_AS_LIST
    WEIGHTS_AS_LIST = WEIGHTS_AS_LIST or np.ones(_outer_len, dtype=np.int8)


    _, __ = len(WEIGHTS_AS_LIST), len(OUTER_KEYS_AS_LIST)
    if _ != __:
        raise ValueError(
            f'Number of weights in WEIGHTS_AS_LIST ({_}) must equal number of '
            f'keys in OUTER_KEYS_AS_LIST ({__})'
        )

    del _, __

    VECTOR_SUM = {int(0): {int(_): 0 for _ in range(_inner_len)}}
    for outer_key in OUTER_KEYS_AS_LIST:
        _weight = WEIGHTS_AS_LIST[OUTER_KEYS_AS_LIST.index(outer_key)]
        for inner_key in DICT1[outer_key]:
            VECTOR_SUM[0][inner_key] += DICT1[outer_key][inner_key] * _weight


    # ANY SUMMED TO ZERO, REMOVE FROM SPARSE DICT
    [VECTOR_SUM[0].pop(__) for __ in VECTOR_SUM[0] if VECTOR_SUM[0][__] == 0]

    # ENFORCE PLACEHOLDER RULE
    if _inner_len - 1 not in VECTOR_SUM[0]:
        VECTOR_SUM[0][int(_inner_len-1)] = 0

    return VECTOR_SUM


def sparse_matrix_math(orig_func):
    '''Function called by decorators of specific matrix math functions.'''
    # CRAZY WAY TO GET FUNCTION NAME TO FEED INTO operation, SINCE @ IS PASSING THE FUNCTION IN HERE
    FUNCTIONS = ['add', 'subtract', 'multiply', 'divide']
    operation = [__ for __ in range(len(FUNCTIONS)) if FUNCTIONS[__] in str(orig_func)][0]
    operation = FUNCTIONS[operation]

    @wraps(orig_func)
    def core_matrix_math(DICT1, DICT2):

        _insufficient_dict_args_2(DICT1, DICT2)
        DICT1 = _dict_init(DICT1)
        DICT2 = _dict_init(DICT2)

        _matrix_shape_check(DICT1, DICT2)

        if operation == 'divide' and sparsity(DICT2) > 0:
            raise ZeroDivisionError(f'{str(orig_func)}() 0/0 division error.')

        # MUST BUILD FROM SCRATCH TO BUILD CORRECT ORDERING IN INNER DICTS,
        # CANT MAINTAIN ORDER IF DOING | ON COPYS OF DICT1 & DICT2
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
    '''Function called by decorators of specific scalar math functions.'''
    # CRAZY WAY TO GET FUNCTION NAME TO FEED INTO operation, SINCE @ IS PASSING THE FUNCTION IN HERE
    FUNCTIONS = ['add', 'subtract', 'multiply', 'divide', 'power', 'exponentiate']
    operation = [__ for __ in range(len(FUNCTIONS)) if FUNCTIONS[__] in str(orig_func)][0]
    operation = FUNCTIONS[operation]

    def core_scalar_math(DICT1, scalar):

        DICT1 = _dict_init(DICT1, f'scalar_' + f'{operation}')
        _insufficient_dict_args_1(DICT1, f'scalar_' + f'{operation}')

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
# END SPARSE SCALAR MATH ######################################################

# SPARSE FUNCTIONS ############################################################
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
# END SPARSE FUNCTIONS ########################################################

# END GENERAL MATH ############################################################
###############################################################################
###############################################################################







def sum_over_outer_key(DICT:dict, outer_key:int) -> [int, float]:
    """Sum all the values in an inner dictionary corresponding to the given
    outer key.

    Parameters
    ----------
    DICT:
        dict - sparse dictionary object for which to sum over an outer key
    outer_key:
        int - the outer dictionary key to sum over


    Return
    ------
    SUM:
        [int, float] - the sum of all the values in the inner dictionary

    See Also
    --------

    Notes
    -----

    Examples
    --------
    PIZZA FINISH


    """

    _insufficient_dict_args_1(DICT)
    DICT = _dict_init(DICT)

    return sum(DICT[outer_key].values())


def sum_over_inner_key(DICT:dict, inner_key:int) -> [int, float]:
    """Sum the values in the given inner key position over all inner dicts.
    PIZZA VERIFY Accepts ragged sparse dictionaries.

    Parameters
    ----------
    DICT:
        dict - the sparse dictionary for which to perform summation
    inner_key:
        int - the inner key position to sum over

    Return
    ------
    SUM:
        [int, float] - the sum of the values in the given inner key position

    """

    DICT = _dict_init(DICT)
    _insufficient_dict_args_1(DICT)

    SUM = 0

    # IF ACCEPTS RAGGED *************

    for outer_key in DICT:
        if inner_key not in range(inner_len(DICT)):
            raise ValueError(f"Key '{inner_key}' out of bounds for inner dict with len {inner_len(DICT1)}.")
        if inner_key in DICT[outer_key]:
            SUM += DICT[outer_key][inner_key]
    return SUM

    # IF DOES NOT ACCEPT RAGGED *************



def summary_stats(orig_func):
    '''Function called by decorators of specific summary statistics functions.'''
    __ = ['sum_', 'median_', 'average_', 'min_', 'max_', 'min_max_', 'centroid_', 'variance_']
    fxn_idx = [op in str(orig_func) for op in __].index(True)
    fxn = __[fxn_idx]

    def statistics(DICT1):
        DICT1 = dict_init(DICT1, fxn)
        insufficient_dict_args_1(DICT1, fxn)

        NON_ZERO_ELEMENTS = np.empty((1,0), dtype=float)[0]
        for _ in DICT1:
            NON_ZERO_ELEMENTS = np.hstack((NON_ZERO_ELEMENTS, np.fromiter(DICT1[_].values(), dtype=float)))

        # REMEMBER PLACEHOLDERS!
        # IF ANY PLACEHOLDERS, CLEAR THEM OUT
        NON_ZERO_ELEMENTS = NON_ZERO_ELEMENTS[np.nonzero(NON_ZERO_ELEMENTS)[-1]]

        total_elements = size_(DICT1)   # MUST ACCOUNT FOR ALL THE ZEROS!

        return orig_func(DICT1, NON_ZERO_ELEMENTS, total_elements)

    return statistics

@summary_stats
def sum_(DICT1, NON_ZERO_ELEMENTS, total_elements):
    '''Sum of all values of a sparse dictionary, across all inner dictionaries.'''
    return np.sum(NON_ZERO_ELEMENTS)  # PLACEHOLDER ZEROS CAN BE IGNORED

@summary_stats
def median_(DICT1, NON_ZERO_ELEMENTS, total_elements):
    '''Median of all values of a sparse dictionary, across all inner dictionaries.'''
    return np.median(
                np.hstack(
                        NON_ZERO_ELEMENTS,
                        np.fromiter((0 for _ in range(total_elements - len(NON_ZERO_ELEMENTS))), dtype=float)
                )
            )

@summary_stats
def average_(DICT1, NON_ZERO_ELEMENTS, total_elements):
    '''Average of all values of a sparse dictionary, across all inner dictionaries.'''
    return np.sum(NON_ZERO_ELEMENTS) / total_elements

@summary_stats
def min_(DICT1, NON_ZERO_ELEMENTS, total_elements):
    '''Minimum value in a sparse dictionary, across all inner dictionaries.'''
    if total_elements - len(NON_ZERO_ELEMENTS) > 0:   # IF A DIFFERENCE, MEANS THERE WERE ZEROS IN THERE
        NON_ZERO_ELEMENTS = np.insert(NON_ZERO_ELEMENTS, 0, 0)

    return np.min(NON_ZERO_ELEMENTS)

@summary_stats
def max_(DICT1, NON_ZERO_ELEMENTS, total_elements):
    '''Maximum value in a sparse dictionary, across all inner dictionaries.'''
    if total_elements - len(NON_ZERO_ELEMENTS) > 0:  # IF A DIFFERENCE, MEANS THERE WERE ZEROS IN THERE
        NON_ZERO_ELEMENTS = np.insert(NON_ZERO_ELEMENTS, 0, 0)
    return np.max(NON_ZERO_ELEMENTS)

@summary_stats
def min_max_(DICT1, NON_ZERO_ELEMENTS, total_elements):
    '''Returns minimum and maximum value in a sparse dictionary, across all inner dictionaries.'''
    if total_elements - len(NON_ZERO_ELEMENTS) > 0:  # IF A DIFFERENCE, MEANS THERE WERE ZEROS IN THERE
        NON_ZERO_ELEMENTS = np.insert(NON_ZERO_ELEMENTS, 0, 0)
    return np.min(NON_ZERO_ELEMENTS), np.max(NON_ZERO_ELEMENTS)

@summary_stats
def centroid_(DICT1, NON_ZERO_ELEMENTS, total_elements):
    '''Centroid of a sparse dictionary.'''

    SUM = vector_sum(DICT1)

    _outer_len = outer_len(DICT1)      # DO THIS BEFORE CHANGING DICT1 to SUM
    # MUST CHANGE DICT1 to SUM IN ORDER TO USE scalar_divide ON IT
    DICT1 = SUM
    CENTROID = scalar_divide(DICT1, _outer_len)
    return CENTROID

@summary_stats
def variance_(DICT1, NON_ZERO_ELEMENTS, total_elements):
    '''Variance of one sparse dictionary column.'''
    _len = inner_len(DICT1)
    _avg = average_(DICT1)
    return np.sum([(DICT1.get(_, 0) - _avg)**2 for _ in range(_len)]) / (_len - 1)


def r_(DICT1, DICT2):
    '''R of two sparsedict vectors of equal inner length.'''
    fxn = inspect.stack()[0][3]
    _sparse_dict_check(DICT1)
    _sparse_dict_check(DICT2)
    insufficient_dict_args_2(DICT1, DICT2, fxn)
    dot_size_check(DICT1, DICT2, fxn)

    # ENSURE THAT DICT1 & DICT2 OUTER IDX IS 0
    DICT1[0] = DICT1.pop([_ for _ in DICT1][0])
    DICT2[0] = DICT2.pop([_ for _ in DICT2][0])

    xavg = average_(DICT1)
    yavg = average_(DICT2)
    numer1, denom1, denom2 = 0, 0, 0
    for _ in range(inner_len(DICT1)):
        numer1 += (DICT1[0].get(_,0) - xavg) * (DICT2[0].get(_,0) - yavg)
        denom1 += (DICT1[0].get(_,0) - xavg)**2
        denom2 += (DICT2[0].get(_,0) - yavg)**2

    return numer1 / np.sqrt(denom1) / np.sqrt(denom2)


def rsq_(DICT1, DICT2):
    '''RSQ of two sparsedict vectors of equal inner length.'''
    return r_(DICT1, DICT2) ** 2











