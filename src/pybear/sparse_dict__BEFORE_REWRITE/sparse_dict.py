import sys, inspect, warnings, time
import numpy as np, pandas as pd
from copy import deepcopy
from functools import wraps

from ..utilities import get_module_name as gmn
from pybear.data_validation import validate_user_input as vui, arg_kwarg_validater as akv
from pybear.new_numpy._random_ import sparse

from general_data_ops import list_sparsity as lsp, get_shape as gs



# *******
# ******* 10/11/22 DO NOT PUT np dtypes AS DICT KEYS!!!! ONLY USE py int.  DO NOT PUT np.float64 AS VALUE dtype!!!!
# ******* ONLY USE py float!!!  NON py NATIVE dtypes CRIPPLE DICT PROCESSING SPEED.  MEASUREMENTS INDICATE BY AT LEAST 50%
# *******



# FOR 500 COLS x 1000 ROWS:
# WHEN NOT SYMMETRIC W/ 10% SPARSE, NP.MATMUL = 36.62s, DENSE DOTS = 50.34s, DICT DOTS = 13.04s
# WHEN NOT SYMMETRIC W/ 90% SPARSE, NP.MATMUL = 35.41s, DENSE DOTS = 48.32s, DICT DOTS = 4.04s
# WHEN SYMMETRIC W/ 10% SPARSE,     NP.MATMUL = 36.97s, DENSE DOTS = 29.17s, DICT DOTS = 7.76s
# WHEN SYMMETRIC W/ 90% SPARSE,     NP.MATMUL = 34.60s, DENSE DOTS = 25.02s, DICT DOTS = 2.51s

'''
__init__ CHECKS & EXCEPTIONS ##########################################################################################################
module_name                     Return file name.
list_init                       Pre-run list config.
dict_init                       Pre-run dict config.
datadict_init                   Pre-run datadict config.
dataframe_init                  Pre-run dataframe config.
list_check                      Require that LIST arg is a list-type, and is not ragged.
is_sparse_dict                  Module for returning True/False indicator.
sparse_dict_check               Require that objects to be processed as sparse dictionaries follow sparse dictionary rules.
datadict_check                  Require that objects to be processed as data dictionaries follow data dict rules: dictionary with list-type as values.
dateframe_check                 Verify DATAFRAME arg is a dataframe.
END __init__ CHECKS & EXCEPTIONS ######################################################################################################
RUNTIME CHECKS & EXCEPTIONS ###########################################################################################################
non_int                         Verify integer.
insufficient_list_args_1        Verify LIST arg is filled when processing a function that requires a list.
insufficient_dict_args_1        Verify DICT1 arg is filled when processing a function that requires one dictionary.
insufficient_dict_args_2        Verify DICT1 and DICT2 args are filled when processing a function that requires two dictionaries.
insufficient_datadict_args_1    Verify DATADICT1 arg is filled when processing a function that requires one data dictionary.
insufficient_dataframe_args_1   Verify DATAFRAME arg is filled when processing a function that requires one dataframe.
dot_size_check                  Verify two vectors are sparse dicts that both have unitary outer length and equal inner length.
broadcast_check                 Verify two sparse dicts follow standard matrix multiplication rules (m, n) x (j, k) ---> n == j.
matrix_shape_check              Verify two sparse dicts have equal outer and inner length.
outer_len_check                 Verify two sparse dicts have equal outer length.
inner_len_check                 Verify two sparse dicts have equal inner length.
symmetric_matmul_check          Verify two sparse dicts will matrix multiply to a symmetric matrix.
END RUNTIME CHECKS & EXCEPTIONS #######################################################################################################
CREATION, HANDLING & MAINTENANCE ######################################################################################################
decorator_for_create_random     Create a positive-valued _random_ sparse matrix of with user-given type, min, max, dimensions and sparsity.
create_random                   Legacy. Create a _random_ sparse matrix of python integers from 0 to 9 with user-given dimensions and sparsity.
create_random_py_bin            Create a _random_ sparse matrix of python integers from 0 to 1 with user-given dimensions and sparsity.
create_random_py_int            Create a positive-valued _random_ sparse matrix of python integers with user-given min, max, dimensions and sparsity.
create_random_py_float          Create a positive-valued _random_ sparse matrix of python floats with user-given min, max, dimensions and sparsity.
create_random_np_bin            Create a _random_ sparse matrix of np.int8s from 0 to 1 with user-given dimensions and sparsity.
create_random_np_int32          Create a positive-valued _random_ sparse matrix of np.int32s from with user-given min, max, dimensions and sparsity.
create_random_np_float64        Create a positive-valued _random_ sparse matrix of np.float64s with user-given min, max, dimensions and sparsity.
decorator_for_zip_list          Base function for zip_list decorator functions.
zip_list                        Legacy. Convert list-type (list, tuple, array) of list-types as [[]] to sparse dict of py floats.
zip_list_as_py_int              Convert list-type (list, tuple, array) of list-types as [[]] to sparse dict of py ints.
zip_list_as_py_float            Convert list-type (list, tuple, array) of list-types as [[]] to sparse dict of py floats.
zip_list_as_np_int8             Convert list-type (list, tuple, array) of list-types as [[]] to sparse dict of int8s.
zip_list_as_np_int16            Convert list-type (list, tuple, array) of list-types as [[]] to sparse dict of int16s.
zip_list_as_np_int32            Convert list-type (list, tuple, array) of list-types as [[]] to sparse dict of int32s.
zip_list_as_np_int64            Convert list-type (list, tuple, array) of list-types as [[]] to sparse dict of int64s.
zip_list_as_np_float64          Convert list-type (list, tuple, array) of list-types as [[]] to sparse dict of float64s.
zip_datadict                    Convert data dict as {_:[], __:[], ...} to a sparse dictionary.
zip_dataframe                   Convert dataframe to sparse dict.
unzip_to_list                   Convert sparse dict to list of lists.
decorator_for_unzip_to_ndarray  Convert sparse dict to ndarray.
unzip_to_ndarray                Legacy. Convert sparse dict to ndarray.
unzip_to_ndarray_int8           Convert sparse dict to ndarray as int8s.
unzip_to_ndarray_int16          Convert sparse dict to ndarray as int16s.
unzip_to_ndarray_int32          Convert sparse dict to ndarray as int32s.
unzip_to_ndarray_int64          Convert sparse dict to ndarray as int64s.
unzip_to_ndarray_float64        Convert sparse dict to ndarray as float64s.
unzip_to_datadict               Convert sparse dict to datadict of lists.
unzip_to_dense_dict             Convert sparse dict to dict with full indices.
unzip_to_dataframe              Convert sparse dict to dataframe.
clean                           Remove any 0 values, enforce contiguous zero-indexed outer keys, build any missing final inner key placeholders.
resize_inner                    Resize sparse dict to user-entered inner dict length.  Reducing size may truncate non-zero values;
                                increasing size will introduce zeros (empties) and original inner size placeholder rule (entry for last item even if 0) holds.
#resize_outer                   Resize sparse dict to user-entered outer dict length.  Reducing size may truncate non-zero values;
                                increasing size will introduce zeros (placeholder inner dicts) and original outer size placeholder rule holds.
#resize                         Resize sparse dict to user-entered (len outer dict, len inner dicts) dimensions.  Reducing size may truncate non-zero values;
                                increasing size will introduce zeros (empties in inner dicts, placeholder in outer dict) and original size placeholder rules hold.
#drop_placeholders              Remove placeholding zeros
#merge_outer                    Merge outer dictionaries of 2 dictionaries with safeguards.  Inner dictionary lengths must be equal.
#core_merge_outer               Merge outer dictionaries of 2 dictionaries without safeguards.  Inner dictionary lengths must be equal.
#merge_inner                    Merge inner dictionaries of 2 dictionaries.  Outer dictionary lengths must be equal.
#delete_outer_key               Equivalent to deleting a row or a column.
#delete_inner_key               Equivalent to deleting a row or a column.
#insert_outer_inner_header_handle Validate size/format dict header and insert object header.   
#core_insert_outer              Insert a single inner dictionary as {0:x, 1:y, ...} at specified index without safeguards.
#insert_outer                   Insert a single inner dictionary at specified index with safeguards and header handling.
#append_outer                   Append an inner dictionary to a sparse dict in last position.
#core_insert_inner              Insert an entry into all inner dictionaries at specified index without safeguards.
#insert_inner                   Insert an entry into all inner dictionaries at specified index with safeguards and header handling.
#append_inner                   Append an entry into all inner dictionaries in the last position.
#split_outer                    Split before user-specified outer index
#split_inner                    Split before user-specified inner index
#multi_select_outer             Build sparse dict from user-specified outer indices of given sparse dict.
#core_multi_select_inner        Build sparse dict from user-specified inner indices of given sparse dict without safeguards.
#multi_select_inner             Build sparse dict from user-specified inner indices of given sparse dict with safeguards.
# END CREATION, HANDLING & MAINTENANCE ##################################################################################################
# ABOUT ###################################################################################################################################
is_sparse_outer                 Returns True if object is an outer sparse dictionary
is_sparse_inner                 Returns True if object is an inner sparse dictionary
inner_len_quick                 Length of inner dictionaries that are held by the outer dictionary assuming clean sparse dict.
inner_len                       Length of inner dictionaries that are held by the outer dictionary.
outer_len                       Length of the outer dictionary that holds the inner dictionaries as values, aka number of inner dictionaries
size_                           Return outer dict length * inner dict length.
shape_                          Return (outer dict length, inner dict length) as tuple.






sum_over_outer_key              Sum all the values in an inner dict, as given by outer dict key.
sum_over_inner_key              Sum over all inner dicts the values that are keyed with the user-entered inner key.
sparsity                        Calculate sparsity of a sparse dict.
list_sparsity                   Calculate sparsity of a list-type of list-types.
core_sparse_equiv               Check for equivalence of two sparse dictionaries without safeguards for speed.
safe_sparse_equiv               Check for equivalence of two sparse dictionaries with safeguards.
return_uniques                  Return unique values of a sparse dictionary as list.
display                         Print sparse dict to screen.
summary_stats                   Function called by decorators of specific summary statistics functions.
core_find_constants             Finds a column of constants. Returns dict/empty dict of non-zero constant indices, list/empty list of zero idxs.
find_constants                  Finds a column of constants with safeguards. Returns dict/empty dict of non-zero constant indices, list/empty list of zero idxs.
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
# END ABOUT #############################################################################################################################
# LINEAR ALGEBRA ########################################################################################################################
sparse_identity                 Identity matrix as sparse dictionary.
sparse_transpose_base           Function called by decorators of DICT1 or DICT2 transposing functions.
new_sparse_tranpose             Transpose a sparse dict to a sparse dict using shady tricks with numpy.
core_sparse_transpose           Transpose a sparse dict to a sparse dict by brute force without safeguards.
sparse_transpose                Transpose a sparse dict to a sparse dict by brute force with safeguards.
sparse_transpose2               Transpose a sparse dict to a sparse dict with Pandas DataFrames.
sparse_transpose3               Transpose a sparse dict to a sparse dict by converting to list, transpose, and convert back to dict.
sparse_transpose_from_list      Transpose a list to a sparse dict.
core_matmul                     DICT1 and DICT2 enter oriented with sparse inner dicts as rows, with multiplication rules like NumPy.  There is
                                no protection here to prevent dissimilar sized rows from DICT1 dotting with columns from DICT2.
                                Create posn for last entry, so that placeholder rules are enforced (original length of object is retained).
core_symmetric_matmul           DICT1 and DICT2 enter oriented with sparse inner dicts as rows, with multiplication rules like NumPy.
                                For use on things like ATA and AAT to save time.  There is no protection here to prevent dissimilar sized
                                rows from DICT1 dotting with columns from DICT2.  Create posn for last entry, so that placeholder rules are
                                enforced (original length of object is retained).
matmul                          Run matmul with safeguards that assure matrix multiplication rules are followed when running core_matmul().
symmetric_matmul                Run matmul with safeguards that assure matrix multiplication rules are followed when running core_symmetric_matmul().
sparse_ATA                      Calculates ATA on DICT1 using symmetric matrix multiplication.
sparse_AAT                      Calculates AAT on DICT1 using symmetric matrix multiplication.
spares_matmul_from_lists        Calculates matrix product of two lists and returns as sparse dict. 
core_dot                        Standard dot product. DICT1 and DICT2 enter as single-keyed outer dicts with dict as value.
                                There is no protection here to prevent dissimilar sized DICT1 and DICT2 from dotting.
dot                             Standard dot product.  DICT1 and DICT2 enter as single-keyed outer dicts with dict as value.
                                Run safeguards that assure dot product rules (dimensionality) are followed when running core_dot().
core_gaussian_dot               Gaussian dot product.  [] of DICT1 are dotted with [] from DICT2.  There is no protection here to prevent 
                                issimilar sized inner dicts from dotting.
gaussian_dot                    Gaussian dot product.  Run safeguards that assure dot product rules (dimensionality) are followed when running core_dot().
hybrid_matmul                   Left and right object oriented as []=row, with standard numpy matmul and linear algebra rules, and 
                                safeguards that assure matrix multiplication rules are followed when running core_hybrid_matmul().
core_hybrid_matmul              Left and right object oriented as []=row, with standard numpy matmul and linear algebra rules. There is
                                no protection here to prevent dissimilar sized rows and columns from dotting.
core_hybrid_dot                 Dot product of a single list and one outer sparse dict without any safeguards to ensure single vectors of same length.
# END LINEAR ALGEBRA ####################################################################################################################
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
'''


''' Take list-type of list-types, or dataframe, or datadict {'a':[]} , convert to a dictionary of sparse dictionaries.
    SparseDict converts data to dictionaries that hold non-zero values as values, with index positions as keys.
    e.g. [[0 0 1 0 3 0 1]] is {0: {2:1, 4:3, 6:1} }
    Always create posn for last entry, so that original length of list is retained.  I.e., if the last entry
    of a list is 0, put it in the dict anyway to placehold the original length of the list.
    Always create a dictionary for every list.  [[0,0,1],[0,0,0],[1,1,0]] looks like { 0:{2:1}, 1:[2:0}, 2:{0:1, 1:1, 2:0} }
    The chances of an entire list (whether it is meant to be a column or a row) being all zeros is small, meaning
    there would be little gained by dropping such rows, but the original dimensionality could be lost.
'''

def module_name():
    '''Return file name.'''
    return gmn.get_module_name(str(sys.modules[__name__]))


def list_init(LIST1=None, LIST_HEADER1=None, fxn=''):

    if LIST_HEADER1 is None: LIST_HEADER1 = [[]]

    if LIST1 is None: LIST1 = list()
    else: list_check(LIST1, fxn)

    return np.array(LIST1), LIST_HEADER1


def dict_init(DICT1=None, fxn=''):
    if DICT1 is None: DICT1 = dict()
    else: sparse_dict_check(DICT1, fxn)

    return DICT1


def datadict_init(DATADICT1=None, fxn=''):

    if DATADICT1 is None:
        DATADICT1 = dict()
        DATADICT_HEADER1 = [[]]
    else:
        datadict_check(DATADICT1, fxn)
        # BUILD DATADICT_HEADER1 AND REKEY DATADICT OUTER KEYS NOW TO LOCK IN SEQUENCE AND NOT LET NON-INT KEY GET PAST HERE!
        DATADICT_HEADER1 = [[]]
        key_counter = 0
        for key in list(DATADICT1.keys()):
            DATADICT_HEADER1[0].append(key)
            DATADICT1[key_counter] = DATADICT1.pop(key)
            key_counter += 1

    return DATADICT1, DATADICT_HEADER1


def dataframe_init(DATAFRAME1=None, fxn=''):

    if DATAFRAME1 is None:
        DATAFRAME1 = pd.DataFrame({})
        DATAFRAME_HEADER1 = [[]]
    else:
        dataframe_check(DATAFRAME1, fxn)
        # BUILD DATAFRAME_HEADER1 AND REKEY DATAFRAME OUTER KEYS NOW TO LOCK IN SEQUENCE AND NOT LET NON-INT KEY GET PAST HERE!
        DATAFRAME_HEADER1 = [[]]
        key_counter = 0
        for key in list(DATAFRAME1.keys()):
            DATAFRAME_HEADER1[0].append(key)
            DATAFRAME1[key_counter] = DATAFRAME1.pop(key)
            key_counter += 1

    return DATAFRAME1, DATAFRAME_HEADER1

#########################################################################################################################################
#########################################################################################################################################
# __init__ CHECKS & EXCEPTIONS ##########################################################################################################

def list_check(LIST1, fxn):
    '''Require that LIST arg is a list-type, and is not ragged.'''
    if not isinstance(LIST1, (list,tuple,np.ndarray)):
        raise Exception(f'{module_name()}.{fxn}() requires LIST arg be a list, array, or tuple of lists, arrays, or '
                            f'tuples, i.e. [[],[]] or ((),()). Cannot be set or dictionary.')

    LIST1 = np.array(LIST1, dtype=object)

    with np.errstate(all='ignore'):
        if LIST1.size == 0:
            raise Exception(f'{module_name()}.{fxn}() LIST1 arg is an empty list-type.')

    try:
        _LENS = list(map(len, LIST1))
        if min(_LENS) != max(_LENS):
            raise Exception(f'{module_name()}.{fxn}() requires non-ragged list-type of list-types.  '
                            f'Min={min(_LENS)}, Max = {max(_LENS)}.')
    except:
        pass


def is_sparse_dict(OBJECT):
    'Module for returning True/False indicator.'

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        try:
            # SEE IF IS DICT
            # OUTER KEYS ARE INTEGERS
            # OUTER VALUES ARE DICTS
            # INNER KEYS ARE INTEGERS
            # INNER VALUES ARE NUMBERS
            if not isinstance(OBJECT, dict) or \
            not isinstance(sum(OBJECT.keys()), (int, np.int8, np.int16, np.int32, np.int64)) or \
            False in map(isinstance, OBJECT.values(), (dict for _ in OBJECT)) or \
            not isinstance(sum(map(sum, map(dict.keys, OBJECT.values()))), (int, np.int8, np.int16, np.int32, np.int64)) or \
            not isinstance(sum(map(sum, map(dict.values, OBJECT.values()))),
                                  (int, float, np.int8, np.int16, np.int32, np.int64, np.float64)):
                    return False

            # IF GET TO THIS POINT
            return True
        except:
            return False


def sparse_dict_check(DICT1, fxn):
    '''Require that objects to be processed as sparse dictionaries follow sparse dictionary rules.'''

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        if not isinstance(DICT1, dict):
            raise Exception(f'{module_name()}.{fxn}() requires dictionary as input.)')

        if DICT1 == {}:
            raise Exception(f'{module_name()}.{fxn}() input is an empty dictionary.)')

        if False in map(isinstance, DICT1.values(), (dict for _ in DICT1)):
            raise Exception(f'{module_name()}.{fxn}() requires input to be a dictionary with values that are dictionaries.')

        if not isinstance(sum(DICT1.keys()), (int, np.int8, np.int16, np.int32, np.int64)):
            raise Exception(f'{module_name()}.{fxn}() requires all outer keys be integers.')

        # 10/8/22 sum/map/sum/map t=0.11 sec, for loop t=0.95 (tested in isolation)
        # for INNER_IDXS in map(dict.keys, DICT1.values()):
        #     if not isinstance(sum(INNER_IDXS), (int, np.int8, np.int16, np.int32, np.int64)):

        if not isinstance(sum(map(sum, map(dict.keys, DICT1.values()))), (int, np.int8, np.int16, np.int32, np.int64)):
            # RELYING ON py sum int int --> int, int float --> float, float float --> float EVEN IF FLOATS ADD TO AN INT
            raise Exception(f'{module_name()}.{fxn}() requires all inner keys be integers.')


def datadict_check(DATADICT1, fxn):
    '''Require that objects to be processed as data dictionaries follow data dict rules: dictionary with list-type as values.'''
    if not isinstance(DATADICT1, dict):
        raise Exception(f'{module_name()}.{fxn}() requires dictionary as input.)')

    if DATADICT1 == {}:
        raise Exception(f'{module_name()}.{fxn}() input is an empty dictionary.)')

    for _ in DATADICT1.values():
        if not isinstance(_, (np.ndarray, list, tuple)):
            raise Exception(f'{module_name()}.{fxn}() requires input to be a dictionary with values that are list-types.')


def dataframe_check(DATAFRAME1, fxn):
    '''Verify DATAFRAME arg is a dataframe.'''
    if 'DATAFRAME' not in str(type(DATAFRAME1)).upper():
        raise Exception(f'{module_name()}.{fxn}() requires input to be a Pandas DataFrame.')

# END __init__ CHECKS & EXCEPTIONS ######################################################################################################
#########################################################################################################################################
#########################################################################################################################################

#########################################################################################################################################
#########################################################################################################################################
# RUNTIME CHECKS & EXCEPTIONS ###########################################################################################################

def non_int(value, fxn, param_name):
    '''Verify integer.'''
    if not isinstance(value, (int, np.int8, np.int16, np.int32, np.int64)):
        raise Exception(f'{module_name()}.{fxn}() requires {param_name} entered as integer.')


def insufficient_list_args_1(LIST1, fxn):
    '''Verify LIST arg is filled when processing a function that requires a list.'''
    with np.errstate(all='ignore'):
        if not isinstance(LIST1, (np.ndarray, list, tuple)):
            raise Exception(f'{module_name()}.{fxn}() requires one list-type arg, LIST1.')
        if len(np.array(LIST1).reshape((1,-1))[0])==0:
            raise Exception(f'{module_name()}.{fxn}() input is an empty list-type.')


def insufficient_dict_args_1(DICT1, fxn):
    '''Verify DICT1 arg is filled when processing a function that requires one dictionary.'''
    if not isinstance(DICT1, dict):
        raise Exception(f'{module_name()}.{fxn}() requires one dictionary arg, DICT1.')
    if DICT1 == {}:
        raise Exception(f'{module_name()}.{fxn}() input is an empty dictionary.')


def insufficient_dict_args_2(DICT1, DICT2, fxn):
    '''Verify DICT1 and DICT2 args are filled when processing a function that requires two dictionaries.'''
    if not isinstance(DICT1, dict) or not isinstance(DICT2, dict):
        raise Exception(f'{module_name()}.{fxn}() requires two dictionary args, DICT1 and DICT2.')
    if DICT1 == {} or DICT2 == {}:
        raise Exception(f'{module_name()}.{fxn}() has at least one empty dictionary arg, DICT1 or DICT2.')


def insufficient_datadict_args_1(DATADICT1, fxn):
    '''Verify DATADICT1 arg is filled when processing a function that requires one data dictionary.'''
    if not isinstance(DATADICT1, dict):
        raise Exception(f'{module_name()}.{fxn}() requires one dictionary arg, DATADICT1.')
    if DATADICT1 == {}:
        raise Exception(f'{module_name()}.{fxn}() input is an empty dictionary.')


def insufficient_dataframe_args_1(DATAFRAME1, fxn):
    '''Verify DATAFRAME arg is filled when processing a function that requires one dataframe.'''
    if "DataFrame" not in str(type(DATAFRAME1)):
        raise Exception(f'{module_name()}.{fxn}() requires one DataFrame arg, DATAFRAME1.')
    if DATAFRAME1.equals(p.DataFrame({})):
        raise Exception(f'{module_name()}.{fxn}() input is an empty dictionary.')


def dot_size_check(DICT1, DICT2, fxn):
    '''Verify two vectors are sparse dicts that both have unitary outer length and equal inner length.'''

    if len(DICT1) != 1 or len(DICT2) != 1:
        raise Exception(f'{module_name()}.{fxn}() requires dictionaries with one integer key, one dict as values.)')
    if inner_len(DICT1) != inner_len(DICT2):
        raise Exception(f'{module_name()}.{fxn}() requires 2 dictionaries of equal stated length (last keys are equal).)')


def broadcast_check(DICT1, DICT2, fxn):        # DO THIS BEFORE TRANSPOSING DICT 2
    '''Verify two sparse dicts follow standard matrix multiplication rules (m, n) x (j, k) ---> n == j.'''
    if inner_len(DICT1) != outer_len(DICT2):
        raise Exception(f'{module_name()}.{fxn}() requires for DICT1(m x n) and DICT2(j x k) that num inner keys (n) of\n'
                        f'DICT1 == num outer keys (j) of DICT2 ---- (m, n) x (j, k) --> (m, k)\n'
                        f'{inner_len(DICT1)} is different than {outer_len(DICT2)}.')


def matrix_shape_check(DICT1, DICT2, fxn):
    '''Verify two sparse dicts have equal outer and inner length.'''
    _shape1, _shape2 = shape_(DICT1), shape_(DICT2)
    if _shape1 != _shape2:
        raise Exception(f'{module_name()}.{fxn}() requires both sparse dicts to be equally sized.  Dict 1 is {_shape1[0]} x '
                        f'{_shape1[1]} and Dict 2 is {_shape2[0]} x {_shape2[1]}')


def outer_len_check(DICT1, DICT2, fxn):
    '''Verify two sparse dicts have equal outer length.'''
    outer_len1 = outer_len(DICT1)
    outer_len2 = outer_len(DICT2)
    if outer_len1 != outer_len2:
        raise Exception(
            f'{module_name()}.{fxn}() requires both sparse dicts to have equal outer length.  Dict 1 is {outer_len1} '
            f'and Dict 2 is {outer_len2}')


def inner_len_check(DICT1, DICT2, fxn):
    '''Verify two sparse dicts have equal inner length.'''
    inner_len1 = inner_len(DICT1)
    inner_len2 = inner_len(DICT2)
    if inner_len1 != inner_len2:
        raise Exception(
            f'{module_name()}.{fxn}() requires both sparse dicts to have equal inner length.  Dict 1 is {inner_len1} '
            f'and Dict 2 is {inner_len2}')


def symmetric_matmul_check(DICT1, DICT2, DICT1_TRANSPOSE=None, DICT2_TRANSPOSE=None):
    '''Verify two sparse dicts will matrix multiply to a symmetric matrix.'''

    # GET DICT2_T FIRST JUST TO TEST DICT2 IS TRANSPOSE OF DICT1
    if not DICT2_TRANSPOSE is None: DICT2_T = DICT2_TRANSPOSE
    else: DICT2_T = sparse_transpose(DICT2)

    # TEST DICT2 IS TRANSPOSE OF DICT1
    DICT2 = sparse_transpose(DICT2)
    if safe_sparse_equiv(DICT1, DICT2_T): return True

    # IF DICT2 IS NOT TRANSPOSE OF DICT1, MAKE DICT1_T & PROCEED TO SYMMETRY TESTS
    if not DICT1_TRANSPOSE is None: DICT1_T = DICT1_TRANSPOSE
    else: DICT1_T = sparse_transpose(DICT1)

    # TEST BOTH DICT1 AND DICT2 ARE SYMMETRIC
    test2 = safe_sparse_equiv(DICT1, DICT1_T)
    if not test2: return False

    test3 = safe_sparse_equiv(DICT2, DICT2_T)
    if not test3: return False

    return True  # IF GET TO THIS POINT, MUST BE TRUE

# END RUNTIME CHECKS & EXCEPTIONS #######################################################################################################
#########################################################################################################################################
#########################################################################################################################################

#########################################################################################################################################
#########################################################################################################################################
# CREATION, HANDLING & MAINTENANCE ######################################################################################################

def decorator_for_create_random(orig_func):

    def timer(_time, _time2, words):
        del_t = time.time() - _time
        _total_time = time.time() - _time2
        # print(f"Time for {words} = {round(del_t, 2)} sec, total elapsed time = {round(_total_time, 2)} sec")
        return time.time()

    @wraps(orig_func)
    def create_random_x(_min, _max, shape_as_tuple, _sparsity):
        '''Create a positive-valued _random_ sparse matrix with user-given type, min, max, dimensions and sparsity.'''

        # 11/23/22 AFTER EXTENSIVE TIME TESTS ON VARIOUS BUILD METHODS.  UNDER ALL CIRCUMSTANCES, BUILDING A FULL-SIZED
        # np ARRAY AND MASKING (WHETHER THE MASKING BE BY np._random_.choice ON [0,1] OR BY NUMERICAL FILTER ON ANOTHER
        # RANDOM ARRAY OF NUMBERS) IS FASTER THAN CONSTRUCTING A SERIALIZED LIST OF IDXS AND VALUES (USING np._random_.choice
        # AND np._random_.randint, RESPECTIVELY.)  ADDITIONALLY, MAPPING THE DENSE VALUES OF A MASKED SPARSE ARRAY TO A SPARSE
        # DICT IS UNDER ALL CIRCUMSTANCES FASTEST WHEN USING np.nonzero TO GET POSNS AND VALUES, THEN USING dict(()) TO MAP.
        # 11/24/22 DECIDED THAT FOR SIZE ABOVE A CERTAIN THRESHOLD, USE "CHOICE" TO BUILD, IS FASTER, AND LAW OF AVERAGES
        # AT LARGER SIZES GETS SPARSITY CLOSE, BUT FOR SMALLER SIZE, USE "SERIALIZED" TO GET MORE ACCURATE SPARSITY

        # SUMMARY OF OPERATIONS
        # 1) VALIDATE PARAMS
        # 2) BUILD OBJECT BY "FULLY SIZED RANDOM ND ARRAY USING np._random_.choice ON [0,1] W p GIVEN BY SPARSITY TO BUILD MASK" METHOD
        #    i) BUILD A FULLY SIZED OUTPUT OBJECT AND FILL W VALUES AND DTYPES AS DETERMINED BY THE CALLING FUNCTION
        #   ii) BUILD A FULLY SIZED MASK RANDOMLY FROM [O,1], WITH PROBABLITIES GIVEN BY 1-_sparsity, _sparsity
        #  iii) APPLY MASK TO ORIGINAL FULLY SIZED OUTPUT OBJECT
        #   iv) FILL A SPARSE DICT
        #       a) USE dict(()) TO BUILD INNER DICTS BY GETTING np.nonzero AND CORRESPONDING VALUES FOR EACH ARRAY
        #          IN fully_sized_array (FASTEST METHOD)
        #       B) FIX PLACEHOLDERS
        # 3) RETURN

        fxn = inspect.stack()[0][3]

        t0 = time.time()
        t1 = time.time()

        _len_outer, _len_inner = shape_as_tuple

        total_size = int(_len_outer * _len_inner)

        non_int(_len_outer, fxn, "len_outer")
        non_int(_len_inner, fxn, "len_inner")

        __ = str(orig_func).upper()

        # INPUT VERIFICATION ################################################################################################
        while True:
            if _sparsity > 100 or _sparsity < 0:
                print(f'{module_name()}.{fxn}() sparsity is {_sparsity}, must be 0 to 100.')
                _sparsity = vui.validate_user_float(f'Enter sparsity (0 to 100) > ', min=0, max=100)

            if _len_outer == 0:
                print(f'{module_name()}.{fxn}() outer len is {_len_outer}, must be greater than 0.')
                _len_outer = vui.validate_user_int(f'Enter outer len > ', min=1)

            if _len_inner == 0:
                print(f'{module_name()}.{fxn}() inner len is {_len_inner}, must be greater than 0.')
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
                            f'User selected outer size = {_len_outer}, inner_size = {_len_inner}, sparsity = {_sparsity}. Accept? (y/n) > ', 'YN') == Y:
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
        # 1) THE np._random_.choice FUNCTION WITHIN CANT HANDLE LOOKING IN SIZE > 100M
        # 2) FOR ALL SIZES AND SPARSITIES, PULLING THE LIST OF DENSE IDXS AND VALUES USING np._random_.choice TAKES AT BEST 1.6X
        #   LONGER THAN use_fully_sized_ndarray_w_mask DOES TO GENERATE A FULLY SIZED DENSE np array, GENERATE A FULLY SIZED
        #   RANDOM MASK OF 1s AND 0s, AND APPLY THE MASK.
        # THE ONLY THING THAT COULD REDEEM serialized IS IF MAPPING THE DENSE IDXS AND VALUES (THAT TAKE LONGER TO
        # MAKE) TO A SPARSE DICT GOES MUCH FASTER THAN ZIPPING A FULLY SIZED np array TO SPARSE DICT. (IT DOESNT, SLOWER TOO).

        ''' 11/23/22 A RELIC OF THE PAST.  PROVED TO BE SLOWER IN BOTH CONSTRUCTING DENSE AND MAPPING TO SPARSE DICT THAN ARRAYS.
        ##########################################################################################################################
        # DENSE SERIALIZED LOCATIONS #############################################################################################
        elif use_serialized_dense_locations:  # THE NEW WAY, CREATE AN EXACTLY SIZED RANDOMLY SELECTED SERIALIZED LIST
            # OF DENSE LOCATIONS & RESPECTIVE VALUES  ( :( np._random_.choice IS SLOWER THAN full_np_array_w_mask METHOD

            # CREATE A SINGLE SERIALIZED VECTOR CONTAINING THE DENSE POSITIONS IN FINAL SPARSE DICT #############################
            # CREATE RANDOM VALUES BASED ON USER WANT int/float, MATCHING THE DENSE SIZE ########################################
            # CREATE AS NUMPY, THEN CONDITIONALLY CONVERT TO py int, py float, np int, np float

            RAND_SERIALIZED_DENSE_POSNS, RAND_SERIALIZED_VALUES = bspv(
                                                    1 if 'BIN' in __ else _min,
                                                    2 if 'BIN' in __ else _max,
                                                    shape_as_tuple,
                                                    _sparsity,
                                                    np.int8 if 'BIN' in __ else np.int32 if 'INT' in __ else np.float64)

            t0 = timer(t0, t1, 'select and sort _random_ dense locations, generate rand values for each dense location')
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
        # ON USER WANT bin/int OR float. CREATE A RANDOMLY FILLED MASK OF ONES AND ZEROS USING np._random_.choice ON [0,1] WITH
        # LIKELIHOODS GIVEN BY _sparsity AND 1-_sparsity TO INDICATE A DENSE (STAYING) POSN IN THE FIRST ARRAY.
        # 11/21/22 FINALIZED A FUNCTION TO HANDLE THIS, general_data_ops.create_random_sparse_numpy.create_random_sparse_numpy.
        # CREATE AS NUMPY ints AND floats, THEN COVERT TO py LATER IF NEEDED.

        VALUES = crsn(1 if 'BIN' in __ else _min,
                      2 if 'BIN' in __ else _max,
                      shape_as_tuple,
                      _sparsity,
                      np.int8 if 'BIN' in __ else np.int32 if 'INT' in __ else np.float64
                      )

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
    '''Legacy. Create a _random_ sparse matrix of python integers from 0 to 9 with user-given dimensions and sparsity.'''
    return create_random_py_int(0, 10, shape_as_tuple, sparsity)

@decorator_for_create_random
def create_random_py_bin(value):
    '''Create a _random_ sparse matrix of python integers from 0 to 1 with user-given dimensions and sparsity.'''
    return int(value)

@decorator_for_create_random
def create_random_py_int(value):
    '''Create a positive-valued _random_ sparse matrix of python integers with user-given min, max, dimensions and sparsity.'''
    return int(value)

@decorator_for_create_random
def create_random_py_float(value):
    '''Create a positive-valued _random_ sparse matrix of python floats with user-given min, max, dimensions and sparsity.'''
    return float(value)

@decorator_for_create_random
def create_random_np_bin(value):
    '''Create a _random_ sparse matrix of np.int8s from 0 to 1 with user-given dimensions and sparsity.'''
    return np.int8(value)

@decorator_for_create_random
def create_random_np_int32(value):
    '''Create a positive-valued _random_ sparse matrix of np.int32s from with user-given min, max, dimensions and sparsity.'''
    return np.int32(value)

@decorator_for_create_random
def create_random_np_float64(value):
    '''Create a positive-valued _random_ sparse matrix of np.float64s with user-given min, max, dimensions and sparsity.'''
    return np.float64(value)

'''    ***** TEST MODULE FOR ALL create_random FUNCTIONS *****
import numpy as np
import sparse_dict as sd
from data_validation import validate_user_input as vui

# TEST MODULE FOR UPDATED sparse_dict.create_random FXNS       CIRCA NOV 2022

min_max_window = 0.01
sparsity_window = .5

FXNS = [sd.create_random, sd.create_random_py_bin, sd.create_random_py_int, sd.create_random_py_float,
               sd.create_random_np_bin, sd.create_random_np_int32, sd.create_random_np_float64]
MIN = [2,0,4,0]
MAX = [10,10,5,2]
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
    print(f'\nWants to Except for actual {attr} ({act_value}) not {constraint} expected value of {expected_value}.')
    if vui.validate_user_str(f'Quit(q) continue(c) > ', 'QC') == 'Q':
        raise Exception(f'\nactual {attr} ({act_value}) not {constraint} expected value of {expected_value}.')


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
    else: raise Exception(f'\n*** GETTING function_name FROM str(FXN) LOGIC IS FAILING ***\n')
    # END ASSIGN A FUNCTION NAME FOR EACH PASS
    for _min, _max in zip(MIN, MAX):
        for len_outer, len_inner in zip(OUTER, INNER):
            for _sparsity in SPARSITY:
                ctr += 1
                print('\n****************************************************************************************')
                print(f'Running trial {ctr} of {np.product(list(map(len, [FXNS, MIN, SPARSITY, OUTER])))}... '
                      f'function name = {function_name}, _min={_min}, _max={_max}, len_outer={len_outer}, len_inner={len_inner}, sparsity={_sparsity}')

                if FXN is sd.create_random:                  #' create_random ' in str(FXN):
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
                RAW_OUTER_KEY_DTYPES = np.unique(np.fromiter((str(type(_)) for _ in RANDOM_SD.keys()), dtype='<U100'))
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
                    else: raise Exception(f'\n*** act_key_dtype LOGIC IS FAILING')
                    del RAW_INNER_KEY_DTYPES, number_of_inner_key_dtypes


                # GET INNER VALUE DTYPE
                RAW_INNER_VALUE_DTYPES = []
                for outer_key in RANDOM_SD:
                    RAW_INNER_VALUE_DTYPES += [str(type(RANDOM_SD[outer_key][inner_key])) for inner_key in
                                               RANDOM_SD[outer_key].keys()]
                RAW_INNER_VALUE_DTYPES = np.unique(RAW_INNER_VALUE_DTYPES)
                number_of_inner_value_dtypes = len(RAW_INNER_VALUE_DTYPES)
                if number_of_inner_value_dtypes > 1:
                    exception_handle('number of inner value dtypes', 'equal to', number_of_inner_value_dtypes, 1)
                else:
                    raw_value_dtype = RAW_INNER_VALUE_DTYPES[0]
                    if 'numpy.int8' in raw_value_dtype: act_value_dtype = 'np_int8'
                    elif 'numpy.int32' in raw_value_dtype: act_value_dtype = 'np_int32'
                    elif 'numpy.float64' in raw_value_dtype: act_value_dtype = 'np_float64'
                    elif raw_value_dtype == "<class 'int'>": act_value_dtype = 'py_int'
                    elif raw_value_dtype == "<class 'float'>": act_value_dtype = 'py_float'
                    else: raise Exception(f'\n*** act_value_dtype LOGIC IS FAILING ***\n')

                    del RAW_INNER_VALUE_DTYPES, number_of_inner_value_dtypes
                # END GET ACTUALS ################################################################################################

                # GET EXPECTEDS ###################################################################################################
                if 'bin' in function_name: exp_min, exp_max = 0,1
                elif 'int' in function_name: exp_min, exp_max = _min, _max-1
                elif 'float' in function_name: exp_min, exp_max = _min, _max
                if function_name == 'create_random': exp_min, exp_max = 0, 9
                if _sparsity == 0:
                    if 'bin' in function_name: exp_min = 1
                    elif function_name == 'create_random': exp_min = 1
                    elif 'int' in function_name: exp_min = _min if _min !=0 else 1
                    elif 'float' in function_name: exp_min = _min
                    else: raise Exception(f'\n*** LOGIC FOR SETTING MIN WHEN SPARSITY IS ZERO IS FAILING ***\n')
                elif _sparsity > 0 and _sparsity < 100: exp_min = 0
                elif _sparsity == 100: exp_min, exp_max = 0, 0

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
                print(f'initial min/max = {_min}/{_max}, expected min/max = {exp_min}/{exp_max}, actual min/max = {act_min}/{act_max}')
                print(f'expected shape = {len_outer}/{len_inner}, actual shape = {act_outer_len}/{act_inner_len}')
                print(f'expected sparsity = {_sparsity}, actual sparsity = {act_sparsity}')
                print(f'expected key dtype = {exp_key_dtype}, actual key dtype = {act_key_dtype}')
                print(f'expected value dtype = {exp_value_dtype}, actual value dtype = {act_value_dtype}')

                # TEST MIN/MAX INTEGER ################################################################################
                if 'int' in EXPECTED_VALUE_DICT[function_name] or 'bin' in EXPECTED_VALUE_DICT[function_name]:
                    if act_min != exp_min: exception_handle('_min', 'equal to', act_min, exp_min)
                    if act_max != exp_max: exception_handle('_max', 'equal to', act_max, exp_max)
                # END TEST MIN/MAX INTEGER ################################################################################

                # TEST MIN/MAX FLOAT ###################################################################################
                elif 'float' in EXPECTED_VALUE_DICT[function_name]:
                    if act_min < exp_min or act_min > exp_min + min_max_window:
                        exception_handle('_min', 'within window of', act_min, exp_min)
                    if act_max > exp_max or act_max < exp_max - min_max_window:
                        exception_handle('_max', 'within window of', act_max, exp_max)
                # END TEST MIN/MAX FLOAT ###################################################################################
                else:
                    raise Exception(f'\n*** LOGIC FOR TESTING exp_min/exp_max VS act_min/act_max IS FAILING ***/n')


                if len_outer != act_outer_len: exception_handle('len_outer', 'equal to', act_outer_len, len_outer)
                if len_inner != act_inner_len: exception_handle('len_inner', 'equal to', act_inner_len, len_inner)
                if act_sparsity > _sparsity + sparsity_window or act_sparsity < _sparsity - sparsity_window:
                    exception_handle('sparsity', 'within range of', act_sparsity, _sparsity)
                if act_key_dtype != exp_key_dtype: exception_handle('key_dtype', 'equal to', act_key_dtype, exp_key_dtype)
                if act_value_dtype != exp_value_dtype: exception_handle('value_dtype', 'equal to', act_value_dtype, exp_value_dtype)

                # ____ = vui.validate_user_str(f'type f > ', 'FA')


print(f'\nDone.')
'''


def decorator_for_zip_list(orig_func):
    '''Base function for zip_list decorator functions.
        Convert list-type (list, tuple, array) of list-types as [[]] to sparse dict.'''
    # COULD BE [ [] = ROWS ] OR [ [] = COLUMNS ], BUT MUST BE [[]]
    '''Create posn for last entry even if value is zero, so that original length of object is retained.'''

    def zip_list_as_x(LIST1, LIST_HEADER1=None):

        fxn = inspect.stack()[0][3]
        LIST1, LIST_HEADER1 = list_init(LIST1=LIST1, LIST_HEADER1=LIST_HEADER1, fxn=fxn)
        insufficient_list_args_1(LIST1, fxn)

        LIST1 = np.array(LIST1, dtype=object)
        _len_inner = len(LIST1[0])
        SPARSE_DICT = {}
        for outer_key in range(len(LIST1)):
            final_idx = len(LIST1[outer_key])-1
            SPARSE_DICT[int(outer_key)] = {}
            # OBSERVE PLACEHOLDER RULES, ALWAYS LAST INNER IDX INCLUDED
            NON_ZERO_KEYS = np.hstack((np.nonzero(LIST1[outer_key][:-1]), np.array(_len_inner-1).reshape((1,-1))))[0].astype(np.int32)
            if 'np_int' not in str(orig_func) and 'np_float' not in str(orig_func):
                VALUES = list(map([int if 'int' in str(orig_func) else float][0], LIST1[outer_key][NON_ZERO_KEYS].tolist()))
            else:
                VALUES = LIST1[outer_key][NON_ZERO_KEYS].astype(orig_func())

            SPARSE_DICT[int(outer_key)] = dict((
                    zip(NON_ZERO_KEYS.tolist(), VALUES)
            ))

        del NON_ZERO_KEYS, final_idx, VALUES

        return SPARSE_DICT

    return zip_list_as_x


@decorator_for_zip_list
def zip_list():
    '''Legacy. Convert list-type (list, tuple, array) of list-types as [[]] to sparse dict of py floats.'''
    return float

@decorator_for_zip_list
def zip_list_as_py_int():
    '''Convert list-type (list, tuple, array) of list-types as [[]] to sparse dict of py ints.'''
    return int

@decorator_for_zip_list
def zip_list_as_py_float():
    '''Convert list-type (list, tuple, array) of list-types as [[]] to sparse dict of py floats.'''
    return float

@decorator_for_zip_list
def zip_list_as_np_int8():
    '''Convert list-type (list, tuple, array) of list-types as [[]] to sparse dict of int8s.'''
    return np.int8

@decorator_for_zip_list
def zip_list_as_np_int16():
    '''Convert list-type (list, tuple, array) of list-types as [[]] to sparse dict of int16s.'''
    return np.int16

@decorator_for_zip_list
def zip_list_as_np_int32():
    '''Convert list-type (list, tuple, array) of list-types as [[]] to sparse dict of int32s.'''
    return np.int32

@decorator_for_zip_list
def zip_list_as_np_int64():
    '''Convert list-type (list, tuple, array) of list-types as [[]] to sparse dict of int64s.'''
    return np.int64

@decorator_for_zip_list
def zip_list_as_np_float64():
    '''Convert list-type (list, tuple, array) of list-types as [[]] to sparse dict of float64s.'''
    return np.float64


def zip_datadict(DATADICT1):
    '''Convert data dict as {_:[], __:[], ...} to a sparse dictionary.'''
    # {'a':LIST-TYPE, 'b':LIST-TYPE} COULD BE = ROWS OR COLUMNS
    #  Create posn for last entry even if value is zero, so that original length of object is retained.

    fxn = inspect.stack()[0][3]
    DATADICT1, DATADICT_HEADER1 = datadict_init(DATADICT1, fxn)
    insufficient_datadict_args_1(DATADICT1, fxn)

    DICT1 = {}
    for key in range(len(DATADICT1)):
        _ = DATADICT1[key]
        #   ik = inner_key, v = values, ilt = inner_list_type
        DICT1[int(key)] = {int(ik): float(v) for ik, v in enumerate(_) if (v != 0 or ik == len(_) - 1)}

    DICT1 = sparse_transpose(DICT1)   # COMES IN AS {'header': {COLUMNAR DATA LIST}} SO TRANSPOSE SPARSE DICT TO {} = ROWS

    sparse_dict_check(DICT1, fxn)   # TO VERIFY CONVERSION WAS DONE CORRECTLY

    return DICT1, DATADICT_HEADER1


def zip_dataframe(DATAFRAME1):
    '''Convert dataframe to sparse dict.  Returns sparse dict object and the extracted header as tuple.'''
    # PANDAS DataFrame
    # Create posn for last entry even if value is zero, so that original length of object is retained.

    fxn = inspect.stack()[0][3]
    DATAFRAME1, DATAFRAME_HEADER1 = dataframe_init(DATAFRAME1, fxn)
    insufficient_dataframe_args_1(DATAFRAME1, fxn)

    DICT1 = {}
    for key in range(len(list(DATAFRAME1.keys()))):
        _ = DATAFRAME1[key]
        #   ik = inner_key, v = values, ilt = inner_list_type
        DICT1[int(key)] = {int(ik): float(v) for ik, v in enumerate(_) if (v != 0 or ik == len(_) - 1)}

    #DataFrame CAN BE CREATED FROM:
    # = pd.DataFrame(data=LIST, columns=[])   CANNOT BE data=DICT or data=DICT OF DICTS (CANNOT PASS DICTS AS KWARGS)
    # = pd.DataFrame(DICT OF LISTS)      WITH DICT KEY AS HEADER  (NOT AS KWARG!)
    # = pd.DataFrame(DICT OF DICTS)      IS OK IF DENSE DICT, OTHERWISE SPARSE DICT GIVES NaN -- MUST USE .fillna(0) (NOT AS KWARG!)

    DICT1 = sparse_transpose(DICT1)   # DF COMES IN AS {'header': {COLUMNAR DATA LIST}} SO TRANSPOSE SPARSE DICT TO {} = ROWS

    sparse_dict_check(DICT1, fxn)   # TO VERIFY CONVERSION WAS DONE CORRECTLY

    return DICT1, DATAFRAME_HEADER1


def unzip_to_list(DICT1, LIST_HEADER1=None):
    '''Convert sparse dict to list of lists.'''

    fxn = inspect.stack()[0][3]
    DICT1 = dict_init(DICT1, fxn)
    insufficient_dict_args_1(DICT1, fxn)

    _outer_len = outer_len(DICT1)
    _inner_len = inner_len(DICT1)

    if not LIST_HEADER1 is None:
        if len(LIST_HEADER1[0]) != _outer_len and True not in [np.array_equiv(LIST_HEADER1, _) for _ in [None, [[]]] ]:
            raise Exception(f'# list columns ({_outer_len}) must equal # header positions ({len(LIST_HEADER1[0])}).')

    ND_ARRAY_ = np.zeros((_outer_len, _inner_len), dtype=np.float64)
    for outer_key in range(_outer_len):
        NDARRAY_[outer_key][np.fromiter((DICT1[outer_key]), dtype=np.int32)] = np.fromiter((DICT1[outer_key].values()), dtype=np.float64)

    LIST_OF_LISTS = list(map(list, ND_ARRAY_))

    return LIST_OF_LISTS, LIST_HEADER1


def decorator_for_unzip_to_ndarray(orig_func):
    '''Convert sparse dict to ndarray.'''
    def unzip_to_ndarray_as_x(DICT1, NDARRAY_HEADER1=None):
        fxn = inspect.stack()[0][3]
        DICT1 = dict_init(DICT1, fxn)
        insufficient_dict_args_1(DICT1, fxn)

        _outer_len = outer_len(DICT1)
        _inner_len = inner_len(DICT1)
        if not NDARRAY_HEADER1 is None:
            if len(NDARRAY_HEADER1[0]) != _outer_len and True not in [np.array_equiv(NDARRAY_HEADER1, _) for _ in [None, [[]]] ]:
                raise Exception(f'{module_name()}.{fxn}() list columns ({_outer_len}) must equal # header positions ({len(NDARRAY_HEADER1[0])}).')

        NDARRAY_ = np.zeros((_outer_len, _inner_len), dtype=orig_func())
        for outer_key in range(_outer_len):
            NDARRAY_[outer_key][np.fromiter((DICT1[outer_key]), dtype=np.int32)] = np.fromiter((DICT1[outer_key].values()), dtype=orig_func())

        return NDARRAY_, NDARRAY_HEADER1

    return unzip_to_ndarray_as_x


@decorator_for_unzip_to_ndarray
def unzip_to_ndarray():
    '''Legacy. Convert sparse dict to ndarray.'''
    return np.float64

@decorator_for_unzip_to_ndarray
def unzip_to_ndarray_int8():
    '''Convert sparse dict to ndarray as int8s.'''
    return np.int8

@decorator_for_unzip_to_ndarray
def unzip_to_ndarray_int16():
    '''Convert sparse dict to ndarray as int16s.'''
    return np.int16

@decorator_for_unzip_to_ndarray
def unzip_to_ndarray_int32():
    '''Convert sparse dict to ndarray as int32s.'''
    return np.int32

@decorator_for_unzip_to_ndarray
def unzip_to_ndarray_int64():
    '''Convert sparse dict to ndarray as int64s.'''
    return np.int64

@decorator_for_unzip_to_ndarray
def unzip_to_ndarray_float64():
    '''Convert sparse dict to ndarray as float64s.'''
    return np.float64


def unzip_to_datadict(DICT1, DATADICT_HEADER1=None):
    '''Convert sparse dict to datadict of lists.'''
    fxn = inspect.stack()[0][3]
    DICT1 = dict_init(DICT1, fxn)
    insufficient_dict_args_1(DICT1, fxn)

    # CAME IN COLUMNAR AS {'header':{DATA LIST} AND WAS TRANSPOSED TO OUTER {} = ROWS WHEN ZIPPED, SO TRANSPOSE BACK TO {} = COLUMN
    DICT1 = core_sparse_transpose(DICT1)

    _outer_len = outer_len(DICT1)
    _inner_len = inner_len(DICT1)

    is_empty = True in [np.array_equiv(DATADICT_HEADER1, _) for _ in [[[]], None]]

    if len(DATADICT_HEADER1[0]) != _outer_len and not is_empty:
        raise Exception(f'{module_name()}.{fxn}() Datadict columns ({_outer_len}) must equal # header positions ({len(DATADICT_HEADER1[0])}).')

    DATADICT1 = {}

    for outer_key in range(_outer_len):
        # IF A HEADER IS AVAILABLE, REPLACE OUTER INT KEYS W HEADER
        if not is_empty: DATADICT1[DATADICT_HEADER1[0][outer_key]] = []
        else:
            DATADICT1[outer_key] = outer_key
            DATADICT_HEADER1[0].append(outer_key)

    for outer_key in range(len(DICT1)):
        for inner_key in range(_inner_len):
            if inner_key not in list(DICT1[outer_key].keys()):
                DATADICT1[DATADICT_HEADER1[0][outer_key]].append(0)
            else:
                DATADICT1[DATADICT_HEADER1[0][outer_key]].append(DICT1[outer_key][inner_key])

    datadict_check(DATADICT1, inspect.stack()[0][3])

    return DATADICT1


def unzip_to_dense_dict(DICT1):
    '''Convert sparse dict to dict with full indices.'''

    fxn = inspect.stack()[0][3]
    DICT1 = dict_init(DICT1, fxn)
    insufficient_dict_args_1(DICT1, fxn)

    DICT1 = clean(DICT1)

    _outer_len = outer_len(DICT1)
    _inner_len = inner_len(DICT1)

    for outer_key in range(_outer_len):
        for inner_key in range(_inner_len):
            # REORDER INNER IDXS ON THE FLY
            if inner_key in DICT1[outer_key]:
                DICT1[int(outer_key)][int(inner_key)] = DICT1[outer_key].pop(inner_key)

            if inner_key not in DICT1[outer_key]:
                DICT1[int(outer_key)][int(inner_key)] = 0

    return DICT1


def unzip_to_dataframe(DICT1, DATAFRAME_HEADER1=None):
    '''Convert sparse dict to dataframe.'''

    fxn = inspect.stack()[0][3]
    DICT1 = dict_init(DICT1, fxn)
    insufficient_dict_args_1(DICT1, fxn)

    # CONVERT FULL DICT TO DATAFRAME

    DATADICT_HEADER1 = DATAFRAME_HEADER1
    DATADICT1 = unzip_to_datadict(DICT1, DATADICT_HEADER1=DATADICT_HEADER1)

    DATAFRAME1 = pd.DataFrame(DATADICT1)

    dataframe_check(DATAFRAME1, 'DATAFRAME1')

    return DATAFRAME1


def clean(DICT1):
    '''Remove any 0 values, enforce contiguous zero-indexed outer keys, build any missing final inner key placeholders.'''

    if DICT1 == {}: return DICT1

    DICT1 = dict_init(DICT1)
    insufficient_dict_args_1(DICT1, inspect.stack()[0][3])

    # CHECK IF KEYS START AT ZERO AND ARE CONTIGUOUS
    # FIND ACTUAL len, SEE IF OUTER KEYS MATCH EVERY POSN IN THAT RANGE
    # IF NOT, REASSIGN keys TO DICT, IDXed TO 0
    if False in np.fromiter((_ in DICT1 for _ in range(len(DICT1))), dtype=bool):
        for _, __ in enumerate(list(DICT1.keys())):
            DICT1[int(_)] = DICT1.pop(__)

    max_inner_key = int(np.max(np.fromiter((max(DICT1[outer_key]) for outer_key in DICT1),dtype=int)))
    for outer_key in DICT1:
        # ENSURE INNER DICT PLACEHOLDER RULE (KEY FOR LAST POSN, EVEN IF VALUE IS ZERO) IS ENFORCED
        DICT1[int(outer_key)][int(max_inner_key)] = DICT1[outer_key].get(max_inner_key, 0)
        # ENSURE THERE ARE NO ZERO VALUES IN ANY INNER DICT EXCEPT LAST POSITION
        VALUES_AS_NP = np.fromiter(DICT1[outer_key].values(), dtype=float)
        if 0 in VALUES_AS_NP[:-1]:
            KEYS_OF_ZEROES = np.fromiter((DICT1[outer_key]), dtype=int)[np.argwhere(VALUES_AS_NP[:-1]==0).transpose()[0]]
            np.fromiter((DICT1[outer_key].pop(_) for _ in KEYS_OF_ZEROES), dtype=object)
            del KEYS_OF_ZEROES

    del VALUES_AS_NP

    return DICT1


# CURRENTLY ONLY CALLED BY resize()
def resize_inner(DICT1, new_inner_len, calling_fxn=None, HEADER=None):   # LAST IDX IS ALWAYS len()-1, DUE TO ZERO INDEX
    '''Resize sparse dict to user-entered inner dict length.  Reducing size may truncate non-zero values;
        increasing size will introduce zeros (empties) and original inner size placeholder rule (entry for last item even if 0) holds.'''
    fxn = inspect.stack()[0][3]
    DICT1 = dict_init(DICT1, fxn)
    insufficient_dict_args_1(DICT1, fxn)
    non_int(new_inner_len, fxn, "new_inner_len")

    while True:
        if calling_fxn == 'DUMMY PLACEHOLDER':   # ALLOW USER SHORT CIRCUIT IN PROCESS.... NOT IN USE
            if vui.validate_user_str(f'\nReally proceed with inner dict resize?  Non-zero data will might be lost (y/n) > ', 'YN') == 'N':
                break

        DICT1 = clean(DICT1)

        old_inner_len = inner_len(DICT1)

        is_empty = True in [np.array_equiv(HEADER, _) for _ in [ [[]], None ] ]

        if new_inner_len == old_inner_len:  # NEW INNER LEN IS SAME AS OLD, DO NOTHING
            pass

        elif new_inner_len > old_inner_len:
            # DELETE OLD PLACEHOLDERS (key = old inner len - 1 and value == 0, if value != 0 then not a placeholder, dont delete)
            # PUT NEW PLACEHOLDER AT new_len_inner - 1
            for outer_key in DICT1:
                if DICT1[outer_key][old_inner_len - 1] == 0: del DICT1[outer_key][old_inner_len - 1]
                DICT1[int(new_inner_len-1)] = 0

            if not is_empty:
                for inner_key in range(old_inner_len, new_inner_len):
                    HEADER[0].append(inner_key)

        elif new_inner_len < old_inner_len:
            # DELETE ANYTHING AFTER old_inner_len - 1, PUT NEW PLACEHOLDERS as new_inner_len - 1 if NEEDED
            for outer_key in DICT1:
                for inner_key in range(new_inner_len, old_inner_len):
                    if inner_key in DICT1[outer_key]: del DICT1[outer_key][inner_key]
                if new_inner_len - 1 not in DICT1[outer_key]:
                    DICT1[int(outer_key)][int(new_inner_len-1)] = 0

            if not is_empty:
                for inner_key in range(new_inner_len, old_inner_len):
                    HEADER[0].pop(inner_key)

        break

    return DICT1, HEADER


# CURRENTLY ONLY CALLED BY resize()
def resize_outer(DICT1, new_outer_len, calling_fxn=None, HEADER=None):   # LAST IDX IS ALWAYS len()-1, DUE TO ZERO INDEX
    '''Resize sparse dict to user-entered outer dict length.  Reducing size may truncate non-zero values;
        increasing size will introduce zeros (placeholder inner dicts) and original outer size placeholder rule holds.'''

    fxn = inspect.stack()[0][3]
    DICT1 = dict_init(DICT1, fxn)
    insufficient_dict_args_1(DICT1, fxn)
    non_int(new_outer_len, fxn, "new_outer_len")

    while True:
        if calling_fxn == 'DUMMY_PLACEHOLDER':   # ALLOW USER SHORT CIRCUIT IN PROCESS.... NOT IN USE
            if vui.validate_user_str(f'\nReally proceed with outer dict resize?  Non-zero data will might be lost (y/n) > ', 'YN') == 'N':
                break

        DICT1 = clean(DICT1)

        old_outer_len = outer_len(DICT1)

        is_empty = True in [np.array_equiv(HEADER, _) for _ in [ [[]], None ] ]

        if new_outer_len == old_outer_len:    # NEW INNER LEN IS SAME AS OLD, DO NOTHING
            pass

        elif new_outer_len > old_outer_len:
            # PUT PLACEHOLDERS IN THE NEW KEYS
            for outer_key in range(old_outer_len, new_outer_len):
                DICT1[int(outer_key)] = {int(inner_len(DICT1)): 0}
                if not is_empty: HEADER[0].append(outer_key)

        elif new_outer_len < old_outer_len:
            for outer_key in range(new_outer_len, old_outer_len):
                del DICT1[outer_key]
                if not is_empty: HEADER[0].pop(outer_key)

        break

    return DICT1, HEADER


def resize(DICT1, len_outer_key, len_inner_key, HEADER=None, header_goes_on=None):  # LAST OUTER AND INNER IDXS ARE ALWAYS len()-1, DUE TO ZERO INDEXING
    '''Resize sparse dict to user-entered (len outer dict, len inner dicts) dimensions.  Reducing size may truncate non-zero values;
        increasing size will introduce zeros (empties in inner dicts, placeholder in outer dict) and original size placeholder rules hold.'''

    fxn = inspect.stack()[0][3]
    DICT1 = dict_init(DICT1, fxn)
    insufficient_dict_args_1(DICT1, fxn)
    non_int(len_outer_key, fxn, "len_outer_key")
    non_int(len_inner_key, fxn, "len_inner_key")

    if isinstance(header_goes_on, str): header_goes_on = header_goes_on.upper()

    if header_goes_on == 'OUTER':
        DICT1, HEADER = resize_outer(DICT1, len_outer_key, calling_fxn=fxn, HEADER=HEADER)
        DICT1, DUM = resize_inner(DICT1, len_inner_key, calling_fxn=fxn, HEADER=HEADER)
    elif header_goes_on == 'INNER':
        DICT1, DUM = resize_outer(DICT1, len_outer_key, calling_fxn=fxn, HEADER=HEADER)
        DICT1, HEADER = resize_inner(DICT1, len_inner_key, calling_fxn=fxn, HEADER=HEADER)
    elif header_goes_on is None:
        DICT1, DUM = resize_outer(DICT1, len_outer_key, calling_fxn=fxn, HEADER=HEADER)
        DICT1, HEADER = resize_inner(DICT1, len_inner_key, calling_fxn=fxn, HEADER=HEADER)
    else:
        raise ValueError(f'INVALID header_goes_on IN {module_name()}.{fxn}().  MUST BE "outer" or "inner".')

    return DICT1, HEADER


def drop_placeholders(DICT1):
    '''Remove placeholding zeros.'''

    fxn = inspect.stack()[0][3]
    DICT1 = dict_init(DICT1, fxn)
    insufficient_dict_args_1(DICT1, fxn)

    last_inner_key = inner_len(DICT1) - 1
    for outer_key in DICT1:
        if last_inner_key in DICT1[outer_key] and DICT1[outer_key][last_inner_key] == 0:  del DICT1[outer_key][last_inner_key]

    return DICT1


def merge_outer(DICT1, DICT2, HEADER1=None, HEADER2=None):
    '''Merge outer dictionaries of 2 dictionaries with safeguards.  Inner dictionary lengths must be equal.'''

    fxn = inspect.stack()[0][3]

    if DICT1 == {} and DICT2 == {}:         # IF DICT1 AND DICT2 ARE EMPTY
        insufficient_dict_args_2(DICT1, DICT2, fxn)
    elif not DICT1 == {} and DICT2 == {}:    # IF DICT2 IS EMPTY
        DICT1 = dict_init(DICT1, fxn)
        insufficient_dict_args_1(DICT1, fxn)
        DICT1 = clean(DICT1)
        return DICT1, HEADER1
    elif not DICT2 == {} and DICT1 == {}:    # IF DICT1 IS EMPTY
        DICT2 = dict_init(DICT2, fxn)
        insufficient_dict_args_1(DICT2, fxn)
        DICT2 = clean(DICT2)
        return DICT2, HEADER2
    else:
        DICT1 = dict_init(DICT1, fxn)
        DICT2 = dict_init(DICT2, fxn)
        insufficient_dict_args_2(DICT1, DICT2, fxn)
        inner_len_check(DICT1, DICT2, fxn)

        DICT1 = clean(DICT1)
        DICT2 = clean(DICT2)

        return core_merge_outer(DICT1, DICT2, HEADER1=HEADER1, HEADER2=HEADER2)


def core_merge_outer(DICT1, DICT2, HEADER1=None, HEADER2=None):
    '''Merge outer dictionaries of 2 dictionaries without safeguards.  Inner dictionary lengths must be equal.'''

    if not DICT1 == {} and DICT2 == {}:    # IF DICT2 IS EMPTY
        return DICT1, HEADER1
    elif not DICT2 == {} and DICT1 == {}:    # IF DICT1 IS EMPTY
        return DICT2, HEADER2
    else:
        # CANT JUST MERGE THE 2 DICTS, THEY MIGHT (PROBABLY) HAVE MATCHING OUTER KEYS AND OVERWRITE
        # GET outer_len of DICT1 TO KNOW HOW TO INDEX DICT2, REINDEX DICT2 ON THE FLY

        NEW_DICT2_KEYS = np.fromiter(DICT2.keys(), dtype=np.int16) + outer_len(DICT1)

        NEW_DICT = DICT1 | dict((zip(NEW_DICT2_KEYS, DICT2.values())))
        del NEW_DICT2_KEYS

        if not True in [np.array_equiv(HEADER1, _) for _ in [[[]], None]] and \
                not True in [np.array_equiv(HEADER2, _) for _ in [[[]], None]]:
            HEADER1 = np.array([*HEADER1[0], *HEADER2[0]], dtype='<U500')
            HEADER1.reshape((1, len(HEADER1)))

        return NEW_DICT, HEADER1


def merge_inner(DICT1, DICT2, HEADER1=None, HEADER2=None):
    '''Merge inner dictionaries of 2 dictionaries.  Outer dictionary lengths must be equal.'''
    fxn = inspect.stack()[0][3]
    DICT1 = dict_init(DICT1, fxn)
    DICT2 = dict_init(DICT2, fxn)
    insufficient_dict_args_2(DICT1, DICT2, fxn)
    DICT1 = clean(DICT1)
    DICT2 = clean(DICT2)
    outer_len_check(DICT1, DICT2, fxn)

    # CANT JUST MERGE THE 2 DICTS, THEY MIGHT (PROBABLY) HAVE MATCHING INNER KEYS AND OVERWRITE
    # GET inner_len of DICT1 TO KNOW HOW TO INDEX DICT2, REINDEX DICT2 ON THE FLY
    _inner_len1 = inner_len(DICT1)
    _inner_len2 = inner_len(DICT2)
    combined_inner_len = _inner_len1 + _inner_len2
    pseudo_dict2_outer_key = 0
    for outer_key in DICT2.keys():   # DICT1 outer len must == DICT 2 outer len
        # CURRENTLY UNABLE TO CLEAN DICT2, SO IF OUTER KEYS NOT CONTIGUOUS, USE PSEUDOKEY TO MATCH AGAINST DICT1
        # CHECK TO SEE IF VALUE AT END OF DICT1 INNER IS 0, IF SO, DELETE
        if DICT1[pseudo_dict2_outer_key][_inner_len1-1] == 0: del DICT1[pseudo_dict2_outer_key][_inner_len1-1]
        for inner_key in DICT2[outer_key]:
            DICT1[int(pseudo_dict2_outer_key)][int(_inner_len1 + inner_key)] = DICT2[outer_key][inner_key]
        else: # WHEN GET TO LAST INNER KEY, ENFORCE PLACEHOLDING RULES
            DICT1[int(pseudo_dict2_outer_key)][int(combined_inner_len - 1)] = \
                DICT1[pseudo_dict2_outer_key].get(combined_inner_len - 1, 0)

        pseudo_dict2_outer_key += 1

    if not True in [np.array_equiv(HEADER1, _) for _ in [[[]], None]] and \
            not True in [np.array_equiv(HEADER2, _) for _ in [[[]], None]]:
        HEADER1 = [[*HEADER1[0], *HEADER2[0]]]

    return DICT1, HEADER1


def delete_outer_key(DICT, OUTER_KEYS_TO_DELETE_AS_LIST, HEADER=None):
    '''Equivalent to deleting a row or a column.'''

    DICT1 = deepcopy(DICT)   # TO PREVENT BLOWBACK TO ORIGINAL OBJECT, DEMONSTRATED TO BE HAPPENING 10/15/22

    fxn = inspect.stack()[0][3]
    insufficient_dict_args_1(DICT1, fxn)
    insufficient_list_args_1(OUTER_KEYS_TO_DELETE_AS_LIST, fxn)
    for delete_key in OUTER_KEYS_TO_DELETE_AS_LIST:
        non_int(delete_key, fxn, "key")

    DICT1 = clean(DICT1)
    _outer_len = outer_len(DICT1)

    _min_delete_key = min(OUTER_KEYS_TO_DELETE_AS_LIST)
    _max_delete_key = max(OUTER_KEYS_TO_DELETE_AS_LIST)

    if _min_delete_key < 0:
        raise Exception(f'Outer key {_min_delete_key} out of bounds for {module_name()}.{fxn}(). Must be >= 0.')
    if _max_delete_key > _outer_len - 1:
        raise Exception(f'Outer key {_max_delete_key} out of bounds for {module_name()}.{fxn}(). Must be <= {_outer_len-1}.')

    outer_key_adjustment = 0
    for outer_key in range(_min_delete_key, _outer_len):   # MUST ITERATE OVER ALL KEYS AFTER LOWEST, TO CAPTURE CORRECT AMOUNT TO SUBTRACT
        if outer_key in OUTER_KEYS_TO_DELETE_AS_LIST:
            del DICT1[outer_key]
            outer_key_adjustment += 1
        if outer_key not in OUTER_KEYS_TO_DELETE_AS_LIST:
            DICT1[int(outer_key - outer_key_adjustment)] = DICT1.pop(outer_key)

        if not True in [np.array_equiv(HEADER, _) for _ in [ [[]], None ] ]:
            HEADER[0].pop(delete_key)

    return DICT1, HEADER


def delete_inner_key(DICT, INNER_KEYS_TO_DELETE_AS_LIST, HEADER=None):
    '''Equivalent to deleting a row or a column.'''

    DICT1 = deepcopy(DICT)   # TO PREVENT BLOWBACK TO ORIGINAL OBJECT, DEMONSTRATED TO BE HAPPENING 10/15/22

    fxn = inspect.stack()[0][3]
    DICT1 = dict_init(DICT1, fxn)
    insufficient_dict_args_1(DICT1, fxn)
    insufficient_list_args_1(INNER_KEYS_TO_DELETE_AS_LIST, fxn)
    for delete_key in INNER_KEYS_TO_DELETE_AS_LIST:
        non_int(delete_key, fxn, "key")

    DICT1 = clean(DICT1)
    _inner_len = inner_len(DICT1)

    _min_delete_key = min(INNER_KEYS_TO_DELETE_AS_LIST)
    _max_delete_key = max(INNER_KEYS_TO_DELETE_AS_LIST)

    if _min_delete_key < 0:
        raise Exception(f'Inner key {_min_delete_key} out of bounds for {module_name()}.{fxn}(). Must be >= 0.')
    if _max_delete_key > _inner_len - 1:
        raise Exception(f'Inner key {_max_delete_key} out of bounds for {module_name()}.{fxn}(). Must be <= {_inner_len - 1}.')

    # MUST REDUCE ALL keys AFTER delete_key BY 1!!! DONT JUST SET delete_key TO ZERO BY JUST DELETING IT!!!

    for outer_key in DICT1.copy():    # ONLY COPY OUTER THINGS
        inner_key_adjustment = 0
        for inner_key in range(_min_delete_key, _inner_len):
            if inner_key in INNER_KEYS_TO_DELETE_AS_LIST:
                inner_key_adjustment += 1
            if inner_key in DICT1[outer_key] and inner_key in INNER_KEYS_TO_DELETE_AS_LIST:
                del DICT1[outer_key][inner_key]
            if inner_key in DICT1[outer_key] and inner_key not in INNER_KEYS_TO_DELETE_AS_LIST:
                DICT1[int(outer_key)][int(inner_key - inner_key_adjustment)] = DICT1[outer_key].pop(inner_key)

            if inner_key == 0 and not True in [np.array_equiv(HEADER, _) for _ in [ [[]], None ] ]:  # DO THIS ON ONLY ONE PASS
                HEADER[0].pop(inner_key)

        if DICT1[outer_key] == {}:
            del DICT1[outer_key]

        else:
            # WHEN DELETING THE LAST inner_key IN INNER DICT, MANAGE PLACEHOLDER RULES
            if _inner_len - 1 - inner_key_adjustment not in DICT1[outer_key]:
                DICT1[int(outer_key)][int(inner_key - inner_key_adjustment)] = 0

    # REORDER INNER IDXS
    if DICT1 != {}:
        for inner_key in sorted(DICT1[outer_key].copy()):
            DICT1[outer_key][inner_key] = DICT1[outer_key].pop(inner_key)

    return DICT1, HEADER


def insert_outer_inner_header_handle(DICT_HEADER1, INSERT_HEADER, dict1_outer_len, dict1_inner_len, header_axis, active_axis,
                                     ins_len, fxn):
    '''Validate size/format of insert object header and receiving object header.'''

    # TEST IS INCIDENTAL WITH insert_outer AND insert_inner MODULES.

    __ = f'sparse_dict.{fxn}()'

    # IF DICT1_HEADER OR INSERT_HEADER ARE NOT None, header_axis MUST BE PROVIDED AND MUST BE 0 OR 1
    if (not DICT_HEADER1 is None or not INSERT_HEADER is None):
        if header_axis not in [0, 1]:
            raise ValueError(f'IF ONE OR BOTH HEADER OBJECTS ARE GIVEN, header_axis MUST BE PROVIDED AND MUST BE 0 OR 1.')
        if active_axis not in [0, 1]:
            raise ValueError(f'active_axis MUST BE PROVIDED AND MUST BE 0 OR 1.')

    # ENSURE DICT_HEADER IS LIST-TYPE AND CONVERT TO np.[[]]
    if not DICT_HEADER1 is None:
        if not isinstance(DICT_HEADER1, (list, tuple, np.ndarray)):
            raise TypeError(f'{__} DICT_HEADER1 MUST BE A LIST-TYPE ENTERED AS [] OR [[]]')
        else: DICT_HEADER1 = np.array(DICT_HEADER1, dtype='<U200').reshape((1,-1))

        # CHECK IF DICT_HEADER1 MATCHES LEN OF ASSIGNED AXIS ( 0 --> len_outer == len(DICT1_HEADER), 1 --> len_inner == len(DICT1_HEADER) )
        _text = lambda axis: f'{__}: DICT_HEADER1 LENGTH MUST MATCH {"OUTER" if axis==0 else "INNER"} LENGTH OF GIVEN DICT WHEN ' \
                                f'header_axis IS {axis}. MAYBE PROVIDING HEADER WHEN NOT NEEDED, OR SPECIFYING WRONG AXIS?'
        if header_axis == 0 and len(DICT_HEADER1[0]) != dict1_outer_len: raise ValueError(_text(0))
        elif header_axis == 1 and len(DICT_HEADER1[0]) != dict1_inner_len: raise ValueError(_text(1))
        del _text

    # ENSURE INSERT_HEADER IS LIST-TYPE AND CONVERT TO np.[[]]
    if not INSERT_HEADER is None:
        if not isinstance(INSERT_HEADER, (list, tuple, np.ndarray)):
            raise TypeError(f'{__} INSERT_HEADER MUST BE A LIST-TYPE ENTERED AS [] OR [[]]')
        else: INSERT_HEADER = np.array(INSERT_HEADER, dtype='<U200').reshape((1, -1))

        # CHECK IF INS_HEADER MATCHES LEN OF INS_OBJ
        if len(INSERT_HEADER[0]) != ins_len: raise ValueError(f'{__}: INS_HEADER LENGTH MUST MATCH INS_OBJ LENGTH.')

    # IF INSERT_HEADER IS PROVIDED BUT NOT DICT_HEADER1, MAKE DUMMY DICT_HEADER1
    if not INSERT_HEADER is None and DICT_HEADER1 is None:
        warnings.warn(f'{__}: HEADER OF INSERTED OBJECT WAS PROVIDED AND HEADER OF RECEIVING OBJECT WAS NOT.')
        if header_axis==0: DICT_HEADER1 = np.fromiter((f'DICT1_COL_{idx+1}' for idx in range(dict1_outer_len)), dtype='<U20').reshape((1,-1))
        elif header_axis == 1: DICT_HEADER1 = np.fromiter((f'DICT1_COL_{idx+1}' for idx in range(dict1_inner_len)), dtype='<U20').reshape((1,-1))

    # IF DICT_HEADER1 WAS PROVIDED BUT NOT INSERT_HEADER, MAKE DUMMY INSERT_HEADER
    if INSERT_HEADER is None and not DICT_HEADER1 is None:
        warnings.warn(f'{__}: HEADER OF RECEIVING OBJECT WAS PROVIDED AND HEADER OF INSERTED OBJECT WAS NOT.')
        if header_axis != active_axis: INSERT_HEADER = [[]]
        elif header_axis == active_axis: INSERT_HEADER = np.fromiter((f'INS_COL_{idx+1}' for idx in range(ins_len)), dtype='<U20').reshape((1,-1))

    if DICT_HEADER1 is None and INSERT_HEADER is None:
        pass
        # DONT BUILD ANY HEADERS IF NONE WERE PASSED

    return DICT_HEADER1, INSERT_HEADER


def core_insert_outer(INPUT_DICT, index, INS_OBJ):
    '''Insert a single inner dictionary as {0:x, 1:y, ...} at specified index without safeguards.'''

    # TEST CODE IS LOCATED IN sparse_dict_test_modules.insert_outer__function_test().

    # 10/11/22 IN SITUATION WHERE DOING y = sd.(core)_insert(append)_outer(x,....) IN ANOTHER MODULE, EVEN THO RETURNING
    # AS y, THE INPUT x IS BEING MODIFIED AS WELL :(.  PRINTING x IN OTHER MODULE AFTER RETURNING y SHOWED x HAD CHANGED.
    # PUTTING deepcopy HERE BREAKS THAT CHAIN (AND BLOWS UP MEMORY :( )

    DICT1 = deepcopy(INPUT_DICT)
    del INPUT_DICT


    if isinstance(INS_OBJ, dict):
        if is_sparse_inner(INS_OBJ):
            INS_OBJ = {0: INS_OBJ}
        if len(INS_OBJ)==1: INS_OBJ[0] = INS_OBJ.pop(list(INS_OBJ.keys())[0])
    elif isinstance(INS_OBJ, (np.ndarray, list, tuple)):
        INS_OBJ = np.array(INS_OBJ)
        if len(INS_OBJ.shape)==1: INS_OBJ = INS_OBJ.reshape((1, -1))
        if 'INT' in str(INS_OBJ.dtype).upper(): INS_OBJ = zip_list_as_py_int(INS_OBJ)
        elif 'FLOAT' in str(INS_OBJ.dtype).upper(): INS_OBJ = zip_list_as_py_float(INS_OBJ)
    else: raise TypeError(f'\n{module_name()}.core_insert_outer() INVALID OBJECT TYPE ({type(INS_OBJ)}) PASSED; '
                                    f'MUST BE LIST-TYPE OR SPARSE DICTIONARY.')

    if DICT1 is None or DICT1 == {}:  # IF NO RECEIVING DICT WAS PASSED, THEN THE INSERT SIMPLY BECOMES THE NEW DICT
        return INS_OBJ


    dict1_outer_len, dict1_inner_len = shape_(DICT1)
    ins_obj_outer_len, ins_obj_inner_len = shape_(INS_OBJ)

    # CHECK IF INSERTION IDX IS IN RANGE # ############################################
    if index > dict1_outer_len or index < 0:
        raise ValueError(f'\n{module_name()}.core_insert_outer() OUTER INSERT INDEX {index} OUT OF RANGE FOR SPARSE DICT OF '
                                  f'OUTER LEN {dict1_outer_len:,.0f}')
    #### VALIDATE INSERT AND DICT LENS #############################################################################################

    # SEE WHAT ORIENTATION OF INS_OBJ MATCHES UP AGAINST INNER LEN OF DICT1
    if ins_obj_inner_len==dict1_inner_len: pass  # INS_OBJ WAS PASSED ORIENTED THE SAME AS DICT1
    elif ins_obj_outer_len==dict1_inner_len:   # INS_OBJ WAS PASSED ORIENTED AS ROW, CHANGE TO MATCH DICT1
        INS_OBJ = core_sparse_transpose(INS_OBJ)
    else:   # NONE OF THE DIMENSIONS OF INS_OBJ MATCH THE INNER LEN OF DICT1
        raise valueError(f'\n{module_name()}.core_insert_outer() INS_OBJ WITH DIMENSIONS {shape_(INS_OBJ)} CANNOT '
                        f'BE FORCED TO FIT A RECEIVING DICT WITH INNER LEN OF {dict1_inner_len}.')

    # GET SHAPE OF INS_OBJ AGAIN AFTER ANY REORIENTING THAT MAY HAVE HAPPENED
    ins_obj_outer_len, ins_obj_inner_len = shape_(INS_OBJ)

    if DICT1 != {} and ins_obj_inner_len != dict1_inner_len:  # ALLOWS APPEND TO {}
        raise ValueError(f'\n{module_name()}.core_insert_outer() '
            f'LENGTH OF INSERTED OBJECT ({ins_obj_inner_len:,.0f}) DOES NOT EQUAL INNER LENGTH OF RECEIVING OBJECT ({dict1_inner_len:,.0f})')

    # APPENDING
    if index == dict1_outer_len:
        for col_idx in set(INS_OBJ.keys()):
            DICT1[int(dict1_outer_len + col_idx)] = INS_OBJ.pop(col_idx)
    # INSERTING
    else:
        DICT2 = {}
        for dict1_outer_key in range(index, dict1_outer_len):
            DICT2[int(dict1_outer_key + ins_obj_outer_len)] = DICT1.pop(dict1_outer_key)

        # INCREMENT INS_OBJs OUTER IDXS
        INS_OBJ = dict((zip(np.add(np.fromiter(INS_OBJ.keys(), dtype=np.int32), index).tolist(), INS_OBJ.values())))

        DICT1 = DICT1 | INS_OBJ | DICT2

        del INS_OBJ, DICT2

    return DICT1


def insert_outer(DICT1, index, LIST_OR_DICT_TO_INSERT, DICT_HEADER1=None, INSERT_HEADER=None,
                 header_axis=None, fxn=None):
    '''Insert a single inner dictionary at specified index with safeguards and header handling.'''

    # TEST CODE IS LOCATED IN sparse_dict_test_modules.insert_outer__function_test().

    # ASSUMES DICT1 IS "CLEAN"
    fxn = inspect.stack()[0][3] if fxn is None else fxn
    DICT1 = dict_init(DICT1, fxn)
    insufficient_dict_args_1(DICT1, fxn)
    dict1_outer_len = outer_len(DICT1)
    dict1_inner_len = inner_len(DICT1)

    DICT1 = core_insert_outer(DICT1, index, LIST_OR_DICT_TO_INSERT)

    del LIST_OR_DICT_TO_INSERT

    # IF INS_HEADER WAS NOT PASSED, MEASURE THE DELTA OF DICT1 BEFORE & AFTER INSERT TO FIND
    # OUT HOW LONG INS_OBJ WAS. USE THIS DELTA TO CREATE THE DUMMY HEADER.
    DICT_HEADER1, INSERT_HEADER = insert_outer_inner_header_handle(DICT_HEADER1, INSERT_HEADER, dict1_outer_len,
                                   dict1_inner_len, header_axis, 0, outer_len(DICT1) - dict1_outer_len, fxn)

    if not DICT_HEADER1 is None and not INSERT_HEADER is None:
        if header_axis == 0:
            # BECAUSE outer AND HEADER IS TIED TO OUTER (AXIS 0), MODIFTY HEADER
            DICT_HEADER1 = np.hstack((DICT_HEADER1[...,:index], INSERT_HEADER, DICT_HEADER1[..., index:])).astype('<U200')
        # elif header_axis == 1: pass
            # NO CHANGE TO DICT_HEADER1, JUST RETURN THE ORIGINAL GIVEN OR WHAT CAME OUT OF header_handle

    if not DICT_HEADER1 is None: return DICT1, DICT_HEADER1
    else: return DICT1


def append_outer(DICT1, LIST_OR_DICT_TO_INSERT, DICT_HEADER1=None, INSERT_HEADER=None, header_axis=None, fxn=None):
    '''Append an inner dictionary to a sparse dict in last position.'''

    # TEST CODE IS LOCATED IN sparse_dict_test_modules.insert_outer__function_test().

    fxn = inspect.stack()[0][3] if fxn is None else fxn
    # VALIDATION IS HANDLED IN insert_outer
    index = len(DICT1)

    if not (DICT_HEADER1 is None and INSERT_HEADER is None):
        DICT1, DICT_HEADER1 = insert_outer(DICT1, index, LIST_OR_DICT_TO_INSERT,
                                DICT_HEADER1=DICT_HEADER1, INSERT_HEADER=INSERT_HEADER, header_axis=header_axis, fxn=fxn)
        return DICT1, DICT_HEADER1
    else:
        DICT1 = insert_outer(DICT1, index, LIST_OR_DICT_TO_INSERT,
                                DICT_HEADER1=DICT_HEADER1, INSERT_HEADER=INSERT_HEADER, header_axis=header_axis, fxn=fxn)
        return DICT1


def core_insert_inner(INPUT_DICT, insert_index, INS_OBJ):
    ''' Insert an entry into all inner dictionaries at specified index without safeguards.'''

    # TEST CODE IS LOCATED IN sparse_dict_test_modules.insert_inner__function_test().

    ''' OUTLINE
    I) GET shape OF INS_OBJ, MAKE SURE IS STANDARDIZED TO [[]] AND {0: {}}
    II) VALIDATE / GET SHAPE OF DICT1
        A) DICT1 NOT GIVEN --- FINAL DICT DIMENSIONS MUST BE SUP_OBJ DIMENSIONS AS GIVEN. IF SHAPE OF INS_OBJ IS (X,Y), Y 
            DETERMINES # OUTER DICTS
            1) CONSTRUCT DICT1 AS EMPTY INNER DICTS, FILLING outer_len
            2) GET SHAPE OF DICT1  (MUST BE (outer_len,0)
        B) DICT1 IS GIVEN --- GET SHAPE OF DICT1
    III) VALIDATE / STANDARDIZE INS_OBJ
        A) DICT1 IS NOT GIVEN --- DONT WORRY ABOUT INS_OBJ SHAPE, JUST GET FROM [[INNER VALUES FOR inner_idx insert_index], 
            [INNER VALUES FOR inner_idx insert_index+1], ...] 
            INTO SD FORMAT WITH [[INNER VALUES FOR outer_idx 0], [INNER VALUES FOR  outer_idx 1], ...]
        B) DICT1 IS GIVEN --- MUST HAVE SHAPE OF DICT1
            1) IF (X,Y) IS SHAPE OF INS_OBJ (COULD BE NP OR SD), ASSUME Y WAS INTENDED TO ALIGN TO outer_len OF DICT1, MAKING
                X THE NUMBER OF VALUES TO BE INSERTED IN EACH INNER DICT
            2) IF DOES NOT ALIGN, CHECK IF X ALIGNS TO outer_len OF DICT
        C) MUST GET INS_OBJ AS SD & ORIENTED AS [[INNER VALUES outer_idx 0], [INNER VALUES outer_idx 1],...]
    IV) INCREMENT INS_OBJ INNER IDXS TO START AT insert_index
    V) COMBINE DICT1 AND INS_OBJ
        A) IF INSERTING AT END (APPENDING)
            1) REMOVE PLACEHOLDERS FROM INPUT_DICT
            2) MERGE DICT1 & INS_OBJ USING |
            3) APPLY PLACEHOLDERS
            4) TERMINATE MODULE BY RETURNING DICT1
        B) IF INSERTING INS_OBJ SOMEWHERE IN THE MIDDLE
            1) SPLIT DICT1[outer_idx] AT insert_index INTO 2 DICTS (LEFT_DICT & RIGHT_DICT)
            2) INCREMENT RIGHT_DICT INNER IDXS BY (number of inner idxs IN SUP_OBJ)
            3) MERGE ALL DICTS TOGETHER USING LEFT_DICT[outer_idx] | INS_OBJ[outer_idx] | RIGHT_DICT[outer_idx]
            4) TERMINATE MODULE BY RETURNING DICT1        
    '''

    # 10/11/22 IN SITUATION WHERE DOING y = sd.(core)_insert(append)_inner(x,....) IN ANOTHER MODULE, EVEN THO RETURNING
    # AS y, THE INPUT x IS BEING MODIFIED AS WELL :(.  PRINTING x IN OTHER MODULE AFTER RETURNING y SHOWED x HAD CHANGED.
    # PUTTING deepcopy HERE BREAKS THAT CHAIN (AND BLOWS UP MEMORY :( )
    DICT1 = deepcopy(INPUT_DICT)
    del INPUT_DICT

    # I) GET shape OF INS_OBJ, MAKE SURE IS STANDARDIZED TO [[]] AND {0: {}}
    # ASSUME INS_OBJ IS NP OR SD, AND USER GAVE IT AS 'ROW'  (3/25/23 THINKING "ROW" OR "COLUMN" DOESNT MATTER, AS LONG AS DICT & INS ARE LABELED THE SAME)

    # CHECK INSERT OBJ IS NOT EMPTY, VERIFY IS LIST-TYPE OR DICT
    empty = False
    if INS_OBJ is None: empty = True
    elif isinstance(INS_OBJ, (list, tuple, np.ndarray)):
        if np.array(INS_OBJ).size == 0: empty = True
    elif isinstance(INS_OBJ, dict):
        if INS_OBJ == {}: empty = True
    else: raise TypeError(f'\n{module_name()}.core_insert_inner() INVALID INSERTION OBJECT TYPE {type(INS_OBJ)}')
    if empty: raise ValueError(f'\n{module_name()}.core_insert_inner() INSERT OBJECT IS EMPTY')

    if is_sparse_inner(INS_OBJ): INS_OBJ = {0: INS_OBJ}
    elif isinstance(INS_OBJ, (np.ndarray, list, tuple)):
        INS_OBJ = np.array(INS_OBJ)
        if len(INS_OBJ.shape)==1: INS_OBJ=INS_OBJ.reshape((1,-1))
    elif is_sparse_outer(INS_OBJ): pass

    ins_shape = gs.get_shape('INS_OBJ', INS_OBJ, 'ROW')

    # II) VALIDATE / GET SHAPE OF DICT1
    # A) DICT1 NOT GIVEN --- FINAL DICT DIMENSIONS MUST BE SUP_OBJ DIMENSIONS AS GIVEN. IF SHAPE OF INS_OBJ IS (X,Y), Y DETERMINES # OUTER DICTS
    if DICT1 == {} or DICT1 is None:
        # 1) CONSTRUCT DICT1 AS EMPTY INNER DICTS, FILLING outer_len
        DICT1 = {int(outer_idx): {} for outer_idx in range(ins_shape[1])}
        # 2) GET SHAPE OF DICT1  (MUST BE (outer_len,0)
        dict_shape = gs.get_shape('DICT1', DICT1, 'ROW')
    # B) DICT1 IS GIVEN --- GET SHAPE OF DICT1
    else: dict_shape = gs.get_shape('DICT1', DICT1, 'ROW')

    # III) VALIDATE / STANDARDIZE INS_OBJ #################################################################################
    # A) DICT1 IS NOT GIVEN --- DONT WORRY ABOUT INS_OBJ SHAPE, JUST GET FROM [[INNER VALUES FOR inner_idx insert_index],
    #         [INNER VALUES FOR inner_idx insert_index+1], ...]
    #         INTO SD FORMAT WITH [[INNER VALUES FOR outer_idx 0], [INNER VALUES FOR  outer_idx 1], ...]
    if 0 in dict_shape:   # MUST HAVE BEEN SET TO THIS ABOVE IF EMPTY INPUT_DICT WAS PASSED, TRANSPOSE & HANDLE CONVERTING TO SD BELOW
        if isinstance(INS_OBJ, dict): INS_OBJ = core_sparse_transpose(INS_OBJ)
        elif isinstance(INS_OBJ, np.ndarray): INS_OBJ = INS_OBJ.transpose()

    # B) DICT1 IS GIVEN --- MUST HAVE SHAPE OF DICT1
    else:
        if insert_index not in range(dict_shape[1] + 1):
            raise ValueError(f'insert_idx ({insert_index}) IS OUTSIDE OF RANGE OF PASSED INPUT OBJECT')

        # 1) IF (X,Y) IS SHAPE OF INS_OBJ (COULD BE NP OR SD), ASSUME Y WAS INTENDED TO ALIGN TO outer_len OF DICT1, MAKING
        #             X THE NUMBER OF VALUES TO BE INSERTED IN EACH INNER DICT
        if dict_shape[0] == ins_shape[1]: # WAS GIVEN WITH EXPECTED ORIENTATION W-R-T GIVEN DICT, SO CHANGE TO [[]=outer]
            if isinstance(INS_OBJ, dict): INS_OBJ = core_sparse_transpose(INS_OBJ)
            elif isinstance(INS_OBJ, np.ndarray): INS_OBJ = INS_OBJ.transpose()
            else: raise AssertionError(f'OBJECT RE-ORIENTATION TO [[]=OUTER] IS FAILING')
        # 2) IF DOES NOT ALIGN, CHECK IF X ALIGNS TO outer_len OF DICT
        elif dict_shape[0] == ins_shape[0]: pass  # ALREADY IN DESIRED ORIENTATION, HANDLE CONVERSION TO SD BELOW
        else: raise ValueError(f'PASSED INSERT OBJECT DOES NOT ALIGN TO INPUT DICT OUTER LENGTH IN ANY ORIENTATION')

    # C) MUST GET INS_OBJ AS SD & ORIENTED AS [[INNER VALUES outer_idx 0], [INNER VALUES outer_idx 1],...]
        # ORIENTATION SHOULD HAVE BEEN HANDLED ABOVE, SO GET INTO SD
    if isinstance(INS_OBJ, np.ndarray):
        if 'INT' in str(INS_OBJ.dtype).upper(): INS_OBJ = zip_list_as_py_int(INS_OBJ)
        elif 'FLOAT' in str(INS_OBJ.dtype).upper(): INS_OBJ = zip_list_as_py_float(INS_OBJ)

    # RESET SHAPE HERE NOW THAT INS LAYOUT IS STANDARDIZED
    ins_shape = gs.get_shape('INS_OBJ', INS_OBJ, 'ROW')

    # END III) VALIDATE / STANDARDIZE INS_OBJ #################################################################################

    # IV) INCREMENT INS_OBJ INNER IDXS TO START AT insert_index
    for outer_key in INS_OBJ:
        INS_OBJ[int(outer_key)] = dict((
                zip(np.add(np.fromiter(INS_OBJ[outer_key].keys(), dtype=int), insert_index).tolist(), list(INS_OBJ[outer_key].values()))
        ))

    # V) COMBINE DICT1 AND INS_OBJ
    #     A) IF INSERTING AT END (APPENDING)
    if insert_index == dict_shape[1]:
        for outer_idx in DICT1:
    #         1) REMOVE PLACEHOLDERS FROM DICT1
            # 6/6/23 -- IF EMPTY IS BLOWING UP WHEN TRY TO INDEX IN --- GO TO try/except
            try:
                if DICT1[outer_idx][dict_shape[1]-1] == 0: del DICT1[outer_idx][dict_shape[1]-1]
            except: pass
    #         2) MERGE DICT1 & INS_OBJ USING |
            DICT1[int(outer_idx)] = DICT1[outer_idx] | INS_OBJ.pop(outer_idx)
    #         3) APPLY PLACEHOLDERS ---- REMEMBER INNER LEN IS NOW ONE LONGER, SO dict_shape[1] NOT dict_shape[1]-1 !!
            DICT1[int(outer_idx)][int(dict_shape[1])] = DICT1[int(outer_idx)].get(dict_shape[1], 0)
    #         4) TERMINATE MODULE BY RETURNING DICT1
        return DICT1
    else:   # B) IF INSERTING INS_OBJ SOMEWHERE IN THE MIDDLE
    #   1) SPLIT DICT1[outer_idx] AT insert_index INTO 2 DICTS (LEFT_DICT & RIGHT_DICT)

        LEFT_DICT, RIGHT_DICT = {}, {}
        for outer_idx in set(DICT1.keys()):
            ACTV_INNER = DICT1.pop(outer_idx)
            ACTV_KEYS = np.fromiter(ACTV_INNER.keys(), dtype=np.int32)
            ACTV_VALUES = np.fromiter(ACTV_INNER.values(), dtype=np.float64)
            del ACTV_INNER

            LEFT_DICT[int(outer_idx)] = dict((
                zip(ACTV_KEYS[ACTV_KEYS < insert_index].tolist(), ACTV_VALUES[ACTV_KEYS < insert_index].tolist())
            ))

            #   2) INCREMENT RIGHT_DICT INNER IDXS BY (number of inner idxs IN SUP_OBJ)
            RIGHT_DICT[int(outer_idx)] = dict((
                zip(np.add(ACTV_KEYS[ACTV_KEYS >= insert_index], ins_shape[1]).tolist(),
                ACTV_VALUES[ACTV_KEYS >= insert_index].tolist())
            ))

            del ACTV_KEYS, ACTV_VALUES

            #   3) MERGE ALL DICTS TOGETHER USING LEFT_DICT | INS_OBJ | RIGHT_DICT
            LEFT_DICT[int(outer_idx)] = LEFT_DICT.pop(outer_idx) | INS_OBJ.pop(outer_idx) | RIGHT_DICT.pop(outer_idx)

        #   4) TERMINATE MODULE BY RETURNING DICT1
        return LEFT_DICT


def insert_inner(DICT1, index, LIST_OR_DICT_TO_INSERT, DICT_HEADER1=None, INSERT_HEADER=None,
                 header_axis=None, fxn=None):
    '''Insert an entry into all inner dictionaries at specified index with safeguards and header handling.'''

    # TEST CODE IS LOCATED IN sparse_dict_test_modules.insert_inner__function_test().

    # ASSUMES DICT1 IS "CLEAN"
    fxn = inspect.stack()[0][3] if fxn is None else fxn
    DICT1 = dict_init(DICT1, fxn)
    insufficient_dict_args_1(DICT1, fxn)
    dict1_outer_len = outer_len(DICT1)
    dict1_inner_len = inner_len(DICT1)

    DICT1 = core_insert_inner(DICT1, index, LIST_OR_DICT_TO_INSERT)

    del LIST_OR_DICT_TO_INSERT

    # IF INS_HEADER WAS NOT PASSED, MEASURE THE DELTA OF DICT1 BEFORE & AFTER INSERT TO FIND
    # OUT HOW LONG INS_OBJ WAS. USE THIS DELTA TO CREATE THE DUMMY HEADER.
    DICT_HEADER1, INSERT_HEADER = insert_outer_inner_header_handle(DICT_HEADER1, INSERT_HEADER, dict1_outer_len,
                           dict1_inner_len, header_axis, 1, inner_len_quick(DICT1) - dict1_inner_len, fxn)

    if not DICT_HEADER1 is None and not INSERT_HEADER is None:
        if header_axis == 1:
            # BECAUSE inner AND HEADER IS TIED TO INNER (AXIS 1), MODIFTY HEADER
            DICT_HEADER1 = np.hstack((DICT_HEADER1[...,:index], INSERT_HEADER, DICT_HEADER1[..., index:])).astype('<U200')
        # elif header_axis == 0: pass
            # NO CHANGE TO DICT_HEADER1, JUST RETURN THE ORIGINAL GIVEN OR WHAT CAME OUT OF header_handle

    if not DICT_HEADER1 is None: return DICT1, DICT_HEADER1
    else: return DICT1


def append_inner(DICT1, LIST_OR_DICT_TO_APPEND, DICT_HEADER1=None, INSERT_HEADER=None, header_axis=None, fxn=None):
    ''' Append an entry into all inner dictionaries in the last position.'''

    # TEST CODE IS LOCATED IN sparse_dict_test_modules.insert_inner__function_test().

    fxn = inspect.stack()[0][3] if fxn is None else fxn
    # VALIDATION IS HANDLED IN insert_inner

    if DICT1 == {}: index = 0
    else: index = inner_len(DICT1)

    if not (DICT_HEADER1 is None and INSERT_HEADER is None):
        DICT1, DICT_HEADER1 = insert_inner(DICT1, index, LIST_OR_DICT_TO_APPEND, DICT_HEADER1=DICT_HEADER1,
                                           INSERT_HEADER=INSERT_HEADER, header_axis=header_axis, fxn=fxn)
        return DICT1, DICT_HEADER1
    else:
        DICT1 = insert_inner(DICT1, index, LIST_OR_DICT_TO_APPEND, DICT_HEADER1=DICT_HEADER1, INSERT_HEADER=INSERT_HEADER,
                                            header_axis=header_axis, fxn=fxn)
        return DICT1


def split_outer(DICT1, index):
    '''Split before user-specified outer index; returns 2 sparse dictionaries as tuple'''
    fxn = inspect.stack()[0][3]
    DICT1 = dict_init(DICT1, fxn)
    insufficient_dict_args_1(DICT1, fxn)
    is_sparse_dict(DICT1)

    _outer_len = outer_len(DICT1)

    if index >= _outer_len:
        DICT2 = {}
    elif index <= 0:
        DICT2 = deepcopy(DICT1)
        DICT1 = {}
    else:
        DICT2 = {int(_):DICT1[__] for _,__ in enumerate(range(index, _outer_len))}
        DICT1 = {int(_):DICT1[_] for _ in range(index)}

    return DICT1, DICT2


def split_inner(DICT1, index):
    '''Split before user-specified inner index; returns 2 sparse dictionaries as tuple'''
    fxn = inspect.stack()[0][3]
    DICT1 = dict_init(DICT1, fxn)
    insufficient_dict_args_1(DICT1, fxn)
    is_sparse_dict(DICT1)

    _outer_len = outer_len(DICT1)
    _inner_len = inner_len(DICT1)

    if index >= _inner_len:
        DICT2 = {}
    elif index <= 0:
        DICT2 = deepcopy(DICT1)
        DICT1 = {}
    else:
        DICT2 = {}
        for outer_key in range(_outer_len):
            DICT2[int(outer_key)] = {}
            for new_key, inner_key in enumerate(range(index, _inner_len)):
                if inner_key in list(DICT1[outer_key].keys()):
                    DICT2[int(outer_key)][int(new_key)] = DICT1[outer_key][inner_key]
                # ENFORCE PLACEHOLDER RULES
                if inner_key == _inner_len - 1 and inner_key not in list(DICT1[outer_key].keys()):
                    DICT2[int(outer_key)][int(inner_key)] = 0

                if inner_key in list(DICT1[outer_key].keys()): del DICT1[outer_key][inner_key]

    return DICT1, DICT2


def multi_select_outer(DICT1, INDICES_AS_LIST):
    '''Build sparse dict from user-specified outer indices of given sparse dict'''
    fxn = inspect.stack()[0][3]
    DICT1 = dict_init(DICT1, fxn)
    insufficient_dict_args_1(DICT1, fxn)
    is_sparse_dict(DICT1)

    NEW_DICT = {int(_):DICT1[__] for _,__ in enumerate(INDICES_AS_LIST)}

    return NEW_DICT


def core_multi_select_inner(DICT, INDICES_AS_LIST, as_inner=True, as_dict=True):
    '''Build sparse dict from user-specified inner indices of given sparse dict without safeguards.'''

    DICT1 = deepcopy(DICT)

    _outer_len = outer_len(DICT1)
    _inner_len = inner_len_quick(DICT1)

    if as_dict is True: NEW_OBJ = {}
    elif as_dict is False:
        if as_inner is True: NEW_OBJ = np.zeros((_outer_len, len(INDICES_AS_LIST)), dtype=np.float64)
        elif as_inner is False: NEW_OBJ = np.zeros((len(INDICES_AS_LIST), _outer_len), dtype=np.float64)

    if as_inner:
        for outer_key in range(_outer_len):
            if as_dict is True: NEW_OBJ[int(outer_key)] = {}
            for new_key, inner_key in enumerate(INDICES_AS_LIST):
                # ENFORCE PLACEHOLDER RULES
                if new_key == len(INDICES_AS_LIST) - 1:
                    NEW_OBJ[int(outer_key)][int(new_key)] = DICT1[outer_key].get(inner_key,0)
                elif inner_key in DICT1[outer_key]:
                    NEW_OBJ[int(outer_key)][int(new_key)] = DICT1[outer_key][inner_key]

    else:  # elif as_outer
        for new_key, old_inner_key in enumerate(INDICES_AS_LIST):
            if as_dict is True: NEW_OBJ[int(new_key)] = {}
            for old_outer_key in range(_outer_len):
                # ENFORCE PLACEHOLDER RULES
                if old_outer_key == _outer_len - 1:
                    NEW_OBJ[int(new_key)][int(old_outer_key)] = DICT1[old_outer_key].get(old_inner_key, 0)
                elif old_inner_key in DICT1[old_outer_key]:
                    NEW_OBJ[int(new_key)][int(old_outer_key)] = DICT1[old_outer_key][old_inner_key]

    return NEW_OBJ


def multi_select_inner(DICT1, INDICES_AS_LIST, as_inner=True, as_dict=True):
    '''Build sparse dict from user-specified inner indices of given sparse dict with safeguards.'''
    fxn = inspect.stack()[0][3]
    DICT1 = dict_init(DICT1, fxn)
    insufficient_dict_args_1(DICT1, fxn)
    is_sparse_dict(DICT1)

    NEW_DICT = core_multi_select_inner(DICT1, INDICES_AS_LIST, as_inner=as_inner, as_dict=as_dict)

    return NEW_DICT


# END CREATION, HANDLING & MAINTENANCE ##################################################################################################
#########################################################################################################################################
#########################################################################################################################################


#########################################################################################################################################
#########################################################################################################################################
# ABOUT #################################################################################################################################
def is_sparse_outer(DICT1):
    '''Returns True if object is an outer sparse dictionary.'''

    if not isinstance(DICT1, dict): return False
    if DICT1 == {}: return True
    first_key = np.fromiter(DICT1.keys(), dtype=np.int32)[0]
    if isinstance(DICT1[first_key], dict): return True
    else: return False


def is_sparse_inner(DICT1):
    '''Returns True if object is an inner sparse dictionary.'''
    if not isinstance(DICT1, dict): return False
    if DICT1 == {}: return False
    first_key = np.fromiter(DICT1.keys(), dtype=np.int32)[0]
    try: int(first_key) == first_key
    except: return False
    if True in map(lambda x: x in str(type(DICT1[first_key])).upper(), ('INT', 'FLOAT')): return True
    else: return False


def inner_len_quick(DICT1):
    '''Length of inner dictionaries that are held by the outer dictionary assuming clean sparse dict.'''
    return max(DICT1[0]) + 1


def inner_len(DICT1):
    '''Length of inner dictionaries that are held by the outer dictionary.'''
    # DONT BOTHER TO clean() OR resize() HERE, SINCE ONLY THE SCALAR LENGTH IS RETURNED (CHANGES TO DICTX ARENT RETAINED)
    fxn = inspect.stack()[0][3]
    DICT1 = dict_init(DICT1, fxn)
    insufficient_dict_args_1(DICT1, fxn)
    if False not in np.fromiter(
            (True if DICT1[outer_idx] == {} else False for outer_idx in DICT1.keys()), dtype=bool):
        return 0
    # try: _inner_len = np.max(np.fromiter((max(DICT1[_]) + 1 for _ in DICT1), dtype=int))
    try: _inner_len = max(map(max, DICT1.values())) + 1
    except: raise Exception(f'Sparse dictionary has a zero-len inner dictionary in {module_name()}.{fxn}()')

    return _inner_len


def outer_len(DICT1):
    '''Length of the outer dictionary that holds the inner dictionaries as values, aka number of inner dictionaries'''
    # DONT BOTHER TO clean() OR resize() HERE, SINCE ONLY THE SCALAR LENGTH IS RETURNED (CHANGES TO DICTX ARENT RETAINED)
    fxn = inspect.stack()[0][3]
    DICT1 = dict_init(DICT1, fxn)
    insufficient_dict_args_1(DICT1, fxn)
    if DICT1 == {}: return (0,)
    try: outer_len = len(DICT1)
    except: raise Exception(f'Sparse dictionary is a zero-len outer dictionary in {module_name()}.{fxn}()')

    return outer_len


def size_(DICT1):
    '''Return outer dict length * inner dict length.'''
    # DONT NEED arg CHECKS HERE, HANDLED BY outer_len() & inner_len()
    return outer_len(DICT1) * inner_len(DICT1)


def shape_(DICT1):
    """Returns shape of non-ragged sparse dict using numpy shape rules."""
    if DICT1 is None: return ()
    if DICT1 == {}: return (0,)
    if is_sparse_inner(DICT1): return (max(DICT1.keys())+1,)   # ASSUMES PLACEHOLDER RULE IS USED (len = max key + 1)
    # AFTER HERE ALL MUST BE DICT OF INNER DICTS
    # IF ALL INNERS ARE {}  (ZERO len) #############################
    _LENS = list(map(len, DICT1.values()))
    if (min(_LENS)==0) and (max(_LENS)==0): return len(DICT1), 0
    del _LENS
    # END IF ALL INNERS ARE {}  (ZERO len) ##########################
    # TEST FOR RAGGEDNESS ###########################################
    _INNER_LENS = list(map(max, DICT1.values()))
    if not min(_INNER_LENS)==max(_INNER_LENS): raise Exception(f'sparse_dict.shape_() >>> DICT1 IS RAGGED.')
    # END TEST FOR RAGGEDNESS ###########################################
    return (len(DICT1), _INNER_LENS[0]+1)


def sum_over_outer_key(DICT1, outer_key):
    '''Sum all the values in an inner dict, as given by outer dict key.'''

    fxn = inspect.stack()[0][3]
    insufficient_dict_args_1(DICT1, fxn)
    DICT1 = dict_init(DICT1, fxn)
    return sum(list(DICT1[outer_key].values()))


def sum_over_inner_key(DICT1, inner_key):
    '''Sum over all inner dicts the values that are keyed with the user-entered inner key.'''

    fxn = inspect.stack()[0][3]
    DICT1 = dict_init(DICT1, fxn)
    insufficient_dict_args_1(DICT1, fxn)
    SUM = 0
    for outer_key in DICT1:
        if inner_key not in range(inner_len(DICT1)):
            raise Exception(f'{module_name()}.{fxn}() Key {inner_key} out of bounds for inner dict with len {inner_len(DICT1)}.')
        if inner_key in DICT1[outer_key]:
            SUM += DICT1[outer_key][inner_key]
    return SUM


def sparsity(DICT1):
    '''Calculate sparsity of a sparse dict.'''
    fxn = inspect.stack()[0][3]
    DICT1 = dict_init(DICT1, fxn)
    insufficient_dict_args_1(DICT1, fxn)
    SIZE = size_(DICT1)
    # dtypes MUST BE FLOAT, int CANT HANDLE THE BIG #s RuntimeWarning: overflow encountered in long_scalars
    total_hits = np.sum(np.fromiter((len(DICT1[outer_key]) - int(0 in DICT1[outer_key].values()) for outer_key in DICT1), dtype=float))

    return 100 - 100 * total_hits / SIZE


def list_sparsity(LIST1):
    '''Calculate sparsity of a list-type of list-types.'''
    LIST1 = list_init(LIST1=LIST1, fxn=inspect.stack()[0][3])[0]

    return lsp.list_sparsity(LIST1)


def core_sparse_equiv(DICT1, DICT2):
    '''Check for equivalence of two sparse dictionaries without safeguards for speed.'''

    # 1) TEST OUTER SIZE
    if len(DICT1) != len(DICT2): return False

    # 2) TEST INNER SIZES
    for outer_key in DICT1:  # ALREADY ESTABLISHED OUTER KEYS ARE EQUAL
        if len(DICT1[outer_key]) != len(DICT2[outer_key]): return False

    # for outer_key in DICT1:
    #     if not np.array_equiv(unzip_to_ndarray_float64({0: DICT1[outer_key]}),
    #                          unzip_to_ndarray_float64({0: DICT2[outer_key]})): return False

    # 3) TEST INNER KEYS ARE EQUAL
    for outer_key in DICT1:   # ALREADY ESTABLISHED OUTER KEYS ARE EQUAL
        if not np.allclose(np.fromiter(DICT1[outer_key], dtype=np.int32),
                             np.fromiter(DICT2[outer_key], dtype=np.int32), rtol=1e-8, atol=1e-8): return False

    # 4) TEST INNER VALUES ARE EQUAL
    for outer_key in DICT1:  # ALREADY ESTABLISHED OUTER KEYS ARE EQUAL
        if not np.allclose(np.fromiter(DICT1[outer_key].values(), dtype=np.float64),
                          np.fromiter(DICT2[outer_key].values(), dtype=np.float64), rtol=1e-8, atol=1e-8): return False

    # IF GET THIS FAR, MUST BE True
    return True


def safe_sparse_equiv(DICT1, DICT2):
    '''Safely check for equivalence of two sparse dictionaries with safeguards.'''

    fxn = inspect.stack()[0][3]
    DICT1 = dict_init(DICT1, fxn)
    DICT2 = dict_init(DICT2, fxn)
    insufficient_dict_args_2(DICT1, DICT2, fxn)

    # 1) TEST OUTER KEYS ARE EQUAL
    if not np.allclose(np.fromiter(DICT1, dtype=np.int32),
                      np.fromiter(DICT2, dtype=np.int32), rtol=1e-8, atol=1e-8): return False

    # 2) RUN core_sparse_equiv
    if core_sparse_equiv(DICT1, DICT2) is False: return False

    return True  # IF GET TO THIS POINT, MUST BE TRUE


def return_uniques(DICT1):
    '''Return unique values of a sparse dictionary as list.'''
    fxn = inspect.stack()[0][3]
    DICT1 = dict_init(DICT1, fxn)
    insufficient_dict_args_1(DICT1, fxn)
    NUM_HOLDER, STR_HOLDER = [], []

    for outer_key in DICT1:   # 10/16/22 DONT CHANGE THIS, HAS TO DEAL W DIFF DTYPES, DONT USE np.unique, BLOWS UP FOR '<' not supported between instances of 'str' and 'int'
        for value in DICT1[outer_key].values():
            if True in map(lambda x: x in str(type(value)).upper(), ['INT', 'FLOAT']):
                if value not in NUM_HOLDER: NUM_HOLDER.append(value)
            else:
                if value not in STR_HOLDER: STR_HOLDER.append(str(value))

    if sparsity(DICT1) < 100 and 0 not in NUM_HOLDER: NUM_HOLDER.append(0)

    UNIQUES = np.array(sorted(NUM_HOLDER) + sorted(STR_HOLDER), dtype=object)

    del NUM_HOLDER, STR_HOLDER

    return UNIQUES


def display(DICT1, number_of_inner_dicts_to_print=float('inf')):
    '''Print sparse dict to screen.'''

    fxn = inspect.stack()[0][3]
    DICT1 = dict_init(DICT1, fxn)
    insufficient_dict_args_1(DICT1, fxn)

    _len = len(DICT1)

    while True:
        if _len > 50 and number_of_inner_dicts_to_print == float('inf'):
            __ = vui.validate_user_str(f'\nDICT has {_len} entries. Print all(p), pick number to print(n), abort(a) > ', 'NPA')
            if __ == 'A': break
            elif __ == 'N': num_rows = vui.validate_user_int(f'Enter number of inner dicts to print (of {_len}) > ', min=1, max=_len)
            elif __ == 'P': num_rows = _len
        else:
            num_rows = min(_len, number_of_inner_dicts_to_print)

        print()
        # IF DICT HASNT BEEN CLEANED, outer_keys MAY NOT BE IN SEQUENCE, SO ONLY PRINT VALID outer_keys
        VALID_OUTER_KEYS = set(DICT1.keys())
        print_count = 0
        outer_key = 0
        while print_count < num_rows:
            _ = outer_key
            if _ in VALID_OUTER_KEYS:
                print(f'{str(_)}:'.ljust(4) +f'{str(DICT1[_])[:100]}' + (f' ...' if len(str(DICT1[_])) > 70 else ''))
                print_count += 1
            outer_key += 1
        print()

        break


def core_find_constants(DICT1, orientation):
    '''Finds a column of constants. Returns dict/empty dict of non-zero constant indices, list/empty list of zero idxs.'''
    # RETURNS COLUMNS OF ZEROS FOR SUBSEQUENT HANDLING. len(COLUMNS OF CONSTANTS) SHOULD BE 1, BUT RETURN FOR HANDLING IF OTHERWISE.

    COLUMNS_OF_CONSTANTS, COLUMNS_OF_ZEROS = {}, []
    if orientation == 'COLUMN':
        for outer_idx in DICT1:
            _min, _max = min_max_({0: DICT1[outer_idx]})
            if _min != _max: continue
            else:  # _min == _max
                if _min == 0: COLUMNS_OF_ZEROS.append(outer_idx)
                else: COLUMNS_OF_CONSTANTS = COLUMNS_OF_CONSTANTS | {outer_idx: _min}

    elif orientation == 'ROW':
        for inner_idx in range(inner_len_quick(DICT1)):
            COL_HOLDER = np.fromiter((map(lambda x: DICT1[x].get(inner_idx, 0), DICT1)), dtype=np.float64)
            _min, _max = int(np.min(COL_HOLDER)), int(np.max(COL_HOLDER))

            if _min != _max: continue
            elif _min == _max and _min == 0: COLUMNS_OF_ZEROS.append(inner_idx)
            elif _min == _max: COLUMNS_OF_CONSTANTS = COLUMNS_OF_CONSTANTS | {inner_idx: _min}
        del COL_HOLDER

    return COLUMNS_OF_CONSTANTS, COLUMNS_OF_ZEROS


def find_constants(DICT1, orientation):
    '''Finds a column of constants with safeguards. Returns dict/empty dict of non-zero constant indices, list/empty list of zero idxs.'''
    fxn = inspect.stack()[0][3]
    DICT1 = dict_init(DICT1, fxn)
    insufficient_dict_args_1(DICT1, fxn)
    sparse_dict_check(DICT1, fxn)

    orientation = akv.arg_kwarg_validater(orientation.upper(), 'outer_or_inner', ['ROW', 'COLUMN'],
                                             'sparse_dict', inspect.stack()[0][3])

    return core_find_contants(DICT1, orientation)


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
    sparse_dict_check(DICT1, fxn)
    sparse_dict_check(DICT2, fxn)
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

# END ABOUT #############################################################################################################################
#########################################################################################################################################
#########################################################################################################################################

#########################################################################################################################################
#########################################################################################################################################
# LINEAR ALGEBRA ########################################################################################################################

def sparse_identity(_outer_len, _inner_len):
    '''Identity matrix as sparse dictionary.'''
    SPARSE_IDENTITY = {}

    for outer_idx in range(_outer_len):
        if outer_idx != _outer_len-1:
            SPARSE_IDENTITY[int(outer_idx)] = {int(outer_idx):int(1), int(_inner_len-1):int(0)}
        elif outer_idx == _outer_len-1:
            SPARSE_IDENTITY[int(outer_idx)] = {int(outer_idx):int(1)}

    return SPARSE_IDENTITY


def defunct_sparse_transpose(DICT1):
    # IS 1% FASTER THAN "BUILD TRANS DICT FROM OLD DICT ON THE FLY", BUT MEMORY IS 5% MORE
    '''Transpose a sparse dict to a sparse dict using shady tricks to avoid creating a new dict object to save memory.'''
    # WORK ON DICT1 IN-PLACE SO TO NOT CREATE ANOTHER COPY OF A POTENTIALLY HUGE OBJECT

    '''
    9/26/22  10 TRIALS EACH
    
    rows=10000, cols=10000, sparsity=80
    Average time old=114.9 sec, sdev=6.0;       Average mem 1 = 706.6 MB, sdev=85.0
    Average time new=113.7 sec, sdev=5.4;       Average mem 2 = 720.6 MB, sdev=56.7
    
    rows=1000, cols=100000, sparsity=80
    Average time old=109.8 sec, sdev=4.2;      Average mem 1 = 774.3 MB, sdev=255.1
    Average time new=108.9 sec, sdev=4.2;      Average mem 2 = 847.5 MB, sdev=291.3
    
    rows=100000, cols=1000, sparsity=80
    Average time old=109.6 sec, sdev=7.4;      Average mem 1 = 568.0 MB, sdev=71.4
    Average time new=107.7 sec, sdev=5.4;      Average mem 2 = 577.9 MB, sdev=70.9
    
    ON AVERAGE, NEW "IN-PLACE" METHOD IS 1.2% FASTER THAN OLD WAY, BUT MEMORY IS 4.7% BIGGER, DEFEATING THE ORIGINAL PURPOSE :(
    '''

    fxn = inspect.stack()[0][3]
    DICT1 = dict_init(DICT1, fxn)
    insufficient_dict_args_1(DICT1, fxn)

    # GET THIS B4 CLEANING OUT PLACEHOLDERS
    _outer_len, _inner_len = outer_len(DICT1), inner_len(DICT1)

    # CLEAN OUT PLACEHOLDERS
    DICT1 = drop_placeholders(DICT1)

    # GET THE SQUARE PART
    bound_idx = min(_outer_len, _inner_len)
    for outer_idx in range(0, bound_idx):   # DONT HAVE TO GET 0,0 FOR TRANSPOSE, BUT GET FOR PLACEHOLDER
        for inner_idx in range(outer_idx):   # DONT HAVE TO GET THE DIAGONAL HERE, STAYS THE SAME
            if inner_idx not in DICT1[outer_idx] and outer_idx not in DICT1[inner_idx]: continue
            else:  # WE KNOW THAT EITHER outer_idx IS IN DICT1[inner] OR inner_idx IS IN DICT1[outer], OR BOTH
                if inner_idx in DICT1[outer_idx] and outer_idx in DICT1[inner_idx]:
                    DICT1[int(outer_idx)][int(inner_idx)], DICT1[int(inner_idx)][int(outer_idx)] = \
                        DICT1[inner_idx].pop(outer_idx), DICT1[outer_idx].pop(inner_idx)
                elif outer_idx in DICT1[inner_idx] and inner_idx not in DICT1[outer_idx]:
                    DICT1[int(outer_idx)][int(inner_idx)] = DICT1[inner_idx].pop(outer_idx)
                elif inner_idx in DICT1[outer_idx] and outer_idx not in DICT1[inner_idx]:
                    DICT1[int(inner_idx)][int(outer_idx)] = DICT1[outer_idx].pop(inner_idx)

    # IF _outer_len > _inner_len, LAST inner_idx WILL BE COVERED DURING TRANSPOSE, SO NO PLACEHOLDERS
    # CANT DO THIS UNDER THE ABOVE LOOP, LAST IDXS ARE ALWAYS CHANGING
    if _inner_len >= _outer_len:
        for outer_idx in DICT1:
            if _outer_len-1 not in DICT1[outer_idx]:
                DICT1[int(outer_idx)][int(_outer_len-1)] = 0

    # GET RESIDUAL AREA
    if _outer_len > _inner_len:
        for old_outer_key in range(bound_idx, _outer_len):  # DO LAST KEY SEPARATE
            if old_outer_key != _outer_len-1:
                for old_inner_key in set(DICT1[old_outer_key]).intersection(range(_inner_len)):
                    DICT1[int(old_inner_key)][int(old_outer_key)] = DICT1[old_outer_key].pop(old_inner_key)
            if old_outer_key == _outer_len - 1:
                for old_inner_key in range(_inner_len):
                    DICT1[int(old_inner_key)][int(old_outer_key)] = DICT1[old_outer_key].get(old_inner_key, 0)
            del DICT1[old_outer_key]

    if _inner_len > _outer_len:
        for old_inner_key in range(bound_idx, _inner_len):
            DICT1[int(old_inner_key)] = {}

        for old_outer_key in range(_outer_len):
            if old_outer_key != _outer_len - 1:
                for old_inner_key in set(DICT1[old_outer_key]).intersection(range(bound_idx, _inner_len)):
                    DICT1[int(old_inner_key)][int(old_outer_key)] = DICT1[old_outer_key].pop(old_inner_key)
            elif old_outer_key == _outer_len-1:
                for old_inner_key in range(bound_idx, _inner_len):
                    # SO ASSHOLE get DOESNT RETURN None
                    if old_inner_key in DICT1[old_outer_key]:
                        DICT1[int(old_inner_key)][int(old_outer_key)] = DICT1[old_outer_key][old_inner_key]
                        del DICT1[old_outer_key][old_inner_key]
                    else: DICT1[int(old_inner_key)][int(old_outer_key)] = 0

    # REORDER IDXS
    for outer_key in sorted([_ for _ in DICT1]):
        DICT1[outer_key] = DICT1.pop(outer_key)
        for inner_key in sorted([__ for __ in DICT1[outer_key]]):
            DICT1[int(outer_key)][int(inner_key)] = DICT1[outer_key].pop(inner_key)

    return DICT1


def new_sparse_transpose(DICT1):
    '''Transpose a sparse dict to a sparse dict using shady tricks with numpy.'''

    old_outer_len = outer_len(DICT1)
    old_inner_len = inner_len_quick(DICT1)

    def placeholder(x): NEW_DICT[int(x)][int(old_outer_len - 1)] = NEW_DICT[x].get(old_outer_len - 1, 0)

    # if old_inner_len >= old_outer_len:

    NEW_DICT = dict((map(lambda x: (int(x),{}), range(old_inner_len))))

    def appender(x, outer_key): NEW_DICT[int(x)][int(outer_key)] = DICT1[outer_key][x]

    list(map(lambda outer_key: list(map(lambda x: appender(x, outer_key), DICT1[outer_key])), DICT1))

    # del appender

    list(map(lambda x: NEW_DICT[old_inner_len-1].pop(x) if NEW_DICT[old_inner_len-1][x]==0 else 1, list(NEW_DICT[old_inner_len-1].keys())))
    list(map(lambda x: placeholder(x), NEW_DICT))

    # if old_outer_len > old_inner_len:
    #
    #     NEW_DICT = {}
    #
    #     # def appender(x): DICT1[x].get(old_inner_idx, 0) if DICT1[x].get(old_inner_idx, 0) != 0 else 1,
    #
    #     for old_inner_idx in range(old_inner_len):
    #         HOLDER = dict((
    #             zip(range(old_outer_len), list(map(lambda x: DICT1[x].get(old_inner_idx, 0), DICT1)))
    #         ))
    #         list(map(lambda x: HOLDER.pop(x) if HOLDER[x]==0 else 1, list(HOLDER.keys())))
    #         NEW_DICT[int(old_inner_idx)] = HOLDER
    #
    #     list(map(lambda x: placeholder(x), NEW_DICT))

    del placeholder

    return NEW_DICT



def core_sparse_transpose(DICT):
    '''Transpose a sparse dict to a sparse dict by brute force without safeguards.'''

    '''Inner len becomes outer len, and outer len becomes inner len.'''
    last_inner_key = max(DICT)
    DICT1 = {int(_):{} for _ in range(inner_len_quick(DICT))}
    for outer_key in DICT:
        for inner_key in list(DICT[outer_key].keys()):
            DICT1[int(inner_key)][int(outer_key)] = DICT[outer_key][inner_key]

    del DICT

    '''If at end of each original inner dict (should be a value there due to placeholding from zip() and clean()),
        skip putting this in TRANSPOSE (otherwise last inner dict in TRANSPOSE would get full of zeros).'''
    last_outer_key = max(DICT1)
    for inner_key in DICT1[last_outer_key].copy():
        if DICT1[last_outer_key][inner_key] == 0: del DICT1[last_outer_key][inner_key]

    '''If any TRANSPOSE inner dicts do not have a key for outer_len(dict)-1, create one with a zero to maintain
        placeholding rule.'''
    for outer_key in DICT1:
        DICT1[int(outer_key)][int(last_inner_key)] = DICT1[outer_key].get(last_inner_key, 0)

    return DICT1


def sparse_transpose(DICT1):
    '''Transpose a sparse dict to a sparse dict by brute force with safeguards.'''

    # VERIFIED 9-17-22 THAT THIS IS ALMOST 1.3X FASTER THAN unzip_to_ndarray > np.transpose() > zip_list

    fxn = inspect.stack()[0][3]
    DICT1 = dict_init(DICT1, fxn)
    insufficient_dict_args_1(DICT1, fxn)

    return core_sparse_transpose(DICT1)


def sparse_transpose2(DICT1):
    '''Transpose a sparse dict to a sparse dict with Pandas DataFrames.'''

    fxn = inspect.stack()[0][3]
    DICT1 = dict_init(DICT1, fxn)
    insufficient_dict_args_1(DICT1, fxn)

    start_outer_len = outer_len(DICT1)
    start_inner_len = inner_len(DICT1)

    TRANSPOSE_DICT = pd.DataFrame(DICT1).fillna(0).sort_index().T
    TRANSPOSE_DICT = TRANSPOSE_DICT.to_dict()

    # MUST DO FULL CLEAN BECAUSE OF fillna(0) FROM PANDAS
    TRANSPOSE_DICT = clean(TRANSPOSE_DICT)

    end_outer_len = outer_len(TRANSPOSE_DICT)
    end_inner_len = inner_len(TRANSPOSE_DICT)

    # THERE IS A NUANCE THAT EXPOSES ITSELF WHERE A COLUMN THAT WAS ALL ZEROS IN THE SPARSE DICTIONARY WONT SHOW UP AS
    # A ROW IN THE PANDAS TRANSPOSED SPARSE DICT, THUS CHANGING THE MATRIX.  CANT GET AROUND THIS OTHER THAN TERMINATE.
    if end_outer_len != start_inner_len or end_inner_len != start_outer_len:
        raise Exception(f'{module_name()}.{fxn}() Sparse dictionary has a column that was all zeros and fell out during transposition by Pandas.')

    return TRANSPOSE_DICT


def sparse_transpose3(DICT1):
    '''Transpose a sparse dict to a sparse dict by converting to array, transpose, and convert back to dict.'''

    fxn = inspect.stack()[0][3]
    DICT1 = dict_init(DICT1, fxn)
    insufficient_dict_args_1(DICT1, fxn)

    _ = unzip_to_ndarray(DICT1)[0].transpose()
    TRANSPOSE_DICT = zip_list(_)

    _outer_len = outer_len(TRANSPOSE_DICT)
    _inner_len = inner_len(TRANSPOSE_DICT)

    for outer_idx in range(_outer_len):
        '''If any TRANSPOSE inner dicts do not have a key for len(dict)-1, create one with a zero to maintain
            placeholding rule.'''
        if _inner_len - 1 not in TRANSPOSE_DICT[outer_idx]:
            TRANSPOSE_DICT[int(outer_idx)][int(_inner_len-1)] = 0

    '''Clean out any zeros put there during transposing from last outer_idx.'''
    for inner_idx in deepcopy(TRANSPOSE_DICT[_outer_len-1]):
        if inner_idx == _inner_len-1: continue       # DONT DELETE PLACEHOLDER
        if TRANSPOSE_DICT[_outer_len-1][inner_idx] == 0:
            del TRANSPOSE_DICT[int(_outer_len-1)][int(inner_idx)]

    return TRANSPOSE_DICT


def sparse_transpose_from_list(LIST1):
    '''Transpose a list to a sparse dict.'''

    fxn = inspect.stack()[0][3]
    LIST1 = list_init(LIST1, fxn)[0]
    insufficient_list_args_1(LIST1, fxn)

    LIST1 = np.array(LIST1).transpose()

    return zip_list(LIST1)


def test_new_core_matmul(DICT1, DICT2, DICT2_TRANSPOSE=None, return_as=None):
    '''DICT1 and DICT2 enter oriented with sparse inner dicts as rows, with multiplication rules like NumPy.  There is
        no protection here to prevent dissimilar sized rows from DICT1 dotting with columns from DICT2.
        Create posn for last entry, so that placeholder rules are enforced (original length of object is retained).'''

    # Transpose DICT2 for ease of multiplication, not for matmul rules
    if not DICT2_TRANSPOSE is None: DICT2_T = DICT2_TRANSPOSE
    else: DICT2_T = sparse_transpose(DICT2)

    if not return_as is None: return_as = return_as.upper()

    if return_as is None or return_as == 'SPARSE_DICT':
        OUTPUT = {int(_):{} for _ in range(outer_len(DICT1))}
    elif return_as == 'ARRAY':
        OUTPUT = np.zeros((outer_len(DICT1), outer_len(DICT2_T)), dtype=float)

    for outer_dict_idx1 in DICT1:
        for outer_dict_idx2 in DICT2_T:
            dot = 0
            # for inner_dict_idx in DICT1[outer_dict_idx1]:
            #     if inner_dict_idx in DICT2_T[outer_dict_idx2]:
            #         dot += DICT1[outer_dict_idx1][inner_dict_idx] * DICT2_T[outer_dict_idx2][inner_dict_idx]
            # for idx in list(set(DICT1[outer_dict_idx1]).intersection(DICT2_T[outer_dict_idx2])):  # SPEED TEST 9/20/22 SUGGEST PULLS AWAY FROM OLD WAY AS SIZES GET BIGGER
            #     dot += DICT1[outer_dict_idx1][idx] * DICT2_T[outer_dict_idx2][idx]

            MASK = set(DICT1[outer_dict_idx1]).intersection(DICT2_T[outer_dict_idx2])

            dot = np.matmul(np.fromiter((DICT1[outer_dict_idx1][_] for _ in MASK), dtype=np.float64),
                                   np.fromiter((np.array(DICT2_T[outer_dict_idx2][_]) for _ in MASK), dtype=np.float64))

            if dot != 0: OUTPUT[int(outer_dict_idx1)][int(outer_dict_idx2)] = dot



    if return_as is None or return_as == 'SPARSE_DICT':
        last_inner_idx = outer_len(DICT2_T) - 1
        for outer_dict_idx1 in OUTPUT:
            if last_inner_idx not in OUTPUT[outer_dict_idx1]:
                OUTPUT[int(outer_dict_idx1)][int(last_inner_idx)] = 0

    return OUTPUT


def core_matmul(DICT1, DICT2, DICT2_TRANSPOSE=None, return_as=None):
    '''DICT1 and DICT2 enter oriented with sparse inner dicts as rows, with multiplication rules like NumPy.  There is
        no protection here to prevent dissimilar sized rows from DICT1 dotting with columns from DICT2.
        Create posn for last entry, so that placeholder rules are enforced (original length of object is retained).'''

    # Transpose DICT2 for ease of multiplication, not for matmul rules
    if not DICT2_TRANSPOSE is None: DICT2_T = DICT2_TRANSPOSE
    else: DICT2_T = sparse_transpose(DICT2)

    if not return_as is None: return_as = return_as.upper()

    if return_as is None or return_as == 'SPARSE_DICT':
        OUTPUT = {int(_):{} for _ in range(outer_len(DICT1))}
    elif return_as == 'ARRAY':
        OUTPUT = np.zeros((outer_len(DICT1), outer_len(DICT2_T)), dtype=float)

    for outer_dict_idx1 in DICT1:
        for outer_dict_idx2 in DICT2_T:
            dot = 0
            # for inner_dict_idx in DICT1[outer_dict_idx1]:
            #     if inner_dict_idx in DICT2_T[outer_dict_idx2]:
            #         dot += DICT1[outer_dict_idx1][inner_dict_idx] * DICT2_T[outer_dict_idx2][inner_dict_idx]
            for idx in set(DICT1[outer_dict_idx1]).intersection(DICT2_T[outer_dict_idx2]):  # SPEED TEST 9/20/22 SUGGEST PULLS AWAY FROM OLD WAY AS SIZES GET BIGGER
                dot += DICT1[outer_dict_idx1][idx] * DICT2_T[outer_dict_idx2][idx]
            if dot != 0: OUTPUT[int(outer_dict_idx1)][int(outer_dict_idx2)] = dot

    if return_as is None or return_as == 'SPARSE_DICT':
        last_inner_idx = outer_len(DICT2_T) - 1
        for outer_dict_idx1 in OUTPUT:
            if last_inner_idx not in OUTPUT[outer_dict_idx1]:
                OUTPUT[int(outer_dict_idx1)][int(last_inner_idx)] = 0

    return OUTPUT


def core_symmetric_matmul(DICT1, DICT2, DICT2_TRANSPOSE=None, return_as=None):
    '''DICT1 and DICT2 enter oriented with sparse inner dicts as rows, with multiplication rules like NumPy.
    For use on things like ATA and AAT to save time.  There is no protection here to prevent dissimilar sized
    rows from DICT1 dotting with columns from DICT2.  Create posn for last entry, so that placeholder rules are
    enforced (original length of object is retained).'''

    BEAR = lambda t0: round((time.time() - t0), 2)

    #********** Transpose DICT2 for ease of multiplication, not for matmul rules.***************************
    print(f'                BEAR TRANSPOSING DICT2 IN core_symmetric_matmul'); t0 = time.time()
    if not DICT2_TRANSPOSE is None: DICT2_T = DICT2_TRANSPOSE
    else: DICT2_T = core_sparse_transpose(DICT2)
    print(f'                END BEAR TRANSPOSING DICT2 IN core_symmetric_matmul. time = {BEAR(t0)}\n')

    _outer_len = outer_len(DICT1)  # == outer_len(DICT2_T) == inner_len(DICT2)
    _inner_len = inner_len(DICT1)

    if not return_as is None: return_as = return_as.upper()

    if return_as is None or return_as == 'SPARSE_DICT':
        OUTPUT = {int(outer_key):{} for outer_key in range(_outer_len)}  # MUST CREATE BEFORE RUNNING, TO HAVE ALL SLOTS AVAILABLE FOR FILLING
    elif return_as == 'ARRAY':
        OUTPUT = np.zeros((_outer_len, _outer_len),dtype=float)   # SYMMETRIC, REMEMBER

    print(f'                BEAR GOING INTO BUILD MATRIX IN core_symmetric_matmul'); t0 = time.time()
    for outer_dict_idx2 in range(_outer_len):
        for outer_dict_idx1 in range(outer_dict_idx2 + 1):  # MUST GET DIAGONAL, SO MUST GET WHERE outer1 = outer2
            # ********* THIS IS THE STEP THAT IS TAKING FOREVER AND JACKING RAM!! ***********
            dot = 0
            # for inner_dict_idx1 in DICT1[outer_dict_idx1]:  # OLD WAY---DOT BY KEY SEARCH
            #     if inner_dict_idx1 in DICT2_T[outer_dict_idx2]:
            #         dot += DICT1[outer_dict_idx1][inner_dict_idx1] * DICT2_T[outer_dict_idx2][inner_dict_idx1]
            for idx in set(DICT1[outer_dict_idx1]).intersection(DICT2_T[outer_dict_idx2]):  # SPEED TEST 9/20/22 SUGGEST PULLS AWAY FROM OLD WAY AS SIZES GET BIGGER
                dot += DICT1[outer_dict_idx1][idx] * DICT2_T[outer_dict_idx2][idx]
            if dot != 0:
                OUTPUT[int(outer_dict_idx2)][int(outer_dict_idx1)] = dot
                OUTPUT[int(outer_dict_idx1)][int(outer_dict_idx2)] = dot

    # CLEAN UP PLACEHOLDER RULES
    if return_as is None or return_as == 'SPARSE_DICT':
        for outer_key in OUTPUT:
            if _outer_len - 1 not in OUTPUT[outer_key]:  # OUTER LEN = INNER LEN
                OUTPUT[int(outer_key)][int(_outer_len - 1)] = 0

    print(f'                BEAR DONE BUILD MATRIX IN core_symmetric_matmul. time = {BEAR(t0)}\n')

    return OUTPUT


def matmul(DICT1, DICT2, DICT2_TRANSPOSE=None, return_as=None):
    '''Run matmul with safeguards that assure matrix multiplication rules are followed when running core_matmul().'''

    fxn = inspect.stack()[0][3]
    DICT1 = dict_init(DICT1, fxn)
    DICT2 = dict_init(DICT2, fxn)
    insufficient_dict_args_2(DICT1, DICT2, fxn)
    broadcast_check(DICT1, DICT2, fxn)   # DO THIS BEFORE TRANSPOSING DICT2

    if not DICT2_TRANSPOSE is None:
        DICT2_TRANSPOSE = dict_init(DICT2_TRANSPOSE, fxn)
        insufficient_dict_args_1(DICT2_TRANSPOSE, fxn)
        DICT2_T = DICT2_TRANSPOSE
    else: DICT2_T = sparse_transpose(DICT2)

    return core_matmul(DICT1, DICT2, DICT2_TRANSPOSE=DICT2_T, return_as=return_as)


def symmetric_matmul(DICT1, DICT2, DICT1_TRANSPOSE=None, DICT2_TRANSPOSE=None, return_as=None):
    '''Run matmul with safeguards that assure matrix multiplication rules are followed when running core_symmetric_matmul().'''

    fxn = inspect.stack()[0][3]
    DICT1 = dict_init(DICT1, fxn)
    DICT2 = dict_init(DICT2, fxn)
    insufficient_dict_args_2(DICT1, DICT2, fxn)
    broadcast_check(DICT1, DICT2, fxn)   # DO THIS BEFORE TRANSPOSING DICT2
    if not DICT1_TRANSPOSE is None:
        DICT1_TRANSPOSE = dict_init(DICT1_TRANSPOSE, fxn)
        insufficient_dict_args_1(DICT1_TRANSPOSE, fxn)
        DICT1_T = DICT1_TRANSPOSE
    else: DICT1_T = sparse_transpose(DICT1)
    if not DICT2_TRANSPOSE is None:
        DICT2_TRANSPOSE = dict_init(DICT2_TRANSPOSE, fxn)
        insufficient_dict_args_1(DICT2_TRANSPOSE, fxn)
        DICT2_T = DICT2_TRANSPOSE
    else: DICT2_T = sparse_transpose(DICT2)

    symmetric_matmul_check(DICT1, DICT2, DICT1_TRANSPOSE=DICT1_T, DICT2_TRANSPOSE=DICT2_T)

    return core_symmetric_matmul(DICT1, DICT2, DICT2_TRANSPOSE=DICT2_T, return_as=return_as)


def sparse_ATA(DICT1, DICT1_TRANSPOSE=None, return_as=None):
    '''Calculates ATA on DICT1 using symmetric matrix multiplication.'''
    fxn = inspect.stack()[0][3]
    DICT1 = dict_init(DICT1, fxn)
    insufficient_dict_args_1(DICT1, fxn)

    if not DICT1_TRANSPOSE is None:
        DICT1_TRANSPOSE = dict_init(DICT1_TRANSPOSE, fxn)
        insufficient_dict_args_1(DICT1_TRANSPOSE, fxn)
        DICT1_T = DICT1_TRANSPOSE
    else: DICT1_T = sparse_transpose(DICT1)

    # 9/18/22 CHANGED FROM symmetric_matmul TO core_symmetric_matmul FOR SPEED
    _ = core_symmetric_matmul(DICT1_T, DICT1, DICT2_TRANSPOSE=DICT1_T, return_as=return_as)
    return _


def sparse_AAT(DICT1, DICT1_TRANSPOSE=None, return_as=None):
    '''Calculates AAT on DICT1 using symmetric matrix multiplication.'''

    fxn = inspect.stack()[0][3]
    DICT1 = dict_init(DICT1, fxn)
    insufficient_dict_args_1(DICT1, fxn)

    if not DICT1_TRANSPOSE is None:
        DICT1_TRANSPOSE = dict_init(DICT1_TRANSPOSE, fxn)
        insufficient_dict_args_1(DICT1_TRANSPOSE, fxn)
        DICT1_T = DICT1_TRANSPOSE
    else: DICT1_T = sparse_transpose(DICT1)

    # 9/18/22 CHANGED FROM symmetric_matmul TO core_symmetric_matmul FOR SPEED
    return core_symmetric_matmul(DICT1, DICT1_T, DICT2_TRANSPOSE=DICT1, return_as=return_as)


def sparse_matmul_from_lists(LIST1, LIST2, LIST2_TRANSPOSE=None, is_symmetric=False):
    '''Calculates matrix product of two lists and returns as sparse dict.'''
    fxn = inspect.stack()[0][3]
    LIST1 = list_init(LIST1, fxn)[0]
    LIST2 = list_init(LIST2, fxn)[0]
    insufficient_list_args_1(LIST1, fxn)
    insufficient_list_args_1(LIST2, fxn)

    # BROADCAST CHECK
    if len(LIST1[0]) != len(LIST2):
        raise Exception(f'{module_name()}.{fxn}() requires for LIST1(m x n) and LIST2(j x k) that num inner keys (n) of\n'
                        f'LIST1 == num outer keys (j) of LIST2 ---- (m, n) x (j, k) --> (m, k)\n'
                        f'{inner_len(LIST1)} is different than {outer_len(LIST2)}.')

    if not LIST2_TRANSPOSE is None:
        LIST2_TRANSPOSE = list_init(LIST2_TRANSPOSE, fxn)
        insufficient_list_args_1(LIST2_TRANSPOSE, fxn)
        LIST2_T = LIST2_TRANSPOSE
    else: LIST2_T = np.array(LIST2).transpose()

    final_inner_len = len(LIST2_T[0])
    DICT1 = {}
    for outer_idx1 in range(len(LIST1)):
        DICT1[outer_idx1] = {}
        if not is_symmetric: inner_iter_end = final_inner_len
        elif is_symmetric: inner_iter_end = outer_idx1 + 1     # HAVE TO ITER TO WHERE outer_idx2 == outer_idx1 TO GET DIAGONAL
        for outer_idx2 in range(inner_iter_end):
            dot = np.matmul(LIST1[outer_idx1], LIST2[outer_idx2], dtype=float)
            if dot != 0:
                if not is_symmetric:
                    DICT1[int(outer_idx1)][int(outer_idx2)] = dot
                if is_symmetric:
                    DICT1[int(outer_idx1)][int(outer_idx2)] = dot
                    DICT1[int(outer_idx2)][int(outer_idx1)] = dot

    # PLACEHOLDER RULES
    for outer_key in range(len(DICT1)):
        DICT1[int(outer_key)][int(final_inner_len - 1)] = DICT1[outer_key].get(final_inner_len - 1, 0)

    # CHECK outer_len
    return DICT1


def core_dot(DICT1, DICT2):
    '''DICT1 and DICT2 enter as single-keyed outer dicts with dict as value.
    There is no protection here to prevent dissimilar sized DICT1 and DICT2 from dotting.
    '''

    # 9-15-22 VERIFIED 5X FASTER THAN CONVERTING TO np.zeros, FILLING, THEN USING np.matmul
    dict1_key = list(DICT1.keys())[0]
    dict2_key = list(DICT2.keys())[0]
    DOT = 0
    for inner_key1 in DICT1[dict1_key]:
        if inner_key1 in DICT2[dict2_key]:
            DOT += DICT1[dict1_key][inner_key1] * DICT2[dict2_key][inner_key1]

    return DOT


def dot(DICT1, DICT2):
    '''DICT1 and DICT2 enter as single-keyed outer dicts with dict as value.
    Run safeguards that assure dot product rules (dimensionality) are followed when running core_dot().'''

    fxn = inspect.stack()[0][3]
    DICT1 = dict_init(DICT1, fxn)
    DICT2 = dict_init(DICT2, fxn)
    dot_size_check(DICT1, DICT2, fxn)

    return core_dot(DICT1, DICT2)


def core_gaussian_dot(DICT1, DICT2, sigma, return_as='SPARSE_DICT'):
    '''Gaussian dot product.  [] of DICT1 are dotted with [] from DICT2.  There is no protection here to prevent
        dissimilar sized inner dicts from dotting.'''

    UNZIP1 = unzip_to_ndarray_float64(DICT1)[0]
    UNZIP2 = unzip_to_ndarray_float64(DICT2)[0]

    final_inner_len = len(UNZIP2)

    GAUSSIAN_DOT = np.zeros((len(UNZIP1), final_inner_len), dtype=np.float64)

    for outer_key1, INNER_DICT1 in enumerate(UNZIP1):
        for outer_key2, INNER_DICT2 in enumerate(UNZIP2):
            GAUSSIAN_DOT[outer_key1][outer_key2] = np.sum((INNER_DICT1 - INNER_DICT2) ** 2)

    del UNZIP1, UNZIP2

    GAUSSIAN_DOT = np.exp(-GAUSSIAN_DOT / (2 * sigma ** 2))

    if return_as == 'SPARSE_DICT':
        GAUSSIAN_DOT = zip_list_as_py_float(GAUSSIAN_DOT)

    return GAUSSIAN_DOT


def gaussian_dot(DICT1, DICT2, sigma, return_as='SPARSE_DICT'):
    fxn = inspect.stack()[0][3]
    DICT1 = dict_init(DICT1, fxn)
    DICT2 = dict_init(DICT2, fxn)
    inner_len_check(DICT1, DICT2, fxn)

    GAUSSIAN_DOT = core_gaussian_dot(DICT1, DICT2, sigma, return_as=return_as)

    return GAUSSIAN_DOT


def core_symmetric_gaussian_dot(DICT1, sigma, return_as='SPARSE_DICT'):
    '''Gaussian dot product for a symmetric result.  [] of DICT1 are dotted with [] from DICT2.  There is no protection
        here to prevent dissimilar sized inner dicts from dotting.'''

    UNZIP1 = unzip_to_ndarray_float64(DICT1)[0]

    final_inner_len = len(UNZIP1)

    GAUSSIAN_DOT = np.zeros((final_inner_len, final_inner_len), dtype=np.float64)

    for outer_key1 in range(len(UNZIP1)):
        for outer_key2 in range(outer_key1 + 1):   # HAVE TO GET DIAGONAL SO +1
            gaussian_dot = np.sum((UNZIP1[outer_key1] - UNZIP1[outer_key2]) ** 2)
            GAUSSIAN_DOT[outer_key1][outer_key2] = gaussian_dot
            GAUSSIAN_DOT[outer_key2][outer_key1] = gaussian_dot

    del UNZIP1

    GAUSSIAN_DOT = np.exp(-GAUSSIAN_DOT / (2 * sigma ** 2))

    if return_as == 'SPARSE_DICT':
        GAUSSIAN_DOT = zip_list_as_py_float(GAUSSIAN_DOT)

    return GAUSSIAN_DOT


def hybrid_matmul(LIST_OR_DICT1, LIST_OR_DICT2, LIST_OR_DICT2_TRANSPOSE=None, return_as='SPARSE_DICT', return_orientation='ROW'):
    '''Left and right object oriented as []=row, with standard numpy matmul and linear algebra rules, and
        safeguards that assure matrix multiplication rules are followed when running core_hybrid_matmul().'''

    fxn = inspect.stack()[0][3]

    return_as = akv.arg_kwarg_validater(return_as, 'return_as', ['ARRAY', 'SPARSE_DICT', None],
                                        'sparse_dict', fxn, return_if_none='SPARSE_DICT')

    return_orientation = akv.arg_kwarg_validater(return_orientation, 'return_orientation', ['ROW', 'COLUMN', None],
                                        'sparse_dict', fxn, return_if_none='ROW')

    if isinstance(LIST_OR_DICT1, dict) and isinstance(LIST_OR_DICT2, dict):
        raise Exception(f'{module_name()}.{fxn}() LIST_OR_DICT1 AND LIST_OR_DICT2 CANNOT BOTH BE dict. USE sparse_matmul.')

    if isinstance(LIST_OR_DICT1, (np.ndarray, list, tuple)) and isinstance(LIST_OR_DICT2, (np.ndarray, list,tuple)):
        raise Exception(f'{module_name()}.{fxn}() LIST_OR_DICT1 AND LIST_OR_DICT2 CANNOT BOTH BE LIST-TYPE. USE numpy.matmul.')

    if not LIST_OR_DICT2_TRANSPOSE is None and type(LIST_OR_DICT2) != type(LIST_OR_DICT2_TRANSPOSE):
        raise Exception(f'{module_name()}.{fxn}() IF LIST_OR_DICT2_TRANSPOSE IS GIVEN, IT MUST BE THE SAME type AS LIST_OR_DICT2.')

    def dimension_exception(name2, _n, _j): \
        raise Exception(f'{module_name()}.{fxn}() requires for LIST_OR_DICT1 (m x n) and {name2} (j x k) that inner length (n) of ' \
        f'LIST_OR_DICT1 == {f"outer" if name2 == "LIST_OR_DICT2" else "inner"} length (j) of {name2} ---- (m, n) x (j, k) --> (m, k).  ' \
        f'n ({_n}) is different than j ({_j}).')

    if isinstance(LIST_OR_DICT1, dict):
        LIST_OR_DICT1 = dict_init(LIST_OR_DICT1, fxn)
        insufficient_dict_args_1(LIST_OR_DICT1, fxn)

        LIST_OR_DICT2 = list_init(np.array(LIST_OR_DICT2), fxn)[0]
        insufficient_list_args_1(LIST_OR_DICT2, fxn)

        if LIST_OR_DICT2_TRANSPOSE is None:
            _n, _j = inner_len(LIST_OR_DICT1), len(LIST_OR_DICT2)
            if _n != _j: dimension_exception("LIST_OR_DICT2", _n, _j)
        elif not LIST_OR_DICT2_TRANSPOSE is None:
            LIST_OR_DICT2_TRANSPOSE = list_init(np.array(LIST_OR_DICT2_TRANSPOSE), fxn)[0]
            insufficient_list_args_1(LIST_OR_DICT2_TRANSPOSE, fxn)
            _n, _j = inner_len(LIST_OR_DICT1), len(LIST_OR_DICT2_TRANSPOSE[0])
            if _n != _j: dimension_exception("LIST_OR_DICT2_TRANSPOSE", _n, _j)
    elif isinstance(LIST_OR_DICT2, dict):
        LIST_OR_DICT1 = list_init(np.array(LIST_OR_DICT1), fxn)[0]
        insufficient_list_args_1(LIST_OR_DICT1, fxn)

        LIST_OR_DICT2 = dict_init(LIST_OR_DICT2, fxn)
        insufficient_dict_args_1(LIST_OR_DICT2, fxn)

        if LIST_OR_DICT2_TRANSPOSE is None:
            _n, _j = len(LIST_OR_DICT1[0]), outer_len(LIST_OR_DICT2)
            if _n != _j: dimension_exception("LIST_OR_DICT2", _n, _j)
        elif not LIST_OR_DICT2_TRANSPOSE is None:
            LIST_OR_DICT2_TRANSPOSE = dict_init(LIST_OR_DICT2_TRANSPOSE, fxn)
            insufficient_dict_args_1(LIST_OR_DICT2_TRANSPOSE, fxn)
            _n, _j = len(LIST_OR_DICT1[0]), inner_len(LIST_OR_DICT2_TRANSPOSE)
            if _n != _j: dimension_exception("LIST_OR_DICT2_TRANSPOSE", _n, _j)

    del dimension_exception

    return core_hybrid_matmul(LIST_OR_DICT1, LIST_OR_DICT2, LIST_OR_DICT2_TRANSPOSE=LIST_OR_DICT2_TRANSPOSE,
                              return_as=return_as, return_orientation=return_orientation)


def core_hybrid_matmul(LIST_OR_DICT1, LIST_OR_DICT2, LIST_OR_DICT2_TRANSPOSE=None, return_as='SPARSE_DICT',
                       return_orientation='ROW'):
    """Left and right object oriented as []=row, with standard numpy matmul and linear algebra rules. There is
        no protection here to prevent dissimilar sized rows and columns from dotting."""

    if isinstance(LIST_OR_DICT1, dict):
        dict_position='LEFT'
        DICT1 = LIST_OR_DICT1
        if LIST_OR_DICT2_TRANSPOSE is None: LIST1 = np.array(LIST_OR_DICT2).transpose()
        elif not LIST_OR_DICT2_TRANSPOSE is None: LIST1 = np.array(LIST_OR_DICT2_TRANSPOSE)

    elif isinstance(LIST_OR_DICT2, dict):
        dict_position='RIGHT'
        LIST1 = np.array(LIST_OR_DICT1)
        if LIST_OR_DICT2_TRANSPOSE is None: DICT1 = sparse_transpose(LIST_OR_DICT2)
        elif not LIST_OR_DICT2_TRANSPOSE is None: DICT1 = LIST_OR_DICT2_TRANSPOSE

    if len(LIST1.shape)==1: LIST1 = LIST1.reshape((1,-1))  # DONT RESHAPE OTHERWISE, COULD BE 2-D ARRAY

    if dict_position=='LEFT' and return_orientation=='ROW':
        final_outer_len, final_inner_len = outer_len(DICT1), len(LIST1)
    elif dict_position=='RIGHT' and return_orientation=='ROW':
        final_outer_len, final_inner_len = len(LIST1), outer_len(DICT1)
    elif dict_position=='LEFT' and return_orientation=='COLUMN':
        final_outer_len, final_inner_len = len(LIST1), outer_len(DICT1)
    elif dict_position=='RIGHT' and return_orientation=='COLUMN':
        final_outer_len, final_inner_len = outer_len(DICT1), len(LIST1)

    if return_as=='ARRAY': HYBRID_MATMUL = np.zeros((final_outer_len, final_inner_len), dtype=np.float64)
    elif return_as=='SPARSE_DICT': HYBRID_MATMUL = {int(_):{} for _ in range(final_outer_len)}

    if return_orientation=='ROW' and dict_position=='LEFT':
        for outer_idx1 in range(final_outer_len if return_orientation == 'ROW' else final_inner_len):
            for outer_idx2 in range(final_inner_len if return_orientation == 'ROW' else final_outer_len):
                dot = core_hybrid_dot(LIST1[outer_idx2], {0:DICT1[outer_idx1]})
                if dot != 0:
                    HYBRID_MATMUL[int(outer_idx1)][int(outer_idx2)] = dot

    elif return_orientation=='ROW' and dict_position=='RIGHT':
        for outer_idx1 in range(final_outer_len if return_orientation == 'ROW' else final_inner_len):
            for outer_idx2 in range(final_inner_len if return_orientation == 'ROW' else final_outer_len):
                dot = core_hybrid_dot(LIST1[outer_idx1], {0: DICT1[outer_idx2]})
                if dot != 0:
                    HYBRID_MATMUL[int(outer_idx1)][int(outer_idx2)] = dot

    elif return_orientation == 'COLUMN' and dict_position == 'LEFT':
        for outer_idx1 in range(final_outer_len if return_orientation == 'ROW' else final_inner_len):
            for outer_idx2 in range(final_inner_len if return_orientation == 'ROW' else final_outer_len):
                dot = core_hybrid_dot(LIST1[outer_idx2], {0: DICT1[outer_idx1]})
                if dot != 0:
                    HYBRID_MATMUL[int(outer_idx2)][int(outer_idx1)] = dot

    elif return_orientation == 'COLUMN' and dict_position == 'RIGHT':
        for outer_idx1 in range(final_outer_len if return_orientation == 'ROW' else final_inner_len):
            for outer_idx2 in range(final_inner_len if return_orientation == 'ROW' else final_outer_len):
                dot = core_hybrid_dot(LIST1[outer_idx1], {0: DICT1[outer_idx2]})
                if dot != 0:
                    HYBRID_MATMUL[int(outer_idx2)][int(outer_idx1)] = dot

    # IF SPARSE_DICT, ENFORCE PLACEHOLDER RULES
    if return_as == 'SPARSE_DICT':
        for outer_key in range(final_outer_len):
            if final_inner_len-1 not in HYBRID_MATMUL[outer_key]:
                HYBRID_MATMUL[int(outer_key)][int(final_inner_len-1)] = 0

    return HYBRID_MATMUL


def core_hybrid_dot(LIST1, DICT1):
    """Dot product of a single list and one outer sparse dict without any safeguards to ensure single vectors of same length."""
    LIST1 = np.array(LIST1)
    if len(LIST1.shape)==1: LIST1 = LIST1.reshape((1,-1))
    DICT1[0] = DICT1[list(DICT1.keys())[0]]
    dot = 0
    for inner_idx in DICT1[0]:   # CANT TO USE set.intersection HERE, LIST IS FULL AND NOT INDEXED
        dot += LIST1[0][inner_idx] * DICT1[0][inner_idx]

    return dot

# END LINEAR ALGEBRA ####################################################################################################################
#########################################################################################################################################
#########################################################################################################################################

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
    '''Function called by decorators of specific matrix math functions.'''
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
    '''Function called by decorators of specific scalar math functions.'''
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






















