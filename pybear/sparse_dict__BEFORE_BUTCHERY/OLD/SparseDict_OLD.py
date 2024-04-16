import numpy as n, pandas as p
from copy import deepcopy
from debug import get_module_name as gmn
import sys, inspect
from pybear.data_validation import validate_user_input as vui

# FOR 50 COLS x 100 ROWS:
# WHEN NOT SYMMETRIC W/ 10% SPARSE, MATMUL = 0.05s, INDIV DOTS = 0.24s, DICT DOTS = 0.07s
# WHEN NOT SYMMETRIC W/ 90% SPARSE, MATMUL = 0.04s, INDIV DOTS = 0.22s, DICT DOTS = 0.03s
# WHEN SYMMETRIC W/ 10% SPARSE,     MATMUL = 0.05s, INDIV DOTS = 0.13s, DICT DOTS = 0.05s
# WHEN SYMMETRIC W/ 90% SPARSE,     MATMUL = 0.04s, INDIV DOTS = 0.12s, DICT DOTS = 0.03s

# FOR 500 COLS x 1000 ROWS:
# WHEN NOT SYMMETRIC W/ 10% SPARSE, MATMUL = 36.62s, INDIV DOTS = 50.34s, DICT DOTS = 13.04s
# WHEN NOT SYMMETRIC W/ 90% SPARSE, MATMUL = 35.41s, INDIV DOTS = 48.32s, DICT DOTS = 4.04s
# WHEN SYMMETRIC W/ 10% SPARSE,     MATMUL = 36.97s, INDIV DOTS = 29.17s, DICT DOTS = 7.76s
# WHEN SYMMETRIC W/ 90% SPARSE,     MATMUL = 34.60s, INDIV DOTS = 25.02s, DICT DOTS = 2.51s

'''
    __init__ CHECKS & EXCEPTIONS ##########################################################################################################
    list_check                      Require that LIST kwarg is a list-type, and is not ragged.
    sparse_dict_check               Require that objects to be processed as sparse dictionaries follow sparse dictionary rules.
    datadict_check                  Require that objects to be processed as data dictionaries follow data dict rules: dictionary with list-type as values.
    dateframe_check                 Verify DATAFRAME kwarg is a dataframe.
    END __init__ CHECKS & EXCEPTIONS ######################################################################################################
    RUNTIME CHECKS & EXCEPTIONS ###########################################################################################################
    non_int                         Verify integer.
    insufficient_list_kwargs_1      Verify LIST kwarg is filled when processing a function that requres a list.
    insufficient_dict_kwargs_1      Verify DICT1 kwarg is filled when processing a function that requres one dictionary.
    insufficient_dict_kwargs_2      Verify DICT1 and DICT2 kwargs are filled when processing a function that requres two dictionaries.
    insufficient_datadict_kwargs_1  Verify DATADICT1 kwarg is filled when processing a function that requres one data dictionary.
    insufficient_dataframe_kwargs_1 Verify DATAFRAME kwarg is filled when processing a function that requres one dataframe.
    dot_size_check                  Verify two vectors are sparse dicts that both have unitary outer length and equal inner length.
    broadcast_check                 Verify two sparse dicts follow standard matrix multiplication rules (m, n) x (j, k) ---> n == j.
    matrix_size_check               Verify two sparse dicts have equal outer and inner length.
    outer_len_check                 Verify two sparse dicts have equal outer length.
    inner_len_check                 Verify two sparse dicts have equal inner length.
    symmetric_matmul_check          Verify two sparse dicts will matrix multiply to a symmetric matrix.
    END RUNTIME CHECKS & EXCEPTIONS #######################################################################################################
    CREATION, HANDLING & MAINTENANCE ######################################################################################################
    create_random                   Create a sparse matrix with user-given dimensions filled with random numbers of user-given sparsity.
    zip_list                        Convert list-type (list, tuple, array) of list-types as [[]] to sparse dict.
    zip_datadict                    Convert data dict as {_:[], __:[], ...} to a sparse dictionary.
    zip_dataframe                   Convert dataframe to sparse dict.
    unzip_to_list                   Convert sparse dict to list of lists.
    unzip_to_datadict               Convert sparse dict to datadict of lists.
    unzip_to_full_dict              Convert sparse dict to dict with full indices.
    unzip_to_dataframe              Convert sparse dict to dataframe.
    clean                           Remove any 0 values, enforce contiguous zero-indexed outer keys, build any missing final inner key placeholders.
    resize_inner                    Resize sparse dict to user-entered inner dict length.  Reducing size may truncate non-zero values;
                                    increasing size will introduce zeros (empties) and original inner size placeholder rule (entry for last item even if 0) holds.
    #resize_outer                   Resize sparse dict to user-entered outer dict length.  Reducing size may truncate non-zero values;
                                    increasing size will introduce zeros (placeholder inner dicts) and original outer size placeholder rule holds.
    #resize                         Resize sparse dict to user-entered (len outer dict, len inner dicts) dimensions.  Reducing size may truncate non-zero values;
                                    increasing size will introduce zeros (empties in inner dicts, placeholder in outer dict) and original size placeholder rules hold.
    #merge_outer                    Merge outer dictionaries of 2 dictionaries.  Inner dictionary lengths must be equal.
    #merge_inner                    Merge inner dictionaries of 2 dictionaries.  Outer dictionary lengths must be equal.
    #delete_outer_key               Equivalent to deleting a row or a column.
    #delete_inner_key               Equivalent to deleting a row or a column.
    # END CREATION, HANDLING & MAINTENANCE ##################################################################################################
    # ABOUT #################################################################################################################################
    len_base                        Function called by decorators of inner_len and outer_len calculation functions.
    inner_len                       Length of inner dictionaries that are held by the outer dictionary for DICT1.
    inner_len1                      Length of inner dictionaries that are held by the outer dictionary for self.DICT1.  For use internally.
    inner_len2                      Length of inner dictionaries that are held by the outer dictionary for self.DICT2.  For use internally.
    outer_len                       Length of the outer dictionary that holds the inner dictionaries as values, aka number of inner dictionaries
    outer_len1                      Length of the outer dictionary that holds the inner dictionaries as values for self.DICT1.  For use interally.
    outer_len2                      Length of the outer dictionary that holds the inner dictionaries as values for self.DICT2.  For use interally.
    size_base                       Function called by decorators of size calculation functions.
    size                            Return <outer dict length, inner dict length> as tuple.
    size1                           Return <outer dict length of self.DICT1, inner dict length of self.DICT1> as tuple, for use internally.
    size2                           Return <outer dict length of self.DICT2, inner dict length of self.DICT2> as tuple, for use internally.
    sum_over_outer_key              Sum all the values in an inner dict, as given by outer dict key.
    sum_over_inner_key              Sum over all inner dicts the values that are keyed with the user-entered inner key.
    sparsity                        Calculate sparsity of a sparse dict.
    list_sparsity                   Calculate sparsity of a list-type of list-types.
    display                         Print sparse dict to screen.
    sparse_equiv                      Check for equivalence of two sparse dictionaries.
    summary_stats                   Function called by decorators of specific summary statistics functions.
    sum_                            Sum of all values of a sparse dictionary, across all inner dictionaries.
    median_                         Median of all values of a sparse dictionary, across all inner dictionaries.
    average_                        Average of all values of a sparse dictionary, across all inner dictionaries.
    max_                            Maximum value in a sparse dictionary, across all inner dictionaries.
    min_                            Minimum value in a sparse dictionary, across all inner dictionaries.
    centroid_                       Centroid of a sparse dictionary.
    # END ABOUT #############################################################################################################################
    # LINEAR ALGEBRA ########################################################################################################################
    sparse_transpose_base           Function called by decorators of DICT1 or DICT2 transposing functions.
    sparse_transpose                Transpose a sparse dict to a sparse dict.
    sparse_transpose1               Transpose self.DICT1 as a sparse dict to a sparse dict, for use internally.
    sparse_transpose2               Transpose self.DICT2 as a sparse dict to a sparse dict, for use internally.
    core_matmul                     DICT1 and DICT2 enter oriented with sparse inner dicts as rows, with multiplication rules like NumPy.  There is
                                    no protection here to prevent dissimilar sized rows from DICT1 dotting with columns from DICT2.
                                    Create posn for last entry, so that placeholder rules are enforced (original length of object is retained).
    core_symmetric_matmul           DICT1 and DICT2 enter oriented with sparse inner dicts as rows, with multiplication rules like NumPy.
                                    For use on things like ATA and AAT to save time.  There is no protection here to prevent dissimilar sized
                                    rows from DICT1 dotting with columns from DICT2.  Create posn for last entry, so that placeholder rules are
                                    enforced (original length of object is retained).
    matmul                          Run safeguards that assure matrix multiplication rules are followed when running core_matmul().
    symmetric_matmul                Run safeguards that assure matrix multiplication rules are followed when running core_symmetric_matmul().
    sparse_ATA                      Calculates ATA on DICT1 using symmetric matrix multiplication.
    sparse_AAT                      Calculates AAT on DICT1 using symmetric matrix multiplication.
    core_dot                        Standard dot product. DICT1 and DICT2 enter as single-keyed outer dicts with dict as value.
                                    There is no protection here to prevent dissimilar sized DICT1 and DICT2 from dotting.
    dot                             Standard dot product.  DICT1 and DICT2 enter as single-keyed outer dicts with dict as value.
                                    Run safeguards that assure dot product rules (dimensionality) are followed when running core_dot().
    core_gaussian_dot               Gaussian dot product.  DICT1 and DICT2 must enter as single-keyed outer dicts with dict as value.
                                    There is no protection here to prevent multi-outer-key dicts and dissimilar sized inner dicts from dotting.
    gaussian_dot                    Gaussian dot product.  DICT1 and DICT2 must enter as single-keyed outer dicts with dict as value.
                                    Run safeguards that assure dot product rules (dimensionality) are followed when running core_dot().
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
    # END  SPARSE FUNCTIONS ###################################################################################################################
'''

class SparseDict:
    ''' Take list-type of list-types (or just one list-type), convert to a dictionary of sparse dictionaries.
        SparseDict converts data to dictionaries that hold non-zero values as values, with index positions as keys.
        e.g. [[0 0 1 0 3 0 1]] is {0: {2:1, 4:3, 6:1} }
        Always create posn for last entry, so that original length of list is retained.  I.e., if the last entry
        of a list is 0, put it in the dict anyway to placehold the original length of the list.
        Always create a dictionary for every list.  [[0,0,1],[0,0,0],[1,1,0]] looks like { 0:{2:1}, 1:[2:0}, 2:{0:1, 1:1, 2:0} }
        The chances of an entire list (whether it is meant to be a column or a row) being all zeros is small, meaning
        there would be little gained by dropping such rows, but the original dimensionality could be lost.
    '''


    def __init__(self, DICT1=None, DICT1_HEADER1=None, DICT2=None, DICT2_HEADER1=None, LIST1=None, LIST_HEADER1=None,
                     DATADICT1=None, DATADICT_HEADER1=None, DATAFRAME1=None, DATAFRAME_HEADER1=None):
        self.module = gmn.get_module_name(str(sys.modules[__name__]))

        if DICT1_HEADER1 is None: self.DICT1_HEADER1 = [[]]
        else: self.DICT1_HEADER1 = DICT1_HEADER1

        if DICT1 is None:
            self.DICT1 = dict()
        else:
            self.DICT1 = DICT1
            self.sparse_dict_check(self.DICT1, 'DICT1')

        if DICT2_HEADER1 is None: self.DICT2_HEADER1 = [[]]
        else: self.DICT2_HEADER1 = DICT2_HEADER1

        if DICT2 is None:
            self.DICT2 = dict()
        else:
            self.DICT2 = DICT2
            self.sparse_dict_check(self.DICT2, 'DICT2')

        if LIST_HEADER1 is None: self.LIST_HEADER1 = [[]]
        else: self.LIST_HEADER1 = LIST_HEADER1

        if LIST1 is None: self.LIST1 = list()
        else:
            self.LIST1 = LIST1
            self.list_check(self.LIST1)


        if DATADICT_HEADER1 is None: self.DATADICT_HEADER1 = [[]]
        else: self.DATADICT_HEADER1 = DATADICT_HEADER1

        if DATADICT1 is None:
            self.DATADICT1 = dict()
        else:
            self.DATADICT1 = DATADICT1
            self.datadict_check(self.DATADICT1, 'DATADICT1')
            # BUILD DATADICT_HEADER1 AND REKEY DATADICT OUTER KEYS NOW TO LOCK IN SEQUENCE AND NOT LET NON-INT KEY GET PAST HERE!
            key_counter = 0
            for key in list(self.DATADICT1.keys()):
                self.DATADICT_HEADER1[0].append(key)
                self.DATADICT1[key_counter] = self.DATADICT1.pop(key)
                key_counter += 1

        if DATAFRAME_HEADER1 is None: self.DATAFRAME_HEADER1 = [[]]
        else: self.DATAFRAME_HEADER1 = DATAFRAME_HEADER1

        if DATAFRAME1 is None:
            self.DATAFRAME1 = p.DataFrame({})
        else:
            self.DATAFRAME1 = DATAFRAME1
            self.dataframe_check(self.DATAFRAME1, 'DATAFRAME1')
            # BUILD DATAFRAME_HEADER1 AND REKEY DATAFRAME OUTER KEYS NOW TO LOCK IN SEQUENCE AND NOT LET NON-INT KEY GET PAST HERE!
            key_counter = 0
            for key in list(self.DATAFRAME1.keys()):
                self.DATAFRAME_HEADER1[0].append(key)
                self.DATAFRAME1[key_counter] = self.DATAFRAME1.pop(key)
                key_counter += 1

    #########################################################################################################################################
    #########################################################################################################################################
    # __init__ CHECKS & EXCEPTIONS ##########################################################################################################

    def list_check(self, LIST1):
        '''Require that LIST kwarg is a list-type, and is not ragged.'''
        if True not in map(lambda x: x in str(type(LIST1)).upper(), ('LIST', 'ARRAY', 'TUPLE')):
            raise Exception(f'{self.module} requires a list, array, or tuple as LIST kwarg. Cannot be set or dictionary.')

        with n.errstate(all='ignore'):
            if True in [n.array_equiv(_, LIST1) for _ in n.array([[[], [[]], (), (())]], dtype=object)]:
                raise Exception(f'{self.module} LIST1 kwarg is an empty list-type.')

        INNER_LEN_HOLDER = []
        for _ in LIST1:
            if True not in [__ in str(type(_)).upper() for __ in ['LIST', 'ARRAY', 'TUPLE']]:
                raise Exception(f'{self.module} requires LIST kwarg be a list, array, or tuple of lists, arrays, or '
                                f'tuples, i.e. [[],[]] or ((),()).')

            INNER_LEN_HOLDER.append(len(_))

        if n.min(INNER_LEN_HOLDER) != n.max(INNER_LEN_HOLDER):
            raise Exception(f'{self.module} requires list-type of list-types with all inner list-types having equal length.')


    def sparse_dict_check(self, DICT1_or_2, name):
        '''Require that objects to be processed as sparse dictionaries follow sparse dictionary rules.'''
        if not isinstance(DICT1_or_2, dict):
            raise Exception(f'{self.module} {name} requires dictionary as input.)')

        if DICT1_or_2 == {}:
            raise Exception(f'{self.module} {name} input is an empty dictionary.)')

        for _ in DICT1_or_2.values():
            if not isinstance(_, dict):
                raise Exception(f'{self.module} {name} requires input to be a dictionary with values that are dictionaries.')

        for key in DICT1_or_2:
            if not isinstance(key, int):
                raise Exception(f'{self.module} {name} requires all outer keys be integers.')
            for subkey in DICT1_or_2[key]:
                if not isinstance(subkey, int):
                    raise Exception(f'{self.module} {name} requires all inner keys be integers.')


    def datadict_check(self, DATADICT1, name):
        '''Require that objects to be processed as data dictionaries follow data dict rules: dictionary with list-type as values.'''
        if not isinstance(DATADICT1, dict):
            raise Exception(f'{self.module} {name} requires dictionary as input.)')

        if DATADICT1 == {}:
            raise Exception(f'{self.module} {name} input is an empty dictionary.)')

        for _ in DATADICT1.values():
            if True not in [__ in str(type(_)).upper() for __ in ['LIST','ARRAY','TUPLE']]:
                raise Exception(f'{self.module} {name} requires input to be a dictionary with values that are list-types.')


    def dataframe_check(self, DATAFRAME1, name):
        '''Verify DATAFRAME kwarg is a dataframe.'''
        if 'DATAFRAME' not in str(type(DATAFRAME1)).upper():
            raise Exception(f'{self.module} {name} requires input to be a Pandas DataFrame.')

    # END __init__ CHECKS & EXCEPTIONS ######################################################################################################
    #########################################################################################################################################
    #########################################################################################################################################

    #########################################################################################################################################
    #########################################################################################################################################
    # RUNTIME CHECKS & EXCEPTIONS ###########################################################################################################

    def non_int(self, value, fxn, param_name):
        '''Verify integer.'''
        if not isinstance(value, int):
            raise Exception(f'{self.module}.{fxn}() requires {param_name} entered as integer.')


    def insufficient_list_kwargs_1(self, fxn):
        '''Verify LIST kwarg is filled when processing a function that requres a list.'''
        with n.errstate(all='ignore'):
            if n.array_equiv(self.LIST1, []):
                raise Exception(f'{self.module}.{fxn}() requires one list-type kwarg, LIST1.')


    def insufficient_dict_kwargs_1(self, fxn):
        '''Verify DICT1 kwarg is filled when processing a function that requres one dictionary.'''
        if self.DICT1 == {}:
            raise Exception(f'{self.module}.{fxn}() requires {self.module} take one dictionary kwarg, DICT1.')


    def insufficient_dict_kwargs_2(self, fxn):
        '''Verify DICT1 and DICT2 kwargs are filled when processing a function that requres two dictionaries.'''
        if self.DICT1 == {} or self.DICT2 == {}:
            raise Exception(f'{self.module}.{fxn}() requires two dictionary kwargs, DICT1 and DICT2.')


    def insufficient_datadict_kwargs_1(self, fxn):
        '''Verify DATADICT1 kwarg is filled when processing a function that requres one data dictionary.'''
        if self.DATADICT1 == {}:
            raise Exception(f'{self.module}.{fxn}() requires one dictionary of list-types as kwarg, DATADICT1.')


    def insufficient_dataframe_kwargs_1(self, fxn):
        '''Verify DATAFRAME kwarg is filled when processing a function that requres one dataframe.'''
        if self.DATAFRAME1.equals(p.DataFrame({})):
            raise Exception(f'{self.module}.{fxn}() requires a Pandas DataFrame as kwarg, DATAFRAME1.')


    def dot_size_check(self, fxn):
        '''Verify two vectors are sparse dicts that both have unitary outer length and equal inner length.'''
        if len(self.DICT1) != 1 or len(self.DICT2) != 1:
            raise Exception(f'{self.module}.{fxn}() requires dictionaries with one integer key, one dict as values.)')
        if self.inner_len1() != self.inner_len2():
            raise Exception(f'{self.module}.{fxn}() requires 2 dictionaries of equal stated length (last keys are equal).)')


    def broadcast_check(self, fxn):        # DO THIS BEFORE TRANSPOSING DICT 2
        '''Verify two sparse dicts follow standard matrix multiplication rules (m, n) x (j, k) ---> n == j.'''
        if self.inner_len1() != self.outer_len2():
            raise Exception(f'{self.module}.{fxn}() requires for DICT1(m x n) and DICT2(j x k) that num inner keys (n) of '
                            f'DICT1 == num outer keys (j) of DICT2 (before transpose) ---- (m, n) x (j, k) --> (m, k)')


    def matrix_size_check(self, fxn):
        '''Verify two sparse dicts have equal outer and inner length.'''
        _size1, _size2 = self.size1(), self.size2()
        if _size1 != _size2:
            raise Exception(f'{self.module}.{fxn}() requires both sparse dicts to be equally sized.  Dict 1 is {_size1[0]} x '
                            f'{_size1[1]} and Dict 2 is {_size2[0]} x {_size2[1]}')


    def outer_len_check(self, fxn):
        '''Verify two sparse dicts have equal outer length.'''
        outer_len1 = self.outer_len1()
        outer_len2 = self.outer_len2()
        if outer_len1 != outer_len2:
            raise Exception(
                f'{self.module}.{fxn}() requires both sparse dicts to have equal outer length.  Dict 1 is {outer_len1} '
                f'and Dict 2 is {outer_len2}')


    def inner_len_check(self, fxn):
        '''Verify two sparse dicts have equal inner length.'''
        inner_len1 = self.inner_len1()
        inner_len2 = self.inner_len2()
        if inner_len1 != inner_len2:
            raise Exception(
                f'{self.module}.{fxn}() requires both sparse dicts to have equal inner length.  Dict 1 is {inner_len1} '
                f'and Dict 2 is {inner_len2}')


    def symmetric_matmul_check(self):
        '''Verify two sparse dicts will matrix multiply to a symmetric matrix.'''
        DICT1 = deepcopy(self.DICT1)
        DICT2 = deepcopy(self.DICT2)

        # TEST DICT2 IS TRANSPOSE OF DICT1
        # TRANSPOSE DICT2
        self.sparse_transpose2()
        test1 = self.sparse_equiv()
        # TRANSPOSE DICT2 BACK
        self.sparse_transpose2()

        if test1: return True

        # TEST BOTH DICT1 AND DICT2 ARE SYMMETRIC
        # SET self.DICT2 TO DICT1
        self.DICT2 = deepcopy(self.DICT1)
        # TAKE TRANSPOSE OF DICT1 IN self.DICT2 POSN
        self.sparse_transpose2()
        test2 = self.sparse_equiv()
        # PUT DICT2 BACK INTO self.DICT2 POSN
        self.DICT2 = deepcopy(DICT2)

        # SET DICT1 POSN TO DICT2
        self.DICT1 = deepcopy(self.DICT2)
        # TAKE TRANSPOSE OF DICT2 IN self.DICT2 POSN
        self.sparse_transpose2()
        test3 = self.sparse_equiv()
        # PUT DICT1 BACK INTO self.DICT1 POSN
        self.DICT1 = deepcopy(DICT1)

        if test2 and test3: return True
        else: return False

    # END RUNTIME CHECKS & EXCEPTIONS #######################################################################################################
    #########################################################################################################################################
    #########################################################################################################################################

    #########################################################################################################################################
    #########################################################################################################################################
    # CREATION, HANDLING & MAINTENANCE ######################################################################################################

    def create_random(self, len_outer, len_inner, sparsity):
        '''Create a sparse matrix with user-given dimensions filled with random integers from 0 to 9 of user-given percent sparsity.'''
        fxn = inspect.stack()[0][3]

        self.non_int(len_outer, fxn, "len_outer")
        self.non_int(len_inner, fxn, "len_inner")

        if sparsity > 100 or sparsity < 0:
            raise Exception(f'{self.module}.{fxn}() sparsity must be 0 to 100.')

        if sparsity != 100:
            SPARSE_DICT = {}
            for outer_key in range(len_outer):
                SPARSE_DICT[outer_key] = {}
                for inner_key in range(len_inner):
                    hit = True if n.random.randint(0, 10000) < 100 * (100-sparsity) else False
                    if hit: SPARSE_DICT[outer_key][inner_key] = n.random.randint(1, 10)
                    elif not hit and inner_key == len_inner - 1: SPARSE_DICT[outer_key][inner_key] = 0

        elif sparsity == 100:
            SPARSE_DICT = {outer_idx:{len_inner-1: 0} for outer_idx in range(len_outer)}

        return SPARSE_DICT


    def zip_list(self):
        '''Convert list-type (list, tuple, array) of list-types as [[]] to sparse dict.'''
        # COULD BE [ [] = ROWS ] OR [ [] = COLUMNS ], BUT MUST BE [[]]
        '''Create posn for last entry even if value is zero, so that original length of object is retained.'''

        self.insufficient_list_kwargs_1(inspect.stack()[0][3])

        SPARSE_DICT = {}
        for outer_key, inner_list_type in enumerate(self.LIST1):
            SPARSE_DICT[outer_key] = {}
            for inner_key,value in enumerate(inner_list_type):
                if value != 0 or inner_key == len(inner_list_type) - 1:
                    SPARSE_DICT[outer_key][inner_key] = value

        return SPARSE_DICT


    def zip_datadict(self):
        '''Convert data dict as {_:[], __:[], ...} to a sparse dictionary.'''
        # {'a':LIST-TYPE, 'b':LIST-TYPE} COULD BE = ROWS OR COLUMNS
        #  Create posn for last entry even if value is zero, so that original length of object is retained.

        self.insufficient_datadict_kwargs_1(inspect.stack()[0][3])

        self.DICT1 = {}
        for key in range(len(list(self.DATADICT1.keys()))):
            _ = self.DATADICT1[key]
            #   ik = inner_key, v = values, ilt = inner_list_type
            self.DICT1[key] = {ik: v for ik, v in enumerate(_) if (v != 0 or ik == len(_) - 1)}

        self.sparse_transpose1()   # COMES IN AS {'header': {COLUMNAR DATA LIST}} SO TRANSPOSE SPARSE DICT TO {} = ROWS

        self.sparse_dict_check(self.DICT1, 'DICT1')   # TO VERIFY CONVERSION WAS DONE CORRECTLY

        return self.DICT1, self.DATADICT_HEADER1


    def zip_dataframe(self):
        '''Convert dataframe to sparse dict.  Returns sparse dict object and the extracted header as tuple.'''
        # PANDAS DataFrame
        # Create posn for last entry even if value is zero, so that original length of object is retained.

        self.insufficient_dataframe_kwargs_1(inspect.stack()[0][3])

        self.DICT1 = {}
        for key in range(len(list(self.DATAFRAME1.keys()))):
            _ = self.DATAFRAME1[key]
            #   ik = inner_key, v = values, ilt = inner_list_type
            self.DICT1[key] = {ik: v for ik, v in enumerate(_) if (v != 0 or ik == len(_) - 1)}

        #DataFrame CAN BE CREATED FROM:
        # = p.DataFrame(data=LIST, columns=[])   CANNOT BE data=DICT or data=DICT OF DICTS
        # = p.DataFrame(DICT OF LISTS)      WITH DICT KEY AS HEADER
        # = p.DataFrame(DICT OF DICTS)      MUST HAVE VALUE FOR EVERY INNER KEY OR GIVES NaN, THEREFORE CANT JUST PUT A SPARSE DICT STRAIGHT IN

        self.sparse_transpose1()   # DF COMES IN AS {'header': {COLUMNAR DATA LIST}} SO TRANSPOSE SPARSE DICT TO {} = ROWS

        self.sparse_dict_check(self.DICT1, 'DICT1')   # TO VERIFY CONVERSION WAS DONE CORRECTLY

        return self.DICT1, self.DATAFRAME_HEADER1


    def unzip_to_list(self):
        '''Convert sparse dict to list of lists.'''
        self.insufficient_dict_kwargs_1(inspect.stack()[0][3])

        _outer_len = self.outer_len1()
        _inner_len = self.inner_len1()

        is_empty = True in [n.array_equiv(self.LIST_HEADER1, _) for _ in [[[]], None] ]

        if not is_empty and len(self.LIST_HEADER1[0]) != _inner_len:
            raise Exception(f'# list columns ({_inner_len}) must equal # header positions ({len(self.LIST_HEADER1[0])}).')

        LIST_OF_LISTS = [[0 for _ in range(_inner_len)] for _ in range(_outer_len)]
        for outer_key in range(_outer_len):
            for inner_key in range(_inner_len):
                try: LIST_OF_LISTS[outer_key][inner_key] = self.DICT1[outer_key].get(inner_key, 0)
                except: LIST_OF_LISTS[outer_key][inner_key] = 0

        return LIST_OF_LISTS, self.LIST_HEADER1


    def unzip_to_full_dict(self):
        '''Convert sparse dict to dict with full indices.'''
        self.insufficient_dict_kwargs_1(inspect.stack()[0][3])

        self.clean()

        _outer_len = self.outer_len1()
        _inner_len = self.inner_len1()

        for outer_key in range(_outer_len):
            for inner_key in range(_inner_len):
                if inner_key not in list(self.DICT1[outer_key].keys()):
                    self.DICT1[outer_key][inner_key] = 0

        return self.DICT1


    def unzip_to_datadict(self):
        '''Convert sparse dict to datadict of lists.'''
        self.insufficient_dict_kwargs_1(inspect.stack()[0][3])

        self.sparse_transpose1()

        _outer_len = self.outer_len1()
        _inner_len = self.inner_len1()

        is_empty = True in [n.array_equiv(self.DATADICT_HEADER1, _) for _ in [[[]], None]]

        if not is_empty and len(self.DATADICT_HEADER1[0]) != _outer_len:
            raise Exception(f'# Datadict columns ({_outer_len}) must equal # header positions ({len(self.DATADICT_HEADER1[0])}).')

        self.DATADICT1 = {}

        for outer_key in range(_outer_len):
            # IF A HEADER IS AVAILABLE, REPLACE OUTER INT KEYS W HEADER
            if not is_empty: self.DATADICT1[self.DATADICT_HEADER1[0][outer_key]] = []
            else:
                self.DATADICT1[outer_key] = outer_key
                self.DATADICT_HEADER1[0].append(outer_key)

        for outer_key in range(len(list(self.DICT1.keys()))):
            for inner_key in range(_inner_len):
                if inner_key not in list(self.DICT1[outer_key].keys()):
                    self.DATADICT1[self.DATADICT_HEADER1[0][outer_key]].append(0)
                else:
                    self.DATADICT1[self.DATADICT_HEADER1[0][outer_key]].append(self.DICT1[outer_key][inner_key])

        self.datadict_check(self.DATADICT1, 'DATADICT1')

        return self.DATADICT1, self.DATADICT_HEADER1


    def unzip_to_dataframe(self):
        '''Convert sparse dict to dataframe.'''
        self.insufficient_dict_kwargs_1(inspect.stack()[0][3])

        # CONVERT FULL DICT TO DATAFRAME
        # CAME IN COLUMNAR AS {'header':{DATA LIST} AND WAS TRANSPOSE TO OUTER {} = ROWS WHEN ZIPPED, SO TRANSPOSE BACK TO {} = COLUMN

        self.DATADICT_HEADER1 = self.DATAFRAME_HEADER1
        self.DATADICT1, DUM = self.unzip_to_datadict()

        _outer_len = self.outer_len1()
        _inner_len = self.inner_len1()

        is_empty = True in [n.array_equiv(self.LIST_HEADER1, _) for _ in [[[]], None]]

        if not is_empty and len(self.DATAFRAME_HEADER1[0]) != _outer_len:
            raise Exception(f'# DataFrame columns ({_outer_len}) must equal # header positions ({len(self.DATAFRAME_HEADER1[0])}).')

        if not is_empty:
            # IF AVAILABLE, PUT DATAFRAME_HEADER AS KEYS OF DATADICT1 & BUILD DF
            KEYS = list(self.DATADICT1.keys())
            for _ in range(len(KEYS)):
                self.DATADICT1[self.DATAFRAME_HEADER1[0][_]] = self.DATADICT1.pop(KEYS[_])

            self.DATAFRAME1 = p.DataFrame(data=self.DATADICT1, columns=self.DATAFRAME_HEADER1[0])
        else:
            # IF NO HEADER, JUST BUILD DF
            self.DATAFRAME1 = p.DataFrame(self.DATADICT1)

        self.dataframe_check(self.DATAFRAME1, 'DATAFRAME1')


        return self.DATAFRAME1


    def clean(self):
        '''Remove any 0 values, enforce contiguous zero-indexed outer keys, build any missing final inner key placeholders.'''

        self.insufficient_dict_kwargs_1(inspect.stack()[0][3])

        # ENSURE OUTER DICT KEYS ARE CONTIGUOUS, IDXed TO ZERO
        while True:
            # CHECK IF KEYS START AT ZERO AND ARE CONTIGUOUS, IF OK, break
            # FIND ACTUAL len, SEE IF OUTER KEYS MATCH EVERY POSN IN THAT RANGE
            if False not in [_ in self.DICT1 for _ in range(len(list(self.DICT1.keys())))]:
                break

            # IF NOT, REASSIGN keys TO DICT, IDXed TO 0
            else:
                KEYS = deepcopy(set(self.DICT1.keys()))
                for _, __ in enumerate(KEYS):
                    self.DICT1[_] = self.DICT1.pop(__)
            break

        # ENSURE INNER DICT PLACEHOLDER RULE (KEY FOR LAST POSN, EVEN IF VALUE IS ZERO) IS ENFORCED
        max_inner_key = int(n.max([n.max(list(self.DICT1[outer_key].keys())) for outer_key in self.DICT1]))
        for outer_key in self.DICT1:
            if max_inner_key not in self.DICT1[outer_key]:
                self.DICT1[outer_key][max_inner_key] = 0

        # ENSURE THERE ARE NO ZERO VALUES IN ANY INNER DICT, BUT ONE FOR PLACEHOLDER
        # CREATE A deepcopy DICT TO THRASH APART
        DEL_HOLDER = []
        WIP_DICT = deepcopy(self.DICT1)
        [WIP_DICT[outer_key].pop(max_inner_key) for outer_key in WIP_DICT]  # TAKE OUT PLACEHOLDER ZEROS
        for outer_key in WIP_DICT:
            if 0 in list(WIP_DICT[outer_key].values()):
                for inner_key in WIP_DICT[outer_key]:
                    if self.DICT1[outer_key][inner_key] == 0:
                        # del self.DICT1[outer_key][inner_key] GIVES RuntimeError: dictionary changed size during iteration
                        DEL_HOLDER.append((outer_key, inner_key))

        #print(f'\nClean up (enforcement of outer key rules, inner key placeholder rules, and non-zero value rules) complete.')

        for outer_key, inner_key in DEL_HOLDER:
            del self.DICT1[outer_key][inner_key]

        del DEL_HOLDER

        return self.DICT1


    # CURRENTLY ONLY CALLED BY resize()
    def resize_inner(self, new_inner_len, calling_fxn=None):   # LAST IDX IS ALWAYS len()-1, DUE TO ZERO INDEX
        '''Resize sparse dict to user-entered inner dict length.  Reducing size may truncate non-zero values;
            increasing size will introduce zeros (empties) and original inner size placeholder rule (entry for last item even if 0) holds.'''
        fxn = inspect.stack()[0][3]
        self.insufficient_dict_kwargs_1(fxn)
        self.non_int(new_inner_len, fxn, "new_inner_len")

        while True:
            if calling_fxn == 'DUMMY PLACEHOLDER':   # ALLOW USER SHORT CIRCUIT IN PROCESS.... NOT IN USE
                if vui.validate_user_str(f'\nReally proceed with inner dict resize?  Non-zero data will might be lost (y/n) > ', 'YN') == 'N':
                    break

            self.clean()

            is_empty = True in [n.array_equiv(self.DICT1_HEADER1, _) for _ in [ [[]], None ] ]

            old_inner_len = self.inner_len1()

            if new_inner_len == old_inner_len:  # NEW INNER LEN IS SAME AS OLD, DO NOTHING
                pass

            elif new_inner_len > old_inner_len:
                # DELETE OLD PLACEHOLDERS (key = old inner len - 1 and value == 0, if value != 0 then not a placeholder, dont delete)
                # PUT NEW PLACEHOLDER AT new_len_inner - 1
                if not is_empty:
                    __ = self.inner_len1()
                    for new_column in range(__, new_inner_len):
                        self.DICT1_HEADER1[0].append(__)

                for outer_key in self.DICT1:
                    if self.DICT1[outer_key][old_inner_len - 1] == 0: del self.DICT1[outer_key][inner_key]
                    self.DICT1[new_inner_len-1] = 0


            elif new_inner_len < old_inner_len:
                # DELETE ANYTHING AFTER old_inner_len - 1, PUT NEW PLACEHOLDERS as new_inner_len - 1 if NEEDED
                for outer_key in self.DICT1:
                    for inner_key in range(new_inner_len, old_inner_len):
                        if inner_key in self.DICT1[outer_key]: del self.DICT1[outer_key][inner_key]
                    if new_inner_len - 1 not in self.DICT1[outer_key]:
                        self.DICT1[outer_key][new_inner_len-1] = 0

                if not is_empty:
                    for inner_key in range(new_inner_len, old_inner_len):
                        self.DICT1_HEADER1[0].pop(inner_key)

            break

        print(f'\nResize of inner dictionaries complete.')

        return self.DICT1, self.DICT1_HEADER1


    # CURRENTLY ONLY CALLED BY resize()
    def resize_outer(self, new_outer_len, calling_fxn=None):   # LAST IDX IS ALWAYS len()-1, DUE TO ZERO INDEX
        '''Resize sparse dict to user-entered outer dict length.  Reducing size may truncate non-zero values;
            increasing size will introduce zeros (placeholder inner dicts) and original outer size placeholder rule holds.'''

        self.insufficient_dict_kwargs_1(inspect.stack()[0][3])
        self.non_int(new_outer_len, fxn, "new_outer_len")

        while True:
            if calling_fxn == 'DUMMY_PLACEHOLDER':   # ALLOW USER SHORT CIRCUIT IN PROCESS.... NOT IN USE
                if vui.validate_user_str(f'\nReally proceed with outer dict resize?  Non-zero data will might be lost (y/n) > ', 'YN') == 'N':
                    break

            self.clean()

            old_outer_len = self.outer_len1()

            is_empty = True in [n.array_equiv(self.DICT1_HEADER1, _) for _ in [[[]], None]]

            if new_outer_len == old_outer_len:    # NEW INNER LEN IS SAME AS OLD, DO NOTHING
                pass

            elif new_outer_len > old_outer_len:
                # PUT PLACEHOLDERS IN THE NEW KEYS
                for outer_key in range(old_outer_len, new_outer_len):
                    self.DICT1[outer_key] = {self.inner_len1(): 0}
                    if not is_empty:
                        self.DICT1_HEADER1[0].append(outer_key)


            elif new_outer_len < old_outer_len:
                for outer_key in range(new_outer_len, old_outer_len):
                    del self.DICT1[outer_key]
                    if not is_empty:
                        self.DICT1_HEADER1[0].pop(outer_key)

            break

        print(f'\nResize of outer dictionary complete.')

        return self.DICT1, self.DICT1_HEADER1


    def resize(self, len_outer_key, len_inner_key, header_goes_on=None):
        # LAST OUTER AND INNER IDXS ARE ALWAYS len()-1, DUE TO ZERO INDEXING
        '''Resize sparse dict to user-entered (len outer dict, len inner dicts) dimensions.  Reducing size may truncate non-zero values;
            increasing size will introduce zeros (empties in inner dicts, placeholder in outer dict) and original size placeholder rules hold.'''
        fxn = inspect.stack()[0][3]
        self.insufficient_dict_kwargs_1(fxn)
        self.non_int(len_outer_key, fxn, "len_outer_key")
        self.non_int(len_inner_key, fxn, "len_inner_key")

        if isinstance(header_goes_on, str): header_goes_on = header_goes_on.upper()

        if header_goes_on == 'OUTER':
            DICT1, self.DICT1_HEADER1 = resize_outer(DICT1, len_outer_key, calling_fxn=fxn)
            DICT1, DUM = resize_inner(DICT1, len_inner_key, calling_fxn=fxn)
        elif header_goes_on == 'INNER':
            DICT1, DUM = resize_outer(DICT1, len_outer_key, calling_fxn=fxn)
            DICT1, self.DICT1_HEADER1 = resize_inner(DICT1, len_inner_key, calling_fxn=fxn)
        elif header_goes_on is None:
            DICT1, DUM = resize_outer(DICT1, len_outer_key, calling_fxn=fxn)
            DICT1, self.DICT1_HEADER1 = resize_inner(DICT1, len_inner_key, calling_fxn=fxn)
        else:
            raise ValueError(f'\nINVALID header_goes_on IN {self.module_name}.{fxn}().  MUST BE "outer" or "inner".')

        return self.DICT1, self.DICT1_HEADER1


    def merge_outer(self):
        '''Merge outer dictionaries of 2 dictionaries.  Inner dictionary lengths must be equal.'''

        self.insufficient_dict_kwargs_2(inspect.stack()[0][3])
        self.inner_len_check(fxn)

        self.clean()    # ONLY CLEANING DICT1 AT THIS POINT, CLEAN DICT2 NOT AVAILABLE

        # CANT JUST MERGE THE 2 DICTS, THEY MIGHT (PROBABLY) HAVE MATCHING OUTER KEYS AND OVERWRITE
        # GET outer_len of DICT1 TO KNOW HOW TO INDEX DICT2, REINDEX DICT2 ON THE FLY
        _outer_len = self.outer_len1()

        counter = 0
        for outer_key in deepcopy(self.DICT2):
            self.DICT1[_outer_len + counter] = self.DICT2[outer_key]
            counter += 1

        self.clean()

        if not True in [n.array_equiv(self.DICT1_HEADER1, _) for _ in [ [[]], None ] ] and \
                        not True in [n.array_equiv(self.DICT2_HEADER1, _) for _ in [[[]], None]]:
            self.DICT1_HEADER1 = [[ *self.DICT1_HEADER1[0], *self.DICT2_HEADER1[0] ]]

        return self.DICT1, self.DICT1_HEADER1


    def merge_inner(self):
        '''Merge inner dictionaries of 2 dictionaries.  Outer dictionary lengths must be equal.'''
        fxn = inspect.stack()[0][3]
        self.insufficient_dict_kwargs_2(fxn)
        self.clean()
        self.outer_len_check(fxn)

        # CANT JUST MERGE THE 2 DICTS, THEY MIGHT (PROBABLY) HAVE MATCHING INNER KEYS AND OVERWRITE
        # GET inner_len of DICT1 TO KNOW HOW TO INDEX DICT2, REINDEX DICT2 ON THE FLY
        _inner_len1 = self.inner_len1()
        _inner_len2 = self.inner_len2()
        combined_inner_len = _inner_len1 + _inner_len2
        pseudo_dict2_outer_key = 0
        for outer_key in self.DICT2.keys():   # DICT1 outer len must == DICT 2 outer len
            # CURRENTLY UNABLE TO CLEAN DICT2, SO IF OUTER KEYS NOT CONTIGUOUS, USE PSEUDOKEY TO MATCH AGAINST DICT1
            # CHECK TO SEE IF VALUE AT END OF DICT1 INNER IS 0, IF SO, DELETE
            if self.DICT1[pseudo_dict2_outer_key][_inner_len1-1] == 0: del self.DICT1[pseudo_dict2_outer_key][_inner_len1-1]
            for inner_key in self.DICT2[outer_key]:
                self.DICT1[pseudo_dict2_outer_key][_inner_len1 + inner_key] = self.DICT2[outer_key][inner_key]
            else: # WHEN GET TO LAST INNER KEY, ENFORCE PLACEHOLDING RULES
                self.DICT1[pseudo_dict2_outer_key][combined_inner_len - 1] = \
                    self.DICT1[pseudo_dict2_outer_key].get(combined_inner_len - 1, 0)

            pseudo_dict2_outer_key += 1

        if not True in [n.array_equiv(self.DICT1_HEADER1, _) for _ in [ [[]], None ] ] and \
                        not True in [n.array_equiv(DICT2_HEADER1, _) for _ in [[[]], None]]:
            self.DICT1_HEADER1 = [[ *self.DICT1_HEADER1[0], *self.DICT2_HEADER1[0] ]]

        return self.DICT1, self.DICT1_HEADER1


    def delete_outer_key(self, delete_key):
        '''Equivalent to deleting a row or a column.'''
        fxn = inspect.stack()[0][3]
        self.insufficient_dict_kwargs_1(fxn)
        self.non_int(delete_key, fxn, "key")

        self.clean()
        _outer_len = self.outer_len1()

        if delete_key < 0 or delete_key > _outer_len - 1:
            raise Exception(f'outer key {delete_key} out of bounds for {self.module}.{fxn}().  '
                            f'Must be between 0 and {_outer_len-1}.')

        del self.DICT1[delete_key]

        # IF NOT DELETING THE LAST OUTER KEY, KNOCK DOWN keys OF DICTS THAT COME AFTER
        if delete_key != _outer_len - 1:
            for outer_key in range(delete_key+1, _outer_len):
                self.DICT1[outer_key-1] = self.DICT1.pop(outer_key)

        if not True in [n.array_equiv(self.DICT1_HEADER1, _) for _ in [ [[]], None ] ]:
            self.DICT1_HEADER1[0] = self.DICT1_HEADER1[0].pop(delete_key)

        print(f'\nDelete of outer key {delete_key} complete.')

        return self.DICT1, self.DICT1_HEADER1[0]


    def delete_inner_key(self, delete_key):
        '''Equivalent to deleting a row or a column.'''
        fxn = inspect.stack()[0][3]
        self.insufficient_dict_kwargs_1(fxn)
        self.non_int(delete_key, fxn, "key")

        self.clean()
        _inner_len = self.inner_len1()

        is_empty = True in [n.array_equiv(self.DICT1_HEADER1, _) for _ in [ [[]], None ] ]

        if delete_key < 0 or delete_key > _inner_len - 1:
            raise Exception(f'inner key {delete_key} out of bounds for {self.module}.{fxn}().  '
                            f'Must be between 0 and {_inner_len-1}.')

        # MUST REDUCE ALL keys AFTER delete_key BY 1!!! DONT JUST SET delete_key TO ZERO BY JUST DELETING IT!!!
        for outer_key in self.DICT1:

            for key in set(self.DICT1[outer_key].keys()):
                if key == delete_key: del self.DICT1[outer_key][key]
                if delete_key < _inner_len - 1 and key > delete_key:  # WHEN NOT DELETING THE LAST INNER KEY
                    self.DICT1[outer_key][key-1] = self.DICT1[outer_key].pop(key)
                elif delete_key == _inner_len - 1: # WHEN DELETING THE LAST inner_key IN INNER DICT, MANAGE PLACEHOLDER RULES
                    if delete_key-1 not in set(self.DICT1[outer_key].keys()):
                        self.DICT1[outer_key][delete_key-1] = 0

        if not is_empty:
            self.DICT1_HEADER1[0].pop(delete_key)

        print(f'\nDelete of inner key {delete_key} complete.')

        return self.DICT1, self.DICT1_HEADER1

    # END CREATION, HANDLING & MAINTENANCE ##################################################################################################
    #########################################################################################################################################
    #########################################################################################################################################


    #########################################################################################################################################
    #########################################################################################################################################
    # ABOUT #################################################################################################################################

    def len_base(orig_func):
        '''Function called by decorators of inner_len and outer_len calculation functions.'''
        if 'inner_len1' in str(orig_func): fxn = 'inner_len1'
        elif 'inner_len2' in str(orig_func): fxn = 'inner_len2'
        elif 'inner_len' in str(orig_func): fxn = 'inner_len'
        elif 'outer_len1' in str(orig_func): fxn = 'outer_len1'
        elif 'outer_len2' in str(orig_func): fxn = 'outer_len2'
        elif 'outer_len' in str(orig_func): fxn = 'outer_len'

        def _len(self):

            self.insufficient_dict_kwargs_1(fxn)  # THINK THIS WILL POINTLESSLY LOOK AT self.DICT1 WHEN USING 'inner_len' AND 'outer_len'
            # OTHERWISE THIS WILL BE USEFUL.  THINKING THAT PLACES THAT WOULD BE CALLING len_base AND LOOKING FOR len OF self.DICT2
            # WILL HAVE ALREADY RUN insufficient_dict_kwargs_2 WITHIN THEMSELVES

            # DONT BOTHER TO clean() OR resize() HERE, SINCE ONLY THE SCALAR LENGTH IS RETURNED (CHANGES TO self.DICTX ARENT RETAINED)
            if 'inner' in fxn:
                try: inner_len = [max(list(orig_func(self)[_].keys())) + 1 for _ in orig_func(self)]
                except: raise Exception(f'Sparse dictionary has a zero-len inner dictionary in {self.module}.{fxn}()')
                return max(inner_len)

            if 'outer' in fxn:
                try: outer_len = len(list(orig_func(self).keys()))
                except: raise Exception(f'Sparse dictionary is a zero-len outer dictionary in {self.module}.{fxn}()')
                return outer_len

        return _len

    @len_base
    def inner_len(self):
        '''Length of inner dictionaries that are held by the outer dictionary'''
        return self.DICT1

    @len_base
    def inner_len1(self):
        '''Length of inner dictionaries that are held by the outer dictionary for self.DICT1.  For use internally.'''
        return self.DICT1

    @len_base
    def inner_len2(self):
        '''Length of inner dictionaries that are held by the outer dictionary for self.DICT2.  For use internally.'''
        return self.DICT2

    @len_base
    def outer_len(self):
        '''Length of the outer dictionary that holds the inner dictionaries as values, aka number of inner dictionaries'''
        return self.DICT1

    @len_base
    def outer_len1(self):
        '''Length of the outer dictionary that holds the inner dictionaries as values for self.DICT1.  For use interally.'''
        return self.DICT1

    @len_base
    def outer_len2(self):
        '''Length of the outer dictionary that holds the inner dictionaries as values for self.DICT2.  For use interally.'''
        return self.DICT2


    def size_base(orig_func):
        '''Function called by decorators of size calculation functions.'''
        def size_(self):
            # DONT NEED kwarg CHECK HERE, HANDLED BY outer_len() & inner_len()
            if 'size2' in str(orig_func):   # IF size2(), DO GYMNASTICS TO GET self.DICT2 INTO self.DICT1 SLOT
                self.DICT1, self.DICT2 = deepcopy(self.DICT2), deepcopy(self.DICT1)
                outer_len, inner_len = self.outer_len(), self.inner_len()
                # AFTER GETTING LEN OF self.DICT2, PUT DICTS BACK IN TO CORRECT SLOTS
                self.DICT1, self.DICT2 = deepcopy(self.DICT2), deepcopy(self.DICT1)
            else:
                # ELIF size1() OR size(), LEAVE self.DICT1 IN self.DICT1
                outer_len, inner_len = self.outer_len(), self.inner_len()

            return (outer_len, inner_len)

        return size_

    @size_base
    def size_(self):
        '''Return <outer dict length, inner dict length> as tuple.'''
        pass

    @size_base
    def size1(self):
        '''Return <outer dict length of self.DICT1, inner dict length of self.DICT1> as tuple, for use internally'''
        pass

    @size_base
    def size2(self):
        '''Return <outer dict length of self.DICT2, inner dict length of self.DICT2> as tuple, for use internally'''
        pass


    def sum_over_outer_key(self, outer_key):
        '''Sum all the values in an inner dict, as given by outer dict key.'''

        self.insufficient_dict_kwargs_1(inspect.stack()[0][3])
        return sum(list(self.DICT1[outer_key].values()))


    def sum_over_inner_key(self, inner_key):
        '''Sum over all inner dicts the values that are keyed with the user-entered inner key.'''

        self.insufficient_dict_kwargs_1(inspect.stack()[0][3])
        SUM = 0
        for outer_key in self.DICT1:
            if inner_key not in range(self.inner_len1()):
                raise Exception(f'Key {inner_key} out of bounds for inner dict with len {self.inner_len1()}.')
            if inner_key in self.DICT1[outer_key]:
                SUM += self.DICT1[outer_key][inner_key]
        return SUM


    def sparsity(self):
        '''Calculate sparsity of a sparse dict.'''
        fxn = inspect.stack()[0][3]
        self.insufficient_dict_kwargs_1(fxn)
        SIZE = self.size1()
        hits = 0
        for outer_key in range(SIZE[0]):
            hits += len([_ for _ in list(self.DICT1[outer_key].values()) if _ != 0])
        return 100 - 100 * hits / (SIZE[0] * SIZE[1])


    def list_sparsity(self):
        '''Calculate sparsity of a list-type of list-types.'''
        fxn = inspect.stack()[0][3]
        self.insufficient_list_kwargs_1(fxn)
        size = 0
        for LIST in self.LIST1:
            size += len(LIST)

        zeros = n.sum(n.array(self.LIST1, dtype=object) == 0)

        return 100 * zeros / size


    def sparse_equiv(self):
        '''Check for equivalence of two sparse dictionaries.'''

        self.insufficient_dict_kwargs_2(inspect.stack()[0][3])

        if self.size1() != self.size2(): return False

        _inner_len = self.inner_len1()
        for outer_key in self.DICT1:
            for inner_key in range(_inner_len):
                if self.DICT1[outer_key].get(inner_key, 0) != self.DICT2[outer_key].get(inner_key, 0):
                    return False
        else:
            return True


    def display(self, number_of_inner_dicts_to_print=float('inf')):
        '''Print sparse dict to screen.'''

        self.insufficient_dict_kwargs_1(inspect.stack()[0][3])

        _len = len(self.DICT1)

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
            VALID_OUTER_KEYS = set(self.DICT1.keys())
            print_count = 0
            outer_key = 0
            while print_count < num_rows:
                _ = outer_key
                if _ in VALID_OUTER_KEYS:
                    print(f'{str(_)}:'.ljust(4) +f'{str(self.DICT1[_])[:100]}' + (f' ...' if len(str(self.DICT1[_])) > 70 else ''))
                    print_count += 1
                outer_key += 1
            print()

            break


    def summary_stats(orig_func):
        '''Function called by decorators of specific summary statistics functions.'''
        __ = ['sum_', 'median_', 'average_', 'max_', 'min_', 'centroid_']
        fxn_idx = [op in str(orig_func) for op in __].index(True)
        fxn = __[fxn_idx]

        def statistics(self):
            self.insufficient_dict_kwargs_1(fxn)

            NON_ZERO_ELEMENTS = []
            for _ in self.DICT1:
                NON_ZERO_ELEMENTS += [self.DICT1[_][i] for i in self.DICT1[_] if self.DICT1[_][i] !=0]   # REMEMBER PLACEHOLDERS!

            SIZE = self.size1()
            total_elements = SIZE[0] * SIZE[1]   # MUST ACCOUNT FOR ALL THE ZEROS!

            return orig_func(self, NON_ZERO_ELEMENTS, total_elements)

        return statistics

    @summary_stats
    def sum_(self, NON_ZERO_ELEMENTS, total_elements):
        '''Sum of all values of a sparse dictionary, across all inner dictionaries.'''
        return n.sum(NON_ZERO_ELEMENTS)  # PLACEHOLDER ZEROS CAN BE IGNORED

    @summary_stats
    def median_(self, NON_ZERO_ELEMENTS, total_elements):
        '''Median of all values of a sparse dictionary, across all inner dictionaries.'''
        return n.median(NON_ZERO_ELEMENTS + [0 for _ in range((total_elements - len(NON_ZERO_ELEMENTS)))])

    @summary_stats
    def average_(self, NON_ZERO_ELEMENTS, total_elements):
        '''Average of all values of a sparse dictionary, across all inner dictionaries.'''
        return n.sum(NON_ZERO_ELEMENTS) / total_elements

    @summary_stats
    def max_(self, NON_ZERO_ELEMENTS, total_elements):
        '''Maximum value in a sparse dictionary, across all inner dictionaries.'''
        return n.max(NON_ZERO_ELEMENTS + [0])

    @summary_stats
    def min_(self, NON_ZERO_ELEMENTS, total_elements):
        '''Minimum value in a sparse dictionary, across all inner dictionaries.'''
        return n.min(NON_ZERO_ELEMENTS + [0])

    @summary_stats
    def centroid_(self, NON_ZERO_ELEMENTS, total_elements):
        '''Centroid of a sparse dictionary.'''

        SUM = self.vector_sum(set(self.DICT1.keys()))

        _outer_len = self.outer_len1()      # DO THIS BEFORE CHANGING self.DICT1 to SUM
        # MUST CHANGE self.DICT1 to SUM IN ORDER TO USE scalar_divide ON IT
        self.DICT1 = SUM
        CENTROID = self.scalar_divide(_outer_len)
        return CENTROID


    # END ABOUT #############################################################################################################################
    #########################################################################################################################################
    #########################################################################################################################################

    #########################################################################################################################################
    #########################################################################################################################################
    # LINEAR ALGEBRA ########################################################################################################################

    def sparse_transpose_base(orig_func):
        '''Function called by decorators of DICT1 or DICT2 transposing functions.'''
        # FUNCTION BELOW ONLY OPERATES ON self.DICT1, SO NEED GYMNASTICS TO PRESERVE ACTUAL DICT1 WHEN OTHER THINGS ARE
        # MOVED INTO THE self.DICT1 POSN
        if 'transpose1' in str(orig_func): fxn = 'transpose1'
        elif 'transpose2' in str(orig_func): fxn = 'transpose2'
        else: fxn = 'transpose'

        def _sparse_transpose(self):

            if 'transpose2' in str(orig_func): # self.DICT2 MOVES TO self.DICT1 POSN, OTHERWISE NO JOCKEYING
                self.DICT1, self.DICT2 = deepcopy(self.DICT2), deepcopy(self.DICT1)

            self.insufficient_dict_kwargs_1(fxn)

            self.clean()

            _outer_len = self.outer_len1()
            _inner_len = self.inner_len1()

            '''Inner len becomes outer len, and outer len becomes inner len.'''
            TRANSPOSE_DICT = {_:{} for _ in range(_inner_len)}

            for outer_idx in range(_outer_len):
                for inner_idx in range(_inner_len):
                    '''If any TRANSPOSE inner dicts do not have a key for len(dict)-1, create one with a zero to maintain
                        placeholding rule.'''
                    if outer_idx == _outer_len-1:
                        TRANSPOSE_DICT[inner_idx][outer_idx] = self.DICT1[outer_idx].get(inner_idx, 0)
                        continue

                    '''If at end of each original inner dict (should be a value there due to placeholding from zip() and clean()),
                        skip putting this in TRANSPOSE (otherwise last inner dict in TRANSPOSE would get full of zeros).'''
                    if inner_idx == _inner_len-1 and self.DICT1[outer_idx].get(inner_idx, 0) == 0: continue

                    # DONT USE get() HERE, IF NO VALUE, JUST continue
                    try: TRANSPOSE_DICT[inner_idx][outer_idx] = self.DICT1[outer_idx][inner_idx]
                    except: pass

            # RESTORATIVE GYMNASTICS
            if fxn == 'transpose1':
                self.DICT1 = TRANSPOSE_DICT
            elif fxn == 'transpose2':
                self.DICT1, self.DICT2 = deepcopy(self.DICT2), TRANSPOSE_DICT
            else:
                return TRANSPOSE_DICT

        return _sparse_transpose


    @sparse_transpose_base
    def sparse_transpose(self):
        '''Transpose a sparse dict to a sparse dict.'''
        pass

    @sparse_transpose_base
    def sparse_transpose1(self):
        '''Transpose self.DICT1 as a sparse dict to a sparse dict, for use internally.'''
        pass

    @sparse_transpose_base
    def sparse_transpose2(self):
        '''Transpose self.DICT2 as a sparse dict to a sparse dict, for use internally.'''
        pass


    def core_matmul(self):
        '''DICT1 and DICT2 enter oriented with sparse inner dicts as rows, with multiplication rules like NumPy.  There is
            no protection here to prevent dissimilar sized rows from DICT1 dotting with columns from DICT2.
            Create posn for last entry, so that placeholder rules are enforced (original length of object is retained).'''

        # Transpose DICT2 for ease of multiplication
        self.sparse_transpose2()

        _outer_len_2 = self.outer_len2()

        OUTPUT = {_:{} for _ in range(self.outer_len1())}

        for outer_dict_idx2 in self.DICT2:
            for outer_dict_idx1 in self.DICT1:
                dot = 0
                for inner_dict_idx in self.DICT1[outer_dict_idx1]:
                    dot += self.DICT1[outer_dict_idx1][inner_dict_idx] * self.DICT2[outer_dict_idx2].get(inner_dict_idx, 0)
                if dot != 0: OUTPUT[outer_dict_idx1][outer_dict_idx2] = dot
                if outer_dict_idx2 == _outer_len_2-1 and outer_dict_idx2 not in OUTPUT[outer_dict_idx1]:
                    OUTPUT[outer_dict_idx1][outer_dict_idx2] = 0

        return OUTPUT


    def core_symmetric_matmul(self):
        '''DICT1 and DICT2 enter oriented with sparse inner dicts as rows, with multiplication rules like NumPy.
        For use on things like ATA and AAT to save time.  There is no protection here to prevent dissimilar sized
        rows from DICT1 dotting with columns from DICT2.  Create posn for last entry, so that placeholder rules are
        enforced (original length of object is retained).'''

        # Transpose DICT2 for ease of multiplication.
        self.sparse_transpose2()

        _outer_len = self.outer_len2()

        OUTPUT = {_: {} for _ in range(self.outer_len1())}

        for outer_dict_idx2 in self.DICT2:
            for outer_dict_idx1 in range(outer_dict_idx2 + 1):
                dot = 0
                for inner_dict_idx in self.DICT1[outer_dict_idx1]:
                    dot += self.DICT1[outer_dict_idx1][inner_dict_idx] * self.DICT2[outer_dict_idx2].get(inner_dict_idx, 0)
                if dot != 0:
                    OUTPUT[outer_dict_idx1][outer_dict_idx2] = dot
                    OUTPUT[outer_dict_idx2][outer_dict_idx1] = dot
                if outer_dict_idx2 == _outer_len - 1 and outer_dict_idx2 not in OUTPUT[outer_dict_idx1]:
                    OUTPUT[outer_dict_idx1][outer_dict_idx2] = 0

        return OUTPUT


    def matmul(self):
        '''Run safeguards that assure matrix multiplication rules are followed when running core_matmul().'''
        fxn = inspect.stack()[0][3]
        self.broadcast_check(fxn)   # DO THIS BEFORE TRANSPOSING DICT2

        return self.core_matmul()


    def symmetric_matmul(self):
        '''Run safeguards that assure matrix multiplication rules are followed when running core_symmetric_matmul().'''

        self.broadcast_check(inspect.stack()[0][3])   # DO THIS BEFORE TRANSPOSING DICT2
        self.symmetric_matmul_check()

        return self.core_symmetric_matmul()


    def sparse_ATA(self):
        '''Calculates ATA on DICT1 using symmetric matrix multiplication.'''

        self.insufficient_dict_kwargs_1(inspect.stack()[0][3])

        self.DICT2 = deepcopy(self.DICT1)
        self.sparse_transpose1()

        return self.symmetric_matmul()


    def sparse_AAT(self):
        '''Calculates AAT on DICT1 using symmetric matrix multiplication.'''

        self.insufficient_dict_kwargs_1(inspect.stack()[0][3])

        self.DICT2 = deepcopy(self.DICT1)
        self.sparse_transpose2()

        return self.symmetric_matmul()


    def core_dot(self):
        '''DICT1 and DICT2 enter as single-keyed outer dicts with dict as value.
        There is no protection here to prevent dissimilar sized DICT1 and DICT2 from dotting.'''

        dict1_key = list(self.DICT1.keys())[0]
        dict2_key = list(self.DICT2.keys())[0]
        DOT = 0
        for _ in self.DICT1[dict1_key]:
            DOT += self.DICT1[dict1_key][_] * self.DICT2[dict2_key].get(_, 0)

        return DOT


    def dot(self):
        '''DICT1 and DICT2 enter as single-keyed outer dicts with dict as value.
        Run safeguards that assure dot product rules (dimensionality) are followed when running core_dot().'''

        self.dot_size_check(inspect.stack()[0][3])

        OUTPUT = self.core_dot()
        return OUTPUT


    def core_gaussian_dot(self, sigma):
        '''DICT1 and DICT2 must enter as single-keyed outer dicts with dict as value.
        There is no protection here to prevent multi-outer-key dicts and dissimilar sized inner dicts from dotting.'''

        DOT = 0
        dict1_key = list(self.DICT1.keys())[0]
        dict2_key = list(self.DICT2.keys())[0]
        COMBINED_DICT = self.DICT1[dict1_key] | self.DICT2[dict2_key]

        for inner_key in COMBINED_DICT:
            DOT += ((self.DICT1[dict1_key].get(inner_key, 0) - self.DICT2[dict2_key].get(inner_key, 0)) ** 2) / (2 * sigma**2)

        return n.exp(-DOT)


    def gaussian_dot(self, sigma):

        self.dot_size_check(inspect.stack()[0][3])

        DOT = self.core_gaussian_dot(sigma)
        return DOT

    # END LINEAR ALGEBRA ####################################################################################################################
    #########################################################################################################################################
    #########################################################################################################################################

    #########################################################################################################################################
    #########################################################################################################################################
    # GENERAL MATH ##########################################################################################################################

    # SPARSE MATRIX MATH ####################################################################################################################

    def vector_sum(self, OUTER_KEYS_AS_SET):
        '''Vector sum of user-specified outer dictionaries, with outer keys given by set. '''

        exception_text = f'Outer dict key out of range for {self.module}.{inspect.stack()[0][3]}().  ' + \
                            f'Must be between 0 and len(DICT)-1, with value '
        if max(OUTER_KEYS_AS_SET) > max(set(self.DICT1.keys())):
            raise Exception(exception_text + f'{max(OUTER_KEYS_AS_SET)}.')
        elif min(OUTER_KEYS_AS_SET) < 0:
            raise Exception(exception_text + f'{min(OUTER_KEYS_AS_SET)}.')

        _inner_len = self.inner_len1()
        VECTOR_SUM = {0: {_: 0 for _ in range(_inner_len)}}
        for outer_key in OUTER_KEYS_AS_SET:
            for inner_key in self.DICT1[outer_key]:
                VECTOR_SUM[0][inner_key] += self.DICT1[outer_key][inner_key]

        if _inner_len - 1 not in VECTOR_SUM[0]:   # ENFORCE PLACEHOLDER RULE
            VECTOR_SUM[0][_inner_len-1] = 0

        return VECTOR_SUM


    def sparse_matrix_math(orig_func):
        '''Function called by decorators of specific matrix math functions.'''
        # CRAZY WAY TO GET FUNCTION NAME TO FEED INTO operation, SINCE @ IS PASSING THE FUNCTION IN HERE
        FUNCTIONS = ['add', 'subtract', 'multiply', 'divide']
        operation = [__ for __ in range(len(FUNCTIONS)) if FUNCTIONS[__] in str(orig_func)][0]
        operation = FUNCTIONS[operation]

        def core_matrix_math(self):

            self.insufficient_dict_kwargs_2(f'matrix_' + f'{operation}')
            self.matrix_size_check(f'matrix_' + f'{operation}')

            # MUST BUILD FROM SCRATCH TO BUILD CORRECT ORDERING IN INNER DICTS, CANT MAINTAIN ORDER IF DOING | ON COPYS OF DICT1 & DICT2
            FINAL_DICT = dict()
            TO_DELETE_HOLDER = []
            _inner_len = self.inner_len()
            for outer_key in self.DICT1:   # outer len of DICT1 & DICT2 already been checked, must be equal & contiguous
                '''FOR ADD, SUBTRACT, MULTIPLY, & DIVIDE, ALL MATCHING ZEROS BETWEEN DICT1 AND DICT2 STAY AT ZERO, SO OK TO SKIP
                    OVER ALL LOCATIONS WHERE DOUBLE ZEROS (THIS ALLOWS FOR PLACES THAT WOULD BE 0/0 TO GET THRU!)'''
                if operation == 'divide' and True in [0 in list(self.DICT2[_].values()) for _ in self.DICT2]:
                    raise Exception(f'0/0 division error in {self.module}.{operation}().')

                # ASSEMBLAGE OF NON-ZERO INNER KEYS FROM DICT1 & DICT2 ASCENDING**************************************************************
                '''IF USE | OR ** ON {0:a, 2:b}, {1:c, 3:d} GET {0:a, 2:b, 1:c, 3:d} SO HAVE TO DO SOMETHING THAT SORTS '''
                INNER_DICT_SORTED_KEYS = set([*self.DICT1[outer_key].keys(), *self.DICT2[outer_key].keys()])
                FINAL_DICT = FINAL_DICT | {outer_key:{_:'' for _ in INNER_DICT_SORTED_KEYS}}
                # ****************************************************************************************************************************

                for inner_key in reversed(list(INNER_DICT_SORTED_KEYS)):  # REVERSED KEEPS INNER DICTS IN ORDER, BUT REVERSED
                    # NOTHING ELSE IS KEEPING INNER DICTS ORDERED :(
                    result = orig_func(self, self.DICT1[outer_key].get(inner_key, 0), self.DICT2[outer_key].get(inner_key, 0))

                    if result != 0: FINAL_DICT[outer_key][inner_key] = result   # IF NON-ZERO, UPDATE KEY W NEW VALUE
                    # IF == 0 AND NOT ON LAST LOCATION, DEL LOCATION
                    if result == 0 and inner_key != _inner_len-1: TO_DELETE_HOLDER.append((outer_key, inner_key))
                    # IF == 0 AND ON LAST LOCATION, ENSURE PLACEHOLDER RULES ARE FOLLOWED
                    if result == 0 and inner_key == _inner_len-1: FINAL_DICT[outer_key][inner_key] = 0

            # Expection WHEN TRYING TO DELETE FROM DICTIONARY ON THE FLY, SO QUEUE DELETIONS UNTIL END
            for outer_key, inner_key in TO_DELETE_HOLDER:
                del FINAL_DICT[outer_key][inner_key]

            del TO_DELETE_HOLDER

            return FINAL_DICT

        return core_matrix_math


    @sparse_matrix_math
    def matrix_add(self, elem1, elem2):
        '''Element-wise addition of two sparse dictionaires representing identically sized matrices.'''
        return elem1 + elem2

    @sparse_matrix_math
    def matrix_subtract(self, elem1, elem2):
        '''Element-wise subtraction of two sparse dictionaires representing identically sized matrices.'''
        return elem1 - elem2

    @sparse_matrix_math
    def matrix_multiply(self, elem1, elem2):
        '''Element-wise multiplication of two sparse dictionaires representing identically sized matrices.'''
        return elem1 * elem2

    @sparse_matrix_math
    def matrix_divide(self, elem1, elem2):
        '''Element-wise division of two sparse dictionaires representing identically sized matrices.'''
        return elem1 / elem2
    # END SPARSE MATRIX MATH #################################################################################################################

    # SPARSE SCALAR MATH #####################################################################################################################
    def sparse_scalar_math(orig_func):
        '''Function called by decorators of specific scalar math functions.'''
        # CRAZY WAY TO GET FUNCTION NAME TO FEED INTO operation, SINCE @ IS PASSING THE FUNCTION IN HERE
        FUNCTIONS = ['add', 'subtract', 'multiply', 'divide', 'power', 'exponentiate']
        operation = [__ for __ in range(len(FUNCTIONS)) if FUNCTIONS[__] in str(orig_func)][0]
        operation = FUNCTIONS[operation]

        def core_scalar_math(self, scalar):

            self.insufficient_dict_kwargs_1(f'scalar_' + f'{operation}')

            TO_DELETE_HOLDER = []
            _inner_len = self.inner_len1()
            HOLDER_DICT = {_:{} for _ in range(self.outer_len1())}   # DOING THIS BECAUSE THE NATURE OF THE OPERATION CAUSES KEYS TO GO OUT OF ORDER
            for outer_key in self.DICT1:

                for inner_key in reversed(range(_inner_len)):   # MUST HIT ALL POSITIONS
                    result = orig_func(self, self.DICT1[outer_key].get(inner_key,0), scalar)

                    if result != 0: HOLDER_DICT[outer_key][inner_key] = result   # IF NON-ZERO, UPDATE (OR CREATE) KEY W NEW VALUE
                    # IF == 0 AND NOT ON LAST LOCATION, DEL LOCATION
                    elif result == 0 and inner_key != _inner_len-1 and inner_key in self.DICT1[outer_key]:
                        TO_DELETE_HOLDER.append((outer_key, inner_key))
                    # IF == 0 AND ON LAST LOCATION, ENSURE PLACEHOLDER RULES ARE FOLLOWED
                    elif result == 0 and inner_key == _inner_len-1: HOLDER_DICT[outer_key][inner_key] = 0

            # Expection WHEN TRYING TO DELETE FROM DICTIONARY ON THE FLY, SO QUEUE DELETIONS UNTIL END
            for outer_key, inner_key in TO_DELETE_HOLDER:
                del self.DICT1[outer_key][inner_key]

            del TO_DELETE_HOLDER

            # for outer_dict in self.DICT1:
            #     self.DICT1[outer_dict].sort()

            return HOLDER_DICT

        return core_scalar_math


    @sparse_scalar_math
    def scalar_add(self, elem, scalar):
        '''Element-wise addition of a scalar to a sparse dictionary representing a matrix.'''
        return elem + scalar

    @sparse_scalar_math
    def scalar_subtract(self, elem, scalar):
        '''Element-wise subraction of a scalar from a sparse dictionary representing a matrix.'''
        return elem - scalar

    @sparse_scalar_math
    def scalar_multiply(self, elem, scalar):
        '''Element-wise multiplication of a sparse dictionary representing a matrix by a scalar.'''
        return elem * scalar

    @sparse_scalar_math
    def scalar_divide(self, elem, scalar):
        '''Element-wise division of a sparse dictionary representing a matrix by a scalar.'''
        return elem / scalar

    @sparse_scalar_math
    def scalar_power(self, elem, scalar):
        '''Raises every element of a sparse dictionary representing a matrix by a scalar.'''
        return elem ** scalar

    @sparse_scalar_math
    def scalar_exponentiate(self, elem, scalar):
        '''Exponentiates a scalar by elements of a sparse dictionary representing a matrix.'''
        return scalar ** elem
    # END SPARSE SCALAR MATH #################################################################################################################

    # SPARSE FUNCTIONS #######################################################################################################################
    def sparse_functions(orig_func):
        '''Function called by decorators of specific miscellaneous functions.'''
        # CRAZY WAY TO GET FUNCTION NAME TO FEED INTO operation, SINCE @ IS PASSING THE FUNCTION IN HERE
        FUNCTIONS = ['exp', 'ln', 'sin', 'cos', 'tan', 'tanh', 'logit', 'relu', 'none']
        operation = [__ for __ in range(len(FUNCTIONS)) if FUNCTIONS[__] in str(orig_func)][0]
        operation = FUNCTIONS[operation]

        def core_sparse_functions(self):

            self.insufficient_dict_kwargs_1(f'{operation}')

            TO_DELETE_HOLDER = []
            _inner_len = self.inner_len1()
            for outer_key in self.DICT1:

                for inner_key in reversed(range(_inner_len)):    # MUST HIT ALL POSITIONS BECAUSE FOR MANY OF THESE FXNS f(0) != 0
                    result = orig_func(self, self.DICT1[outer_key].get(inner_key, 0))

                    if result != 0: self.DICT1[outer_key][inner_key] = result  # IF NON-ZERO, UPDATE (OR CREATE) KEY W NEW VALUE
                    # IF == 0 AND NOT ON LAST LOCATION, DEL LOCATION
                    if result == 0 and inner_key != _inner_len-1 and inner_key in self.DICT1[outer_key]:
                        TO_DELETE_HOLDER.append((outer_key, inner_key))
                    # IF == 0 AND ON LAST LOCATION, ENSURE PLACEHOLDER RULES ARE FOLLOWED
                    if result == 0 and inner_key == _inner_len-1: self.DICT1[outer_key][inner_key] = 0

            # Expection WHEN TRYING TO DELETE FROM DICTIONARY ON THE FLY, SO QUEUE DELETIONS UNTIL END
            for outer_key, inner_key in TO_DELETE_HOLDER:
                del self.DICT1[outer_key][inner_key]

            del TO_DELETE_HOLDER

            return self.DICT1

        return core_sparse_functions

    @sparse_functions
    def exp(self, elem):
        '''Exponentiation of e by elements of a sparse dictionary representing a matrix.'''
        return n.exp(elem)

    @sparse_functions
    def ln(self, elem):
        '''Element-wise natural logarithm of a sparse dictionary representing a matrix.'''
        return n.log(elem)

    @sparse_functions
    def sin(self, elem):
        '''Element-wise sine of a sparse dictionary representing a matrix.'''
        return n.sin(elem)

    @sparse_functions
    def cos(self, elem):
        '''Element-wise cosine of a sparse dictionary representing a matrix.'''
        return n.cos(elem)

    @sparse_functions
    def tan(self, elem):
        '''Element-wise tangent of a sparse dictionary representing a matrix.'''
        return n.tan(elem)

    @sparse_functions
    def tanh(self, elem):
        '''Element-wise hyperbolic tangent of a sparse dictionary representing a matrix.'''
        return n.tanh(elem)

    @sparse_functions
    def logit(self, elem):
        '''Element-wise logistic transformation of a sparse dictionary representing a matrix.'''
        return 1 / (1 + n.exp(-elem))

    @sparse_functions
    def relu(self, elem):
        '''Element-wise linear rectification of a sparse dictionary representing a matrix.'''
        return max(0, elem)

    @sparse_functions
    def none(self, elem):
        '''Element-wise linear pass-through of a sparse dictionary representing a matrix (no change).'''
        return elem
    # END SPARSE FUNCTIONS ##################################################################################################################

    # END GENERAL MATH ######################################################################################################################
    #########################################################################################################################################
    #########################################################################################################################################









if __name__ == '__main__':
    pass








