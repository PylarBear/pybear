import sys, inspect
from copy import deepcopy
import numpy as np
import sparse_dict as sd
from debug import get_module_name as gmn
from general_data_ops import get_shape as gs, get_dummies as gd
from data_validation import arg_kwarg_validater as akv
from ML_PACKAGE._data_validation import list_dict_validater as ldv
from MLObjects import MLObject as mlo


# (X OR Y)_OCCUR: ESSENTIALLY DO CATEGORY EXPANSION ON TARGET
# (X OR Y)_SUM: SUM ALL THE COLUMNS IN THE EXPANDED TARGET
# (X OR Y)_FREQ: DIVIDE Y_SUMS BY TOTAL NUMBER OF EXAMPLES, THEY MUST ADD TO 1


def occurrence(OBJECT, OBJECT_UNIQUES=None, return_as='ARRAY', bypass_validation=None, calling_module=None, calling_fxn=None):
    """OBJECT must be a single vector. Returns a grid of occurrences as [] = column for each unique in OBJECT."""

    this_module = calling_module if not calling_module is None else gmn.get_module_name(str(sys.modules[__name__]))
    fxn = inspect.stack()[0][3] if not calling_fxn is None else calling_fxn

    bypass_validation = akv.arg_kwarg_validater(bypass_validation, 'bypass_validation', [True, False, None],
                                                this_module, fxn, return_if_none=False)

    # GET FORMAT REGARDLESS OF bypass_validation
    object_format, OBJECT = ldv.list_dict_validater(OBJECT, 'OBJECT')

    if object_format == 'ARRAY' and len(OBJECT.shape) == 1: OBJECT = OBJECT.reshape((1, -1))
    elif object_format == 'SPARSE_DICT' and len(sd.shape_(OBJECT)) == 1: OBJECT = {0: OBJECT}

    if not bypass_validation:
        _ = gs.get_shape('OBJECT', OBJECT, 'COLUMN')   # JUST PASS COLUMN AS A DUMMY, ORIENT DOESNT MATTER FOR THE NEXT LINE
        if 1 not in _: raise Exception(f'*** OBJECT MUST BE A SINGLE VECTOR - IS A {object_format} SHAPED {_} ***')

        OBJECT_UNIQUES = ldv.list_dict_validater(OBJECT_UNIQUES, 'OBJECT_UNIQUES')[1]  # COULD STILL BE None

        return_as = akv.arg_kwarg_validater(return_as, 'return_as', ['ARRAY', 'SPARSE_DICT', 'AS_GIVEN'], this_module, fxn)

    if return_as=='AS_GIVEN': return_as = object_format

    # IF SPARSE_DICT, UNZIP AND reshape()
    if object_format == 'SPARSE_DICT': OBJECT = sd.unzip_to_ndarray_float64(OBJECT)[0].reshape((1,-1))
    elif object_format == 'ARRAY': OBJECT = OBJECT.reshape((1,-1))

    if np.array_equiv(OBJECT, OBJECT.astype(np.int32)): OBJECT = OBJECT.astype(np.int32)

    # MUST BE ARRAY AT THIS POINT SO DONT NEED sd.unique
    if OBJECT_UNIQUES is None:

        # OBJECT WAS UNZIPPED IF DICT AND ALWAYS RESHAPED((1,-1)) ABOVE
        UniquesClass = mlo.MLObject(OBJECT,
                                      'COLUMN',
                                      name='OBJECT',
                                      return_orientation='COLUMN',
                                      return_format='ARRAY',
                                      bypass_validation=bypass_validation,
                                      calling_module=this_module, calling_fxn=fxn)

        OBJECT_UNIQUES = UniquesClass.unique(0).reshape((1, -1))
        del UniquesClass

        OBJECT_UNIQUES = OBJECT_UNIQUES.reshape((1,-1))

    else: OBJECT_UNIQUES = np.array(OBJECT_UNIQUES).reshape((1,-1))   # CATCH UNIQUES WHEN PASSED AS KWARG

    return gd.get_dummies(OBJECT.astype(str),
                            OBJECT_HEADER=None,
                            given_orientation='COLUMN',   # COLUMN BECAUSE OF reshape ABOVE
                            IDXS=None,
                            UNIQUES=OBJECT_UNIQUES.astype(str),
                            return_orientation='COLUMN',
                            expand_as_sparse_dict=False if return_as=='ARRAY' else True,
                            auto_drop_rightmost_column=False,
                            append_ones=False,
                            bypass_validation=bypass_validation,
                            bypass_sum_check=True,
                            calling_module=this_module,
                            calling_fxn=fxn
    )[0]

    '''
    RELIC OCCURRENCES CODE
    
    if self.is_list:
        X_OCCUR = np.int8(self.DATA.astype(np.float64) == self.DATA_UNIQUES[x_idx].astype(np.float64))
    elif self.is_dict:
        if self.DATA_UNIQUES[x_idx] == 0:  # HAVE TO DO WHOLE THING IF LOOKING FOR ZEROS
            X_OCCUR = {0: {_: 1 for _ in range(self.rows) if self.DATA[0].get(_, 0) == self.DATA_UNIQUES[x_idx]}}
        else:  # BREAKING THIS OUT SAVES TIME OVER DOING EVERY ONE WITH THE ZERO METHOD
            X_OCCUR = {0: {k: 1 for k, v in self.DATA[0].items() if float(v) == float(self.DATA_UNIQUES[x_idx])}}
        if self.rows - 1 not in X_OCCUR[0]: X_OCCUR[0][int(self.rows - 1)] = 0  # OBSERVE PLACEHOLDER RULES
    '''



def sums(OCCURRENCES, return_as='ARRAY', calling_module=None, calling_fxn=None):
    "Returns a vector of the column sums for OCCURRENCES in the same format."

    this_module = calling_module if not calling_module is None else gmn.get_module_name(str(sys.modules[__name__]))
    fxn = inspect.stack()[0][3] if not calling_fxn is None else calling_fxn

    occ_format, OCCURRENCES = ldv.list_dict_validater(OCCURRENCES, OCCURRENCES)

    return_as = akv.arg_kwarg_validater(return_as, 'return_as', ['ARRAY','SPARSE_DICT','AS_GIVEN'], this_module, fxn)

    if return_as == 'AS_GIVEN': return_as = occ_format

    if occ_format=='ARRAY':
        SUMS = np.sum(OCCURRENCES, axis=1).reshape((1,-1))
        if return_as=='ARRAY': pass
        elif return_as=='SPARSE_DICT': SUMS = sd.zip_list_as_py_int(SUMS)
    elif occ_format=='SPARSE_DICT':
        SUMS = {0:{int(col_idx):sd.sum_({0:OCCURRENCES[col_idx]}) for col_idx in range(len(OCCURRENCES))}}
        if return_as=='ARRAY': SUMS = sd.unzip_to_ndarray_int32(SUMS)[0]
        elif return_as=='SPARSE_DICT': pass

    return SUMS    # RETURN AS DOUBLE


def frequencies(SUMS, return_as='ARRAY', calling_module=None, calling_fxn=None):

    this_module = calling_module if not calling_module is None else gmn.get_module_name(str(sys.modules[__name__]))
    fxn = inspect.stack()[0][3] if not calling_fxn is None else calling_fxn

    frq_format, SUMS = ldv.list_dict_validater(SUMS, 'SUMS')

    if not 1 in gs.get_shape('SUMS', SUMS, 'ROW'):
        raise Exception(f'*** SUMS MUST BE A ONE DIMENSIONAL VECTOR ***')

    return_as = akv.arg_kwarg_validater(return_as, 'return_as', ['ARRAY', 'SPARSE_DICT', 'AS_GIVEN'], this_module, fxn)

    if return_as == 'AS_GIVEN': return_as = frq_format

    if isinstance(SUMS, np.ndarray):
        SUMS = SUMS.reshape((1,-1))[0]
        FREQ = SUMS / np.sum(SUMS)
        FREQ = FREQ.reshape((1, -1))
        if return_as=='ARRAY': pass
        elif return_as=='SPARSE_DICT': FREQ = sd.zip_list_as_py_int(FREQ)
    elif isinstance(SUMS, dict):
        total_sum = sd.sum_(SUMS)
        FREQ = {0:{k:v / total_sum for k,v in SUMS[0].items()}}
        del total_sum
        if return_as=='ARRAY': FREQ = sd.unzip_to_ndarray_int32(FREQ)[0]
        elif return_as=='SPARSE_DICT': pass


    return FREQ



class MICrossEntropyObjects:
    """OBJECT must be a single vector. Returns occurrences, sums, and frequencies for each unique in OBJECT."""
    def __init__(self, OBJECT, UNIQUES=None, return_as='ARRAY', bypass_validation=None):

        this_module = gmn.get_module_name(str(sys.modules[__name__]))
        fxn = '__init__'

        bypass_validation = akv.arg_kwarg_validater(bypass_validation, 'bypass_validation', [True, False, None],
                                                    this_module, fxn, return_if_none=False)

        # GET FORMAT REGARDLESS OF bypass_validation
        object_format, OBJECT = ldv.list_dict_validater(OBJECT, 'OBJECT')

        if object_format == 'ARRAY' and len(OBJECT.shape)==1: OBJECT = OBJECT.reshape((1,-1))
        elif object_format == 'SPARSE_DICT' and len(sd.shape_(OBJECT)) == 1: OBJECT = {0: OBJECT}

        if not bypass_validation:
            _ = gs.get_shape('OBJECT', OBJECT, 'COLUMN')  # JUST PASS COLUMN AS A DUMMY, ORIENT DOESNT MATTER FOR THE NEXT LINE
            if 1 not in _: raise Exception(f'*** OBJECT MUST BE A SINGLE VECTOR - IS A {object_format} SHAPED {_} ***')

            UNIQUES = ldv.list_dict_validater(UNIQUES, 'UNIQUES')[1]  # COULD STILL BE None

            return_as = akv.arg_kwarg_validater(return_as, 'return_as', ['ARRAY', 'SPARSE_DICT', 'AS_GIVEN'], this_module, fxn)

        if return_as=='AS_GIVEN': return_as = object_format

        # ALWAYS BYPASS VALIDATION INSIDE occurrence SINCE WOULD HAVE BEEN HANDLED ABOVE
        self.OCCURRENCES = occurrence(OBJECT, OBJECT_UNIQUES=UNIQUES, return_as=return_as, bypass_validation=True,
                                        calling_module=this_module, calling_fxn=fxn)

        self.SUMS = sums(self.OCCURRENCES, return_as=return_as, calling_module=this_module, calling_fxn=fxn)

        self.FREQ = frequencies(self.SUMS, return_as=return_as, calling_module=this_module, calling_fxn=fxn)

        del this_module, fxn, bypass_validation, object_format, OBJECT, UNIQUES, return_as

















if __name__ == '__main__':
    
    # TEST MODULE

    # TEST & FUNCTIONS VERIFIED GOOD 5/4/23

    import time
    from general_sound import winlinsound as wls

    def array_dict_equiv(OBJ1, OBJ2, return_as):
        if return_as=='ARRAY': return np.array_equiv(OBJ1, OBJ2)
        elif return_as=='SPARSE_DICT': return sd.core_sparse_equiv(OBJ1, OBJ2)

    calling_module = gmn.get_module_name(str(sys.modules[__name__]))
    calling_fxn = 'tests'

    rows = 10
    UNIQUES_DATA_1 = [1,2,3,4]  # CAN NEVER BE DONE ON STR COLUMNS DUH!!
    TEST_DATA_1 = np.random.choice(UNIQUES_DATA_1, (1,rows), replace=True).astype(np.int32).reshape((1,-1))

    # GET UNIQUES AGAIN -- ALL OF THE UNIQUES MAY NOT HAVE BEEN USED IN GENERATING DATA
    UNIQUES_DATA_1 = np.unique(TEST_DATA_1).reshape((1,-1))[0]

    MASTER_DATA = [TEST_DATA_1]
    MASTER_DATA_NAME = ['TEST_DATA_1']
    MASTER_UNIQUES = [UNIQUES_DATA_1]
    MASTER_UNIQUES_GIVEN = [True, False]
    MASTER_GIVEN_FORMAT = ['ARRAY', 'SPARSE_DICT']
    MASTER_GIVEN_ORIENT = ['ROW', 'COLUMN']
    MASTER_RETURN_FORMAT = ['ARRAY', 'SPARSE_DICT', 'AS_GIVEN']
    MASTER_SINGLE_DOUBLE = ['SINGLE', 'DOUBLE']
    MASTER_VALIDATION = [True, False]

    total_trials = np.product(list(map(len, (MASTER_VALIDATION, MASTER_DATA, MASTER_UNIQUES_GIVEN, MASTER_GIVEN_ORIENT,
                                             MASTER_GIVEN_ORIENT, MASTER_SINGLE_DOUBLE, MASTER_RETURN_FORMAT))))
    ctr = 0
    for bypass_validation in MASTER_VALIDATION:
        for BASE_DATA, name, UNIQUES in zip(MASTER_DATA, MASTER_DATA_NAME, MASTER_UNIQUES):
            for uniques_given_t_f in MASTER_UNIQUES_GIVEN:
                if uniques_given_t_f is False: UNIQUES = None
                for given_format in MASTER_GIVEN_FORMAT:
                    for given_orient in MASTER_GIVEN_ORIENT:
                        for single_double in MASTER_SINGLE_DOUBLE:
                            if given_orient=='ROW': single_double='DOUBLE'
                            for return_as in MASTER_RETURN_FORMAT:
                                ctr += 1

                                if return_as == 'AS_GIVEN': return_as = given_format

                                print(f'\nRunning trial {ctr} of {total_trials}...')
                                print(f'DATA = {name} {single_double} {given_format} AS {given_orient}, '
                                      f'RETURN FORMAT = {return_as}, '
                                      f'VALIDATION={bypass_validation}')

                                #########################################################################################################
                                # BUILD GIVEN OBJECT ####################################################################################
                                # BASE DATA IS ALWAYS GIVEN AS COLUMN

                                GIVEN_DATA = deepcopy(BASE_DATA)
                                if given_orient == 'COLUMN': pass
                                elif given_orient == 'ROW': GIVEN_DATA = GIVEN_DATA.transpose()

                                if given_format == 'ARRAY': pass
                                elif given_format == 'SPARSE_DICT': GIVEN_DATA = sd.zip_list_as_py_int(GIVEN_DATA)

                                if single_double=='SINGLE': GIVEN_DATA = GIVEN_DATA[0]

                                # END BUILD GIVEN OBJECT ################################################################################
                                #########################################################################################################

                                ####################################################################################################
                                # GET EXPECTEDS ####################################################################################

                                if isinstance(GIVEN_DATA, dict): TEST_DATA = sd.unzip_to_ndarray_int32(GIVEN_DATA if sd.is_sparse_outer(GIVEN_DATA) else {0:GIVEN_DATA})[0]
                                else: TEST_DATA = GIVEN_DATA.copy()
                                if given_orient=='ROW': TEST_DATA = TEST_DATA.transpose()

                                EXP_UNIQUES = np.unique(TEST_DATA).reshape((1,-1))[0].astype('<U1').tolist()

                                EXP_OCCURRENCES = np.zeros((len(EXP_UNIQUES), rows)).astype(np.int8)
                                for idx, unq in enumerate(sorted(EXP_UNIQUES)):
                                    EXP_OCCURRENCES[idx] = np.int8(TEST_DATA.reshape((1,-1))[0].astype(str) == str(unq))

                                if return_as == 'SPARSE_DICT': EXP_OCCURRENCES = sd.zip_list_as_py_int(EXP_OCCURRENCES)

                                if return_as=='ARRAY': EXP_SUMS = np.sum(EXP_OCCURRENCES, axis=1).reshape((1,-1))
                                elif return_as=='SPARSE_DICT': EXP_SUMS = {0:{k:sd.sum_({0:v}) for k,v in EXP_OCCURRENCES.items()}}

                                if return_as=='ARRAY': EXP_FREQUENCIES = np.array(EXP_SUMS / np.sum(EXP_SUMS)).reshape((1,-1))
                                elif return_as=='SPARSE_DICT':
                                    total = sd.sum_(EXP_SUMS)
                                    EXP_FREQUENCIES = {0:{k:EXP_SUMS[0][k] / total for k in EXP_SUMS[0]}}


                                # END GET EXPECTEDS ################################################################################
                                ####################################################################################################

                                ####################################################################################################
                                # TEST INDIV FXNS ##################################################################################
                                ACT_OCCURRENCES = occurrence(GIVEN_DATA,
                                                             OBJECT_UNIQUES=UNIQUES,
                                                             return_as=return_as,
                                                             bypass_validation=bypass_validation,
                                                             calling_module=calling_module,
                                                             calling_fxn=calling_fxn)

                                ACT_SUMS = sums(ACT_OCCURRENCES,
                                                return_as=return_as,
                                                calling_module=calling_module,
                                                calling_fxn=calling_fxn)


                                ACT_FREQUENCIES = frequencies(ACT_SUMS,
                                                              return_as=return_as,
                                                              calling_module=calling_module,
                                                              calling_fxn=calling_fxn)

                                print(f'\033[91m')

                                if not array_dict_equiv(EXP_OCCURRENCES, ACT_OCCURRENCES, return_as):
                                    print(f'\nEXP_OCCURRENCES = ')
                                    print(EXP_OCCURRENCES)
                                    print(f'\nACT_OCCURRENCES = ')
                                    print(ACT_OCCURRENCES)
                                    raise Exception(f'*** INCONGRUENCE OF EXP & ACT OCCURRENCES ***')

                                if not array_dict_equiv(EXP_SUMS, ACT_SUMS, return_as):
                                    print(f'\nEXP_SUMS = ')
                                    print(EXP_SUMS)
                                    print(f'\nACT_SUMS = ')
                                    print(ACT_SUMS)
                                    raise Exception(f'*** INCONGRUENCE OF EXP & ACT SUMS ***')

                                if not array_dict_equiv(EXP_FREQUENCIES, ACT_FREQUENCIES, return_as):
                                    print(f'\nEXP_FREQUENCIES = ')
                                    print(EXP_FREQUENCIES)
                                    print(f'\nACT_FREQUENCIES = ')
                                    print(ACT_FREQUENCIES)
                                    raise Exception(f'*** INCONGRUENCE OF EXP & ACT FREQUENCIES ***')

                                print(f'\033[0m')
                                
                                # TEST INDIV FXNS ##################################################################################
                                ####################################################################################################
                                
                                ####################################################################################################
                                # TEST MICEO #######################################################################################

                                TestMICEO = MICrossEntropyObjects(GIVEN_DATA, UNIQUES=UNIQUES, return_as=return_as,
                                                                  bypass_validation=bypass_validation)
                                
                                ACT_OCCURRENCES = TestMICEO.OCCURRENCES
                                ACT_SUMS = TestMICEO.SUMS
                                ACT_FREQUENCIES = TestMICEO.FREQ

                                print(f'\033[91m')

                                if not array_dict_equiv(EXP_OCCURRENCES, ACT_OCCURRENCES, return_as):
                                    print(f'\nEXP_OCCURRENCES = ')
                                    print(EXP_OCCURRENCES)
                                    print(f'\nMICEO_ACT_OCCURRENCES = ')
                                    print(ACT_OCCURRENCES)
                                    raise Exception(f'*** INCONGRUENCE OF EXP & MICEO ACT OCCURRENCES ***')

                                if not array_dict_equiv(EXP_SUMS, ACT_SUMS, return_as):
                                    print(f'\nEXP_SUMS = ')
                                    print(EXP_SUMS)
                                    print(f'\nMICEO_ACT_SUMS = ')
                                    print(ACT_SUMS)
                                    raise Exception(f'*** INCONGRUENCE OF EXP & MICEO ACT SUMS ***')

                                if not array_dict_equiv(EXP_FREQUENCIES, ACT_FREQUENCIES, return_as):
                                    print(f'\nEXP_FREQUENCIES = ')
                                    print(EXP_FREQUENCIES)
                                    print(f'\nMICEO ACT_FREQUENCIES = ')
                                    print(ACT_FREQUENCIES)
                                    raise Exception(f'*** INCONGRUENCE OF EXP & MICEO ACT FREQUENCIES ***')

                                print(f'\033[0m')
                                
                                # END TEST MICEO ###################################################################################
                                ####################################################################################################
                                
                                

    print(f'\033[92m*** ALL TESTS PASSED ***\033[0m')
    for _ in range(3): wls.winlinsound(888, 500); time.sleep(1)



































