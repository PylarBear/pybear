
import pytest

pytest.skip(reason=f'24_09_07_11_41_00 need rewrite', allow_module_level=True)



import numpy as n, pandas as p
import time
from MLObjects.ObjectOrienter import MLObjectOrienter as mloo
from MLObjects.TestObjectCreators.SXNL import CreateSXNL as csxnl
from MLObjects import MLObject as mlo


# NOTES 4/23/23
# WHAT PROMPTED THIS TEST WAS THAT ObjectOrienter FOR MLRegressionCoreRunCode WAS TAKING LONGER THAN EXPECTED WHEN PASSED
# ALL OBJECTS, AND NONE OF THEM NEEDED TO BE REORIENTED.  FIRST FOUND THAT WHAT CAUSED IT TO BE SLOW WAS PASSING XTX TO
# ObjectOrienter.  THEN FOUND THAT DOING DATA VALIDATION IN ObjectOrienter WAS ALSO WEIGHING IT DOWN.  SO PASSING XTX AND
# DOING VALIDATION ON IT WAS THE CULPRIT. WHAT WAS TAKING 5.5 SECONDS WITH VALIDATION ON TAKES 0.02 SECONDS WITH
# VALIDATION OFF.



DATA = p.read_csv(r'C:\Users\Bill\Documents\WORK STUFF\RESUME\1 - OTHER\SRP\beer_reviews.csv',
                  nrows=10000,
                  header=0).dropna(axis=0)

DATA = DATA[DATA.keys()[[3, 4, 5, 7, 8, 9, 11]]]

TARGET = DATA['review_overall'].to_numpy()
TARGET.resize(1, len(TARGET))
TARGET_HEADER = [['review_overall']]

DATA = DATA.drop(columns=['review_overall'])

# KEEP THIS FOR SRNL
RAW_DATA = DATA.copy()
del DATA
RAW_DATA_HEADER = n.fromiter(RAW_DATA.keys(), dtype='<U50').reshape((1, -1))
RAW_DATA = RAW_DATA.to_numpy().transpose()

REF_VEC = n.fromiter(range(len(RAW_DATA[0])), dtype=int).reshape((1, -1))
REF_VEC_HEADER = [['ROW_ID']]

data_given_format = 'ARRAY'
data_given_orient = 'COLUMN'
target_given_format = 'ARRAY'
target_given_orient = 'COLUMN'
refvecs_given_format = 'ARRAY'
refvecs_given_orient = 'COLUMN'

data_return_format = 'ARRAY'
data_return_orient = 'COLUMN'
target_return_format = 'ARRAY'
target_return_orient = 'COLUMN'
refvecs_return_format = 'ARRAY'
refvecs_return_orient = 'COLUMN'

SXNLClass = csxnl.CreateSXNL(rows=None,
                             bypass_validation=True,
                             data_return_format=data_return_format,
                             data_return_orientation=data_return_orient,
                             DATA_OBJECT=RAW_DATA,
                             DATA_OBJECT_HEADER=RAW_DATA_HEADER,
                             DATA_FULL_SUPOBJ_OR_SINGLE_MDTYPES=None,
                             data_override_sup_obj=False,
                             data_given_orientation=data_given_orient,
                             data_columns=None,
                             DATA_BUILD_FROM_MOD_DTYPES=None,
                             DATA_NUMBER_OF_CATEGORIES=None,
                             DATA_MIN_VALUES=None,
                             DATA_MAX_VALUES=None,
                             DATA_SPARSITIES=None,
                             DATA_WORD_COUNT=None,
                             DATA_POOL_SIZE=None,
                             target_return_format=target_return_format,
                             target_return_orientation=target_return_orient,
                             TARGET_OBJECT=TARGET,
                             TARGET_OBJECT_HEADER=TARGET_HEADER,
                             TARGET_FULL_SUPOBJ_OR_SINGLE_MDTYPES=None,
                             target_type='FLOAT',
                             target_override_sup_obj=False,
                             target_given_orientation=target_given_orient,
                             target_sparsity=None,
                             target_build_from_mod_dtype=None,
                             target_min_value=None,
                             target_max_value=None,
                             target_number_of_categories=None,
                             refvecs_return_format=refvecs_return_format,
                             refvecs_return_orientation=refvecs_return_orient,
                             REFVECS_OBJECT=REF_VEC,
                             REFVECS_OBJECT_HEADER=REF_VEC_HEADER,
                             REFVECS_FULL_SUPOBJ_OR_SINGLE_MDTYPES=None,
                             REFVECS_BUILD_FROM_MOD_DTYPES=None,
                             refvecs_override_sup_obj=False,
                             refvecs_given_orientation=refvecs_given_orient,
                             refvecs_columns=None,
                             REFVECS_NUMBER_OF_CATEGORIES=None,
                             REFVECS_MIN_VALUES=None,
                             REFVECS_MAX_VALUES=None,
                             REFVECS_SPARSITIES=None,
                             REFVECS_WORD_COUNT=None,
                             REFVECS_POOL_SIZE=None
                             )

SRNL = SXNLClass.SXNL.copy()
SRNL_SUPOBJS = SXNLClass.SXNL_SUPPORT_OBJECTS.copy()

# expand #########################################################################################################
SXNLClass.expand_data(expand_as_sparse_dict=False, auto_drop_rightmost_column=False)

SWNL = SXNLClass.SXNL
SWNL_SUPOBJS = SXNLClass.SXNL_SUPPORT_OBJECTS

DATA = SWNL[0].copy()
DATA_TRANSPOSE = DATA.copy().transpose()
TARGET = SWNL[1].copy()
TARGET_TRANSPOSE = TARGET.copy().transpose()
TARGET_AS_LIST = TARGET.copy()

del SWNL

XTX = mlo.MLObject(DATA, data_return_orient, 'DATA').return_XTX(return_format='ARRAY')


trials = 10
bypass_validation = True

for WITH_XTX in [[], ['XTX']]:
    TIMES = []
    for trial in range(trials):
        t0 = time.time()
        OrienterClass = mloo.MLObjectOrienter(
                                                DATA=DATA,
                                                data_given_orientation=data_return_orient,
                                                data_return_orientation='AS_GIVEN',
                                                data_return_format='AS_GIVEN',

                                                DATA_TRANSPOSE=DATA_TRANSPOSE,
                                                data_transpose_given_orientation=data_return_orient,
                                                data_transpose_return_orientation='AS_GIVEN',
                                                data_transpose_return_format='AS_GIVEN',

                                                XTX=XTX, #None if WITH_XTX==[] else XTX,
                                                xtx_return_format='AS_GIVEN',

                                                XTX_INV=None,
                                                xtx_inv_return_format=None,

                                                target_is_multiclass=False,
                                                TARGET=TARGET,
                                                target_given_orientation=target_return_orient,
                                                target_return_orientation='AS_GIVEN',
                                                target_return_format='AS_GIVEN',

                                                TARGET_TRANSPOSE=TARGET_TRANSPOSE,
                                                target_transpose_given_orientation=target_return_orient,
                                                target_transpose_return_orientation='AS_GIVEN',
                                                target_transpose_return_format='AS_GIVEN',

                                                TARGET_AS_LIST=TARGET_AS_LIST,
                                                target_as_list_given_orientation=target_return_orient,
                                                target_as_list_return_orientation='AS_GIVEN',

                                                RETURN_OBJECTS=['DATA', 'DATA_TRANSPOSE', 'TARGET', 'TARGET_TRANSPOSE', 'TARGET_AS_LIST'] + WITH_XTX,

                                                bypass_validation=bypass_validation,
                                                calling_module='MLOOPerformanceTest',
                                                calling_fxn='tests'
        )

        data_return_format = OrienterClass.data_return_format
        data_return_orient = OrienterClass.data_return_orientation
        data_transpose_return_format = OrienterClass.data_transpose_return_format
        data_transpose_return_orient = OrienterClass.data_transpose_return_orientation
        target_return_format = OrienterClass.target_return_format
        target_return_orient = OrienterClass.target_return_orientation
        target_transpose_return_format = OrienterClass.target_transpose_return_format
        target_transpose_return_orient = OrienterClass.target_transpose_return_orientation
        DATA = OrienterClass.DATA
        DATA_TRANSPOSE = OrienterClass.DATA_TRANSPOSE
        TARGET = OrienterClass.TARGET
        TARGET_TRANSPOSE = OrienterClass.TARGET_TRANSPOSE
        TARGET_AS_LIST = OrienterClass.TARGET_AS_LIST
        is_list, is_dict = data_return_format == 'ARRAY', data_return_format == 'SPARSE_DICT'
        del OrienterClass

        tf = time.time() - t0
        print(f'\nTRIAL {trial+1}: {tf} sec')
        TIMES.append(tf)


    print(f'\n\033[92m')
    print(f'*** TESTS FINISHED ***')
    print(f'SHAPE = {n.shape(DATA)}')
    print(f'TIMES = {TIMES}')
    MIDDLE_TIMES = [_ for _ in TIMES if not _ in [min(TIMES), max(TIMES)]]
    print(f'AVERAGE TIME = {n.average(MIDDLE_TIMES)}, STDEV = {n.std(MIDDLE_TIMES, axis=0)}')
    print(f'\033[0m')
    [print(f'*'*120) for _ in range(3)]



































