import numpy as np, pandas as pd
from MLObjects.TestObjectCreators.SXNL import CreateSXNL as csxnl
from MLObjects.PrintMLObject import SmallObjectPreview as sop


DATA = pd.read_csv(r'C:\Users\Bill\Documents\WORK STUFF\RESUME\1 - OTHER\SRP\beer_reviews.csv',
                  nrows=500000,
                  header=0).dropna(axis=0)

DATA = DATA[DATA.keys()[[3, 4, 5, 7, 8, 9, 11]]]

# TAKE OUT ANY LEVELS IN BEER STYLE THAT HAVE LESS THAN 10 OCCURRENCES
value_counts = DATA['beer_style'].value_counts()
DATA = DATA[DATA['beer_style'].isin(value_counts.index[value_counts >= 10])]
del value_counts

TARGET = DATA['review_overall'].to_numpy()
TARGET.resize(1, len(TARGET))
TARGET_HEADER = [['review_overall']]

DATA = DATA.drop(columns=['review_overall'])

# KEEP THIS FOR SRNL
RAW_DATA = DATA.copy()
RAW_DATA_HEADER = np.fromiter(RAW_DATA.keys(), dtype='<U50').reshape((1, -1))
RAW_DATA = RAW_DATA.to_numpy().transpose()

# KEEP THIS FOR SWNL
DATA = pd.get_dummies(DATA, columns=['beer_style'], prefix='', prefix_sep='')
DATA_HEADER = np.fromiter(DATA.keys(), dtype='<U50').reshape((1, -1))
DATA = DATA.to_numpy().transpose()

REF_VEC = np.fromiter(range(len(DATA[0])), dtype=int).reshape((1, -1))
REF_VEC_HEADER = [['ROW_ID']]










TC = csxnl.CreateSXNL(
                     rows=None,
                     bypass_validation=False,
                     ##################################################################################################################
                     # DATA ############################################################################################################
                     data_return_format='ARRAY',
                     data_return_orientation='COLUMN',
                     DATA_OBJECT=RAW_DATA,
                     DATA_OBJECT_HEADER=RAW_DATA_HEADER,
                     DATA_FULL_SUPOBJ_OR_SINGLE_MDTYPES=None,
                     data_override_sup_obj=False,
                     # CREATE FROM GIVEN ONLY ###############################################
                     data_given_orientation='COLUMN',
                     # END CREATE FROM GIVEN ONLY #############################################
                     # CREATE FROM SCRATCH_ONLY ################################
                     data_columns=None,
                     DATA_BUILD_FROM_MOD_DTYPES=None,
                     DATA_NUMBER_OF_CATEGORIES=None,
                     DATA_MIN_VALUES=None,
                     DATA_MAX_VALUES=None,
                     DATA_SPARSITIES=None,
                     DATA_WORD_COUNT=None,
                     DATA_POOL_SIZE=None,
                     # END DATA ###########################################################################################################
                     ##################################################################################################################

                     #################################################################################################################
                     # TARGET #########################################################################################################
                     target_return_format='ARRAY',
                     target_return_orientation='COLUMN',
                     TARGET_OBJECT=TARGET,
                     TARGET_OBJECT_HEADER=TARGET_HEADER,
                     TARGET_FULL_SUPOBJ_OR_SINGLE_MDTYPES=None,
                     target_type='FLOAT',  # MUST BE 'BINARY','FLOAT', OR 'SOFTMAX'
                     target_override_sup_obj=False,
                     target_given_orientation='COLUMN',
                     # END CORE TARGET_ARGS ########################################################
                     # FLOAT AND BINARY
                     target_sparsity=None,
                     # FLOAT ONLY
                     target_build_from_mod_dtype=None,  # COULD BE FLOAT OR INT
                     target_min_value=None,
                     target_max_value=None,
                     # SOFTMAX ONLY
                     target_number_of_categories=None,

                    # END TARGET ####################################################################################################
                    #################################################################################################################

                    #################################################################################################################
                    # REFVECS ########################################################################################################
                    refvecs_return_format='ARRAY',
                    refvecs_return_orientation='COLUMN',
                    REFVECS_OBJECT=REF_VEC,
                    REFVECS_OBJECT_HEADER=REF_VEC_HEADER,
                    REFVECS_FULL_SUPOBJ_OR_SINGLE_MDTYPES=None,
                    REFVECS_BUILD_FROM_MOD_DTYPES=None,
                    refvecs_override_sup_obj=False,
                    refvecs_given_orientation='COLUMN',
                    refvecs_columns=None,
                    REFVECS_NUMBER_OF_CATEGORIES=None,
                    REFVECS_MIN_VALUES=None,
                    REFVECS_MAX_VALUES=None,
                    REFVECS_SPARSITIES=None,
                    REFVECS_WORD_COUNT=None,
                    REFVECS_POOL_SIZE=None
                    # END REFVECS ########################################################################################################
                    #################################################################################################################
                    )











NAMES = (f'DATA', f'TARGET', f'REFVECS') #, f'DATA_SUPPORT_OBJECT', f'TARGET_SUPPORT_OBJECT', f'REFVECS_SUPPORT_OBJECT')
OBJS = TC.SXNL.copy()
SUPOBJS = TC.SXNL_SUPPORT_OBJECTS.copy()

for name, _OBJ, _SUP_OBJS in zip(NAMES, OBJS, SUPOBJS):
    print(f'{name}:')
    sop.GeneralSmallObjectPreview(_OBJ, 'COLUMN', _SUP_OBJS, 'MODIFIEDDATATYPES')
    print(f'\n\n')

TC.expand_data(expand_as_sparse_dict=True)

print(f'*'*400)
print(f'AFTER EXPAND:\n')

NAMES = (f'DATA', f'TARGET', f'REFVECS') #, f'DATA_SUPPORT_OBJECT', f'TARGET_SUPPORT_OBJECT', f'REFVECS_SUPPORT_OBJECT')
OBJS = TC.SXNL
SUPOBJS = TC.SXNL_SUPPORT_OBJECTS


for name, _OBJ, _SUP_OBJS in zip(NAMES, OBJS, SUPOBJS):
    print(f'{name}:')
    sop.GeneralSmallObjectPreview(_OBJ, 'COLUMN', _SUP_OBJS, 'MODIFIEDDATATYPES')
    print(f'\n\n')



TC.train_dev_test_split()

print(f'*'*400)
print(f'AFTER SPLIT:\n')

NAMES = (f'TRAIN_DATA', f'TRAIN_TARGET', f'TRAIN_REFVECS')
OBJS = (TC.TRAIN_SWNL[0], TC.TRAIN_SWNL[1], TC.TRAIN_SWNL[2])
SUPOBJS = TC.SXNL_SUPPORT_OBJECTS

for name, _OBJ, _SUP_OBJS in zip(NAMES, OBJS, SUPOBJS):
    print(f'{name}:')
    # [print(f'{_}: {str(_OBJ[_])[:100]}') for _ in _OBJ]

    sop.GeneralSmallObjectPreview(_OBJ, 'COLUMN', _SUP_OBJS, 'MODIFIEDDATATYPES')
    print(f'\n\n')


NAMES = (f'DEV_DATA', f'DEV_TARGET', f'DEV_REFVECS')
OBJS = (TC.DEV_SWNL[0], TC.DEV_SWNL[1], TC.DEV_SWNL[2])
SUPOBJS = TC.SXNL_SUPPORT_OBJECTS

for name, _OBJ, _SUP_OBJS in zip(NAMES, OBJS, SUPOBJS):
    print(f'{name}:')
    sop.GeneralSmallObjectPreview(_OBJ, 'COLUMN', _SUP_OBJS, 'MODIFIEDDATATYPES')
    print(f'\n\n')


NAMES = (f'TEST_DATA', f'TEST_TARGET', f'TEST_REFVECS')
OBJS = (TC.TEST_SWNL[0], TC.TEST_SWNL[1], TC.TEST_SWNL[2])
SUPOBJS = TC.SXNL_SUPPORT_OBJECTS

for name, _OBJ, _SUP_OBJS in zip(NAMES, OBJS, SUPOBJS):
    print(f'{name}:')
    sop.GeneralSmallObjectPreview(_OBJ, 'COLUMN', _SUP_OBJS, 'MODIFIEDDATATYPES')
    print(f'\n\n')







