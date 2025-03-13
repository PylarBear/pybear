import numpy as np, pandas as pd
from MLObjects.TestObjectCreators.SXNL import CreateSXNL as csxnl
from MLObjects.PrintMLObject import SmallObjectPreview as sop






TC = csxnl.CreateSXNL(
                     rows=1000,
                     bypass_validation=False,
                     ##################################################################################################################
                     # DATA ############################################################################################################
                     data_return_format='ARRAY',
                     data_return_orientation='COLUMN',
                     DATA_OBJECT=None,
                     DATA_OBJECT_HEADER=None,
                     DATA_FULL_SUPOBJ_OR_SINGLE_MDTYPES=None,
                     data_override_sup_obj=False,
                     # CREATE FROM GIVEN ONLY ###############################################
                     data_given_orientation=None,
                     # END CREATE FROM GIVEN ONLY #############################################
                     # CREATE FROM SCRATCH_ONLY ################################
                     data_columns=5,
                     DATA_BUILD_FROM_MOD_DTYPES=['FLOAT', 'INT', 'BIN'],
                     DATA_NUMBER_OF_CATEGORIES=None,
                     DATA_MIN_VALUES=-10,
                     DATA_MAX_VALUES=10,
                     DATA_SPARSITIES=50,
                     DATA_WORD_COUNT=None,
                     DATA_POOL_SIZE=None,
                     # END DATA ###########################################################################################################
                     ##################################################################################################################

                     #################################################################################################################
                     # TARGET #########################################################################################################
                     target_return_format='ARRAY',
                     target_return_orientation='COLUMN',
                     TARGET_OBJECT=None,
                     TARGET_OBJECT_HEADER=None,
                     TARGET_FULL_SUPOBJ_OR_SINGLE_MDTYPES=None,
                     target_type='BINARY',  # MUST BE 'BINARY','FLOAT', OR 'SOFTMAX'
                     target_override_sup_obj=False,
                     target_given_orientation=None,
                     # END CORE TARGET_ARGS ########################################################
                     # FLOAT AND BINARY
                     target_sparsity=50,
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
                    REFVECS_OBJECT=None,
                    REFVECS_OBJECT_HEADER=None,
                    REFVECS_FULL_SUPOBJ_OR_SINGLE_MDTYPES=None,
                    REFVECS_BUILD_FROM_MOD_DTYPES='STR',
                    refvecs_override_sup_obj=False,
                    refvecs_given_orientation=None,
                    refvecs_columns=3,
                    REFVECS_NUMBER_OF_CATEGORIES=[5,10,20],
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



print(f'len(TC.TRAIN_DATA[0) = {len(TC.TRAIN_DATA[0])}')
print(f'len(TC.DEV_DATA[0]) = {len(TC.DEV_DATA[0])}')
print(f'len(TC.TEST_DATA[0]) = {len(TC.TEST_DATA[0])}')










