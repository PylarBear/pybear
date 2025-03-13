import numpy as np
from copy import deepcopy


# 3/20/23 BEAR DISPOSITION THIS MODULE, DECIDE TO KEEP OR TRASH


#  THINGS BEAR HAS TO DO FOR TEST OBJECT
# 1) FINISH expand_SRNL
# 8) MAKE CreateSRNL
# 9) MAKE merge_SXNL





def object_dict():
    return dict(((0,'DATA'),(2,'TARGET'),(4,'REF'),(6,'TEST')))


def exception_handle(words):
    raise Exception(f'\n*** {words} ***\n')


def expand_SRNL(SRNL_OBJECT,
                OBJECT_IDXS_TO_EXPAND_AS_LIST=[0,2],
                RETURN_FORMATS_AS_LIST=None,
                DROP_ONE_AS_LIST=None):

    '''Expands SRNL ndarray objects with categorical data.'''

    # arg/kwarg VALIDATION ##############################################################################################

    #  MUST BE EVEN NUMBER OF THINGS IN SRNL
    if len(SRNL_OBJECT) % 2 == 1:
        exception_handle(f'ODD NUMBER OF THINGS IN SRNL arg. ALL OBJECTS MUST HAVE A HEADER.')

    # STANDARDIZED HEADERS TO np.[[]] FORMAT
    for hdr_idx in range(1, len(SRNL_OBJECT), 2):
        SRNL_OBJECT[hdr_idx] = np.array(SRNL_OBJECT[hdr_idx], dtype='<U500').reshape((1,-1))

    # VERIFY IDXS_TO_EXPAND ARE EVEN NUMBERED, AND WITHIN RANGE OF SRNL_OBJECT



    if RETURN_FORMATS_AS_LIST is None: RETURN_FORMATS_AS_LIST = ['ARRAY' for _ in range(len(SRNL_OBJECT) // 2)]
    else:
        if len(RETURN_FORMATS_AS_LIST) != len(SRNL_OBJECT) // 2:
            exception_handle(f'RETURN_FORMATS_AS_LIST, IF NOT None, MUST HAVE AN ENTRY FOR EVERY OBJECT IN SRNL')

        try: RETURN_FORMATS_AS_LIST = list(map(str.upper, RETURN_FORMATS_AS_LIST))
        except: exception_handle(f'RETURN_FORMATS_AS_LIST kwarg MUST BE (list of "ARRAY" or "SPARSE_DICT") OR None')

        if False in map(lambda x: x in ('ÁRRAY', 'SPARSE_DICT'), RETURN_FORMATS_AS_LIST):
            exception_handle(f'RETURN_FORMATS_AS_LIST kwarg ENTRIES MUST BE "ARRAY" OR "SPARSE_DICT"')
    # END arg/kwarg VALIDATION ##############################################################################################


    # VERIFY HEADER LEN, EACH OBJECT KNOWS ITS OWN ORIENTATION ####################################################################
    for obj_idx, hdr_idx in [zip(list(range(0,len(SRNL_OBJECT), 2)), list(range(1,len(SRNL_OBJECT), 2)))]:
        if len(SRNL_OBJECT[obj_idx]) != len(SRNL_OBJECT[hdr_idx][0]) and len(SRNL_OBJECT[obj_idx][0]) != len(SRNL_OBJECT[hdr_idx][0]):
            exception_handle(f'{object_dict()[obj_idx]} OBJECT (idx {obj_idx}) AND ITS HEADER DO NOT MATCH len ON COLUMN NOR ROW AXES')
    # END GET ORIENTATIONS OF OBJECTS, VERIFY HEADER LEN #########################################################################


    if False in list(map(isinstance, DROP_ONE_AS_LIST, (bool for _ in range(len(SRNL_OBJECT)//2)))):
        exception_handle(f'DROP_ONE_AS_LIST kwarg MUST BE A BOOLEAN')


    SWNL_OBJECT = deepcopy(SRNL_OBJECT)

    for obj_idx in range(0,2,len(SWNL_OBJECT)):
        EXPANDED_DATA = deepcopy(SWNL_OBJECT[obj_idx])
        EXPANDED_DATA.expand()





    return SWNL_OBJECT



















