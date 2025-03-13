import numpy as np
import sparse_dict as sparse_dict
import sys, inspect; from copy import deepcopy
from debug import get_module_name as gmn


# 3/20/23 BEAR DISPOSITION THIS.... DECIDE TO KEEP OR TRASH
# 11/26/22 THIS ONLY WORKS WITH SRNL/SWNL GENERATED THRU general_test_objects (OBJECTS ARE ATTRIBUTES OF A CLASS)

def this_module():
    return gmn.get_module_name(str(sys.modules[__name__]))


def _exception(words):
    raise Exception(f'\n*** {words} ***\n')


def obj_dict():
    return dict(((0,'DATA'), (2,'TARGET'), (4,'REF'), (6,'TEST')))


def merge_SXNL(SXNL1, SXNL2, OBJ_IDXS_TO_MERGE_AS_LIST):

    if len(SXNL1) != len(SXNL2):
        _exception(f'SUPER NUMPY 1 ({len(SXNL1)}) AND SUPER NUMPY 2 ({len(SXNL2)}) MUST HAVE THE SAME NUMBER OF OBJECTS AND HEADERS')

    # ENSURE RESPECTIVE OBJECT FORMATS ARE EQUAL
    OBJECT_FORMATS = []
    for obj_idx in OBJ_IDXS_TO_MERGE_AS_LIST:
        if obj_idx % 2 == 1:
            OBJECT_FORMATS.append('NA')   # HEADER ORIENTATIONS ARE IRRELEVANT

        if SXNL1[obj_idx].return_format != SXNL2[obj_idx].return_format:
            _exception(f'RESPECTIVE OBJECT FORMATS IN SUPER NUMPY LIST 1 AND 2 MUST MATCH')
        else:
            OBJECT_FORMATS.append(SXNL1[obj_idx].return_format)

    # ENSURE RESPECTIVE ORIENTATIONS ARE EQUAL
    OBJECT_ORIENTATIONS = []
    for obj_idx in OBJ_IDXS_TO_MERGE_AS_LIST:
        if obj_idx % 2 == 1:
            OBJECT_ORIENTATIONS.append('NA')   # HEADER ORIENTATIONS ARE IRRELEVANT
        if SXNL1[obj_idx].return_orientation != SXNL2[obj_idx].return_orientation:
            _exception(f'RESPECTIVE OBJECT ORIENTATIONS IN SUPER NUMPY LIST 1 AND 2 MUST MATCH')

    # TEST ROWS OF EACH RESPECTIVE OBJECT IS EQUAL
    for obj_idx in range(0, len(SXNL1), 2):
        len_1 = SXNL1[obj_idx].rows_out
        len_2 = SXNL2[obj_idx].rows_out
        if len_1 != len_2:
            _exception(f'NUMBER OF EXAMPLES (ROWS) IN SUPER NUMPYS 1 AND 2 MUST MATCH '
                       f'({obj_dict(obj_idx)}, 1 = {len_1}, 2 = {len_2})')
    del len_1, len_2

    # WOULD HAVE TO TEST COLUMNS OF EACH RESPECTIVE OBJECT IS EQUAL

    # WOULD HAVE TO TEST HEADERS AND SUPOBJS OF EACH RESPECTIVE OBJECT ARE EQUAL








