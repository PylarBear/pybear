
import pytest

pytest.skip(reason=f'24_09_07_11_43_00 need rewrite', allow_module_level=True)


import numpy as np
import sparse_dict as sd
from MLObjects.SupportObjects import build_random_full_support_object as brfso, ApexSupportObjectHandling as soh
from general_text import alphanumeric_str
from general_data_ops import create_random_sparse_numpy as crsn


MASTER_GIVEN_FORMAT = ['ARRAY', 'SPARSE_DICT']
MASTER_GIVEN_ORIENTATION = ['ROW', 'COLUMN']
alpha_str = alphabet_str

total_trials = np.product(list(map(len, (MASTER_GIVEN_FORMAT, MASTER_GIVEN_ORIENTATION))))

_columns = 10
_rows = 20

RAW_DUM_OBJECT = crsn.create_random_sparse_numpy(0, 10, (_columns, _rows), _sparsity=70, _dtype=np.int32)

ctr = 0
for given_format in MASTER_GIVEN_FORMAT:
    for given_orientation in MASTER_GIVEN_ORIENTATION:
        ctr += 1
        print(f'*'*100)
        print(f'Running trial {ctr} of {total_trials}...')

        DUM_OBJECT = RAW_DUM_OBJECT.copy()


        if given_orientation == 'COLUMN': pass
        elif given_orientation == 'ROW':
            DUM_OBJECT = DUM_OBJECT.transpose()

        if given_format == 'SPARSE_DICT':
            DUM_OBJECT = sd.zip_list_as_py_float(DUM_OBJECT)


        DUM_SUPPORT = brfso.build_random_full_support_object(_columns)

        TestClass = soh.Scaling(
                 OBJECT=DUM_OBJECT,
                 object_given_orientation=given_orientation,
                 SUPPORT_OBJECT=DUM_SUPPORT,
                 columns=_columns,
                 prompt_to_edit_given_support_object=False,
                 return_support_object_as_full_array=True,
                 bypass_validation=False,
                 calling_module=f'SupportObjectHandling_small_obj_preview__test',
                 calling_fxn='TestClass'
                 )

        print(f'GIVEN OBJECT (orientation = {given_orientation}):')
        if given_format == 'ARRAY':
            print(DUM_OBJECT)
        elif given_format == 'SPARSE_DICT':
            [print(f'{_}: {DUM_OBJECT[_]}') for _ in range(10)]

        print(f'\nTEST OUTPUT:')
        TestClass.small_preview_object()

        # TEST FOR FULL RETURN AS
        # TestClass.small_preview_object()
        # TEST FOR SINGLE RETURN AS
        # TestClass.small_preview_object(idx=?)

















