import sys

from MLObjects import MLObject as mlo
import numpy as np, sparse_dict as sd
from debug import get_module_name as gmn
from data_validation import validate_user_input as vui

#



_rows, _cols = 100, 50

BASE_DATA = np.random.randint(1,10, (_rows, _cols))
_orient = 'COLUMN'

GIVEN_DATA = BASE_DATA.copy()

if vui.validate_user_str(f'Run as row(r) or column(c) > ', 'RC') == 'C':
    GIVEN_DATA = GIVEN_DATA.copy()
    _orient = 'ROW'

if vui.validate_user_str(f'Run as sparse dict(s) or array(a) > ', 'AS') == 'S':
    GIVEN_DATA = sd.zip_list_as_py_int(GIVEN_DATA)


TestClass = mlo.MLObject(
                         GIVEN_DATA,
                         'ROW',
                         name='DATA',
                         return_orientation='AS_GIVEN',
                         return_format='AS_GIVEN',
                         bypass_validation=False,
                         calling_module=gmn.get_module_name(str(sys.modules[__name__])),
                         calling_fxn='test'
)

TestClass.intercept_manager(DATA_FULL_SUPOBJ_OR_HEADER=None, intcpt_col_idx=None)




print(f'\nCONTEXT = {TestClass.CONTEXT}')
print(f'intcpt_col_idx = {TestClass.intcpt_col_idx}')





















































