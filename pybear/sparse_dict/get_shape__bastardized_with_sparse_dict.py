import sys, inspect
import numpy as np, sparse_dict as sd
from debug import get_module_name as gmn

# ALWAYS RETURNS (rows, columns) REGARDLESS OF ORIENTATION - CONTRAST W NP THAT ALWAYS RETURNS (NUMBER OF INNER, LEN OF INNER)

def get_shape(name, OBJECT, given_orientation):
    """Return shape in numpy format for list-types and sparse dicts, except (x,y) x is always rows and y is always columns w-r-t to given orientation."""
    _module, _fxn = gmn.get_module_name(str(sys.modules[__name__])), inspect.stack()[0][3]
    def _exception(words): raise Exception(f'{_module}.{_fxn}() >>> {words}')

    if name is None or not isinstance(name, str): _exception(f'name MUST BE PASSED AND MUST BE A str')

    if OBJECT is None: del _exception; return ()

    # if not OBJECT is None
    is_list = isinstance(OBJECT, (np.ndarray, list, tuple))
    is_dict = isinstance(OBJECT, dict)

    # IF OBJECT IS PASSED AS SINGLE [] OR {}, DONT WORRY ABOUT ORIENTATION, ( CANT RETURN (,x) ONLY (x,) )
    if is_list: _shape = np.array(OBJECT).shape
    elif is_dict: _shape = sd.shape_(OBJECT)
    else: raise Exception(f'PASSED {name} OBJECT IS INVALID TYPE {type(OBJECT)}.')
    if len(_shape)==1: return _shape

    # CAN ONLY GET HERE IF len(_shape) >= 2
    if given_orientation is None: _exception(f'given_orientation MUST BE PASSED, CANNOT BE None.')
    if not given_orientation in ['ROW', 'COLUMN']: _exception(f'INVALID {name} given_orientation "{given_orientation}", MUST BE "ROW" OR "COLUMN".')

    #if given_orientation == 'ROW': _rows, _columns = _shape    _shape STAYS AS _shape
    if given_orientation == 'COLUMN': _shape = _shape[1], _shape[0]       # _shape GETS REVERSED

    del _exception; return _shape










if __name__ == '__main__':

    pass
























