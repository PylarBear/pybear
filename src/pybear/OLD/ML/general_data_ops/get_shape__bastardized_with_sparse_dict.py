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

    # TEST MODULE!

    NAMES = ['None'] + [f'{"NP_ARRAY" if _ < 5 else "SPARSE_DICT"}{_+1}' for _ in range(10)]
    OBJECTS = (None, [], [[],[]], [2,3,4], [[2,3,4]], [[1,2],[3,4]], {}, {0:{}, 1:{}}, {0:2, 1:3, 2:4}, {0:{0:2,1:3,2:4}},
               {0:{0:1,1:2}, 1:{0:3,1:4}})
    KEY = ( (), (0,), (2,0), (3,), (1,3), (2,2), (0,), (2,0), (3,), (1,3), (2,2) )

    for name, _OBJ, _key in zip(NAMES, OBJECTS, KEY):
        _shape = get_shape(name, _OBJ, 'ROW')   # given_orientation == 'ROW'
        if not np.array_equiv(_shape, _key):
            raise Exception(f'INCONGRUITY BETWEEN MEASURED SHAPE {_shape} AND ANSWER KEY {_key} FOR {_OBJ}')
        else: print(f'\033[92m*** {_OBJ} = {_key}   PASSED! ***\033[0m')

    print(f'\n\033[92m*** TESTS COMPLETE.  ALL PASSED. ***\033[0m')
























