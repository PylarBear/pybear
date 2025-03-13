import inspect
import numpy as np
from ML_PACKAGE._data_validation import list_dict_validater as ldv

# CENTRALIZED PLACE FOR MANAGING VDTYPES, MDTYPES, SUPOBJ LAYOUT, AND OTHER ALLOWED #####################################
'''Configure available text analytics options here (number options should not be changed) for easy menu /
    function accessibility management.'''

def empty_value():
    return 'None'

def val_text_dtypes():
    return {'S': 'STR', 'L':'BOOL'}

def val_num_dtypes():
    return {'I': 'INT', 'F': 'FLOAT', 'B': 'BIN'}

def val_reverse_lookup():
    return {'NNLM50': 'STR', 'SPLIT_STR': 'STR', 'STR': 'STR', 'FLOAT': 'FLOAT', 'BIN': 'BIN', 'INT': 'INT'}


def mod_text_dtypes():
    return {'S': 'STR', 'N': 'NNLM50', 'P': 'SPLIT_STR'}  # , '4':'TXT4', '5':'TXT5'       DONT USE I,B,F

def mod_num_dtypes():
    return {'I': 'INT', 'F': 'FLOAT', 'B': 'BIN'}


def master_support_object_dict():
    # MUST init ApexSupportObjectHandle TO GET VAL_ALLOWED

    VAL_ALLOWED_LIST = list(val_num_dtypes().values()) + list(val_text_dtypes().values())
    MOD_ALLOWED_LIST = list(mod_num_dtypes().values()) + list(mod_text_dtypes().values())

    null_func = lambda x: x
    str_of_type = lambda x: str(type(x))

    return \
        {
            'HEADER':             {'position': 0, 'transform': str_of_type, 'allowed': str(str)                   },
            'VALIDATEDDATATYPES': {'position': 1, 'transform': null_func,   'allowed': ", ".join(VAL_ALLOWED_LIST)},
            'MODIFIEDDATATYPES':  {'position': 2, 'transform': null_func,   'allowed': ", ".join(MOD_ALLOWED_LIST)},
            'FILTERING':          {'position': 3, 'transform': str_of_type, 'allowed': str(list)                  },
            'MINCUTOFFS':         {'position': 4, 'transform': null_func,   'allowed': range(0, int(1e7))         },
            'USEOTHER':           {'position': 5, 'transform': null_func,   'allowed': ", ".join(['Y', 'N'])      },
            'STARTLAG':           {'position': 6, 'transform': null_func,   'allowed': range(int(-1e7), int(1e7)) },
            'ENDLAG':             {'position': 7, 'transform': null_func,   'allowed': range(int(-1e7), int(1e7)) },
            'SCALING':            {'position': 8, 'transform': str_of_type, 'allowed': str(str)                   }
        }


def QUICK_POSN_DICT():
    return {k:master_support_object_dict()[k]['position'] for k in master_support_object_dict()}


def is_empty_getter(supobj_idx=None, supobj_name=None, SUPOBJ=None, calling_module=None, calling_fxn=None):
    '''Returns True/False indication of whether a support object is filled with msod.empty_value().
        If passing single supobj, could pass, but dont need to, pass zero as idx.  If full, could pass either idx or name.'''

    fxn = inspect.stack()[0][3]
    if not calling_module is None and not isinstance(calling_module, str): raise Exception(f'calling_module MUST BE A str')
    calling_module = '' if calling_module is None else f'{calling_module}.'
    calling_fxn = fxn if calling_fxn is None else calling_fxn

    max_len = len(master_support_object_dict())

    def _exception(words): raise Exception(f'{calling_module}{calling_fxn}() >>> {words}')

    if SUPOBJ is None: _exception(f'MUST PASS A SINGLE OR FULL SUPPORT OBJECT')
    elif not SUPOBJ is None:
        _dtype, SUPOBJ = ldv.list_dict_validater(SUPOBJ, 'SUPOBJ')   # IF SUPOBJ WAS [] GOES TO [[]], IF WAS [[],[],...] STAYS AS WAS
        if not _dtype=='ARRAY': _exception(f'SUPOBJ MUST BE GIVEN AS LIST-TYPE')

        if len(SUPOBJ) not in [1,max_len]:
            _exception(f'SINGLE_SUPOBJ MUST BE A LIST-TYPE WITH 1 OR {max_len} ROWS')

        if len(SUPOBJ)==1:
            # ANYTHING PUT INTO supobj_idx AND supobj_name ARE IGNORED
            actv_idx = 0
            supobj_name = 'GIVEN SUPOBJ VECTOR'

        elif len(SUPOBJ)==max_len:

            if not supobj_idx is None and not supobj_name is None: _exception(f'DONT PASS BOTH supobj_idx AND supobj_name')

            if not supobj_idx is None:
                if not 'INT' in str(type(supobj_idx)).upper() or supobj_idx < 0 or supobj_idx > max_len - 1:
                    _exception(f'full_supobj_idx MUST BE AN INTEGER >= 0 AND <= {max_len - 1}')
                actv_idx = supobj_idx
                supobj_name = {master_support_object_dict()[k]['position']:k for k in master_support_object_dict()}[supobj_idx]

            elif not supobj_name is None:
                if not isinstance(supobj_name, str): _exception(f'supobj_name MUST BE A str')
                supobj_name = supobj_name.upper()
                if not supobj_name in QUICK_POSN_DICT():
                    _exception(f'supobj_name "{supobj_name}" IS NOT VALID. MUST BE IN {", ".join(list(QUICK_POSN_DICT().keys()))}.')
                actv_idx = QUICK_POSN_DICT()[supobj_name]

        # MUST HAVE actv_idx BY THIS POINT

        # 3/14/23 KNOWINGLY COPPING OUT AND ASSUMING NO ONE WOULD EVER PASS SINGLE FILTERING SUPOBJ TO THIS MODULE
        # FUTURE ME, IF YOU EVER NEED TO GET is_empty FOR A SINGLE FILTERING VECTOR, JUST MAP len TO IT AND min==max==0 MEANS EMPTY

        try: UNIQUES = np.unique(SUPOBJ[actv_idx])
        except:
            if TypeError: raise Exception(f'\n*** TypeError WHILE TRYING TO GET UNIQUES IN {supobj_name}. '
                f'CHECK THAT EACH ROW IN SUPPORT_OBJECT IS OF SAME dtype (i.e., CANNOT MIX INTS AND STRS WITHIN A ROW), '
                f'AND IS NOT None. ***\n')
            else: raise Exception (f'\n*** EXCEPTION OTHER THAN TypeError WHILE TRYING TO GET UNIQUES IN {supobj_name}. ***\n')

        expected_empty_value = [] if supobj_name == 'FILTERING' else empty_value()

        if len(UNIQUES) == 1 and UNIQUES[0] == expected_empty_value: del UNIQUES, supobj_idx, supobj_name, expected_empty_value; return True
        else: del UNIQUES, supobj_idx, supobj_name, expected_empty_value; return False



def build_empty_support_object(_columns):
    _ = np.full((len(master_support_object_dict()), _columns), empty_value(), dtype=object)
    for idx in range(_columns):   # FILTERING SLOTS MUST BE [] !
        _[QUICK_POSN_DICT()["FILTERING"]][idx] = []
    return _


def build_random_support_object(_columns):

    VAL_DTYPES = ['BIN', 'INT', 'FLOAT', 'STR']
    MOD_DTYPES = ['BIN', 'INT', 'FLOAT', 'STR', 'SPLIT_STR', 'NNLM50']
    STR_DTYPES = ['STR', 'SPLIT_STR', 'NNLM50']

    _a = 'BIN'
    _b = 'INT'
    _c = 'FLOAT'
    _d = 'STR'
    _e = 'SPLIT_STR'
    _f = 'NNLM50'

    MOD_REVERSE = {_f:_d, _e:_d, _d:_d, _c:_c, _b:_b, _a:_a}

    SUPPORT_OBJECT = np.empty((9, _columns), dtype=object)

    SUPPORT_OBJECT[0] = np.fromiter((f'COLUMN{_+1}' for _ in range(_columns)), dtype=object)
    SUPPORT_OBJECT[1] = np.random.choice(VAL_DTYPES, _columns, replace=True)
    SUPPORT_OBJECT[2] = np.fromiter((np.random.choice(STR_DTYPES,1)[0] if SUPPORT_OBJECT[1][_]=='STR' else SUPPORT_OBJECT[1][_] for _ in range(_columns)), dtype=object)
    for _ in range(_columns): SUPPORT_OBJECT[3][_] = []
    SUPPORT_OBJECT[4] = np.fromiter((np.random.choice([0,10,20], 1, p=[0.6,0.2,0.2])[0] if SUPPORT_OBJECT[1][_]=='STR' else 0 for _ in range(_columns)), dtype=object)
    SUPPORT_OBJECT[5] = np.fromiter((np.random.choice(['Y','N'], 1)[0] if SUPPORT_OBJECT[4][_]>0 else 'N' for _ in range(_columns)), dtype=object)
    SUPPORT_OBJECT[6] = np.fromiter((0 for _ in range(_columns)), dtype=object)
    SUPPORT_OBJECT[7] = np.fromiter((0 for _ in range(_columns)), dtype=object)
    SUPPORT_OBJECT[8] = np.fromiter((f'' for _ in range(_columns)), dtype=object)

    return SUPPORT_OBJECT





