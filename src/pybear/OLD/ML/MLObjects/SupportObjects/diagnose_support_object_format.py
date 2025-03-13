import sys, inspect
from MLObjects.SupportObjects import master_support_object_dict as msod
from ML_PACKAGE._data_validation import list_dict_validater as ldv
from debug import get_module_name as gmn


def _exception(words, fxn=None):
    _module = gmn.get_module_name(str(sys.modules[__name__]))
    fxn = f'.{fxn}()' if not fxn is None else inspect.stack()[0][3]
    raise Exception(f'{_module}{fxn} >>> {words}')


def diagnose_support_object_format(SUPPORT_OBJECT_AS_SINGLE_OR_FULL):
        '''Validates passed object and returns True if object is full, False if single, exception otherwise.'''
        fxn = inspect.stack()[0][3]

        obj_format, SUPPORT_OBJECT_AS_SINGLE_OR_FULL = ldv.list_dict_validater(SUPPORT_OBJECT_AS_SINGLE_OR_FULL,
                                                                           f'SUPPORT_OBJECT_AS_SINGLE_OR_FULL')
        if obj_format != 'ARRAY': _exception(f'SUPPORT_OBJECT_AS_SINGLE_OR_FULL MUST BE PASSED AS LIST-TYPE',fxn=fxn)
        del obj_format
        # SOASOF WOULD BE RETURNED AS [[]] IF WENT IN AS []....... [[],[],...] IF WENT IN AS [[],[],...]
        _ = SUPPORT_OBJECT_AS_SINGLE_OR_FULL
        _full_len = len(msod.master_support_object_dict())  # NUMBER OF ROWS IN A FULL SUPOBJ

        if len(_)!=1 and len(_)!=_full_len:
            _exception(f'SUPPORT OBJECT MUST CONTAIN 1 SUPPORT OBJECT OR ALL {_full_len} OF THEM', fxn=fxn)

        return True if len(_)==_full_len else False









