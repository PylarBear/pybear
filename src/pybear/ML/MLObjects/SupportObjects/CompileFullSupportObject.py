import sys, inspect
import numpy as np
from MLObjects.SupportObjects import master_support_object_dict as msod
from ML_PACKAGE._data_validation import list_dict_validater as ldv
from MLObjects.SupportObjects import diagnose_support_object_format as dsof
from debug import get_module_name as gmn


class CompileFullSupportObject:
    '''Compiles given supobjs into a full supobj, overwriting respective contents of FULL_SUPPORT_OBJECT if given.
        Access output as CompileFullSupportObject.SUPPORT_OBJECT or thru .build() method.'''
    def __init__(self,
                 FULL_SUPPORT_OBJECT=None,
                 HEADER=None,
                 VALIDATED_DATATYPES=None,
                 MODIFIED_DATATYPES=None,
                 FILTERING=None,
                 MIN_CUTOFF=None,
                 USE_OTHER=None,
                 START_LAG=None,
                 END_LAG=None,
                 SCALING=None
                 ):

        self.this_module = gmn.get_module_name(str(sys.modules[__name__]))
        fxn = '__init__'

        NAMES = tuple(msod.QUICK_POSN_DICT().keys())

        if HEADER is None: pass
        elif isinstance(HEADER, (np.ndarray, list, tuple)): HEADER = np.array(HEADER).reshape((1,-1))[0]
        else: self._exception(f'HEADER MUST BE PASSED AS A LIST-TYPE OR None', fxn=fxn)

        SUP_OBJS = (HEADER, VALIDATED_DATATYPES, MODIFIED_DATATYPES, FILTERING, MIN_CUTOFF,
                    USE_OTHER, START_LAG, END_LAG, SCALING)

        for _object in SUP_OBJS:
            if not _object is None and not isinstance(_object, (np.ndarray, list, tuple)):
                self._exception(f'ONE OF THE PASSED INDIVIDUAL SUPPORT OBJECTS IS NOT A LIST-TYPE', fxn=fxn)


        # BUILD A DICT OF ACTIVE SUPOBJS
        ACTV_OBJ_DICT = {name:obj for name,obj in zip(NAMES,SUP_OBJS) if not obj is None}

        # VALIDATE GIVEN LENS ARE ALL EQUAL
        _LENS = tuple(map(len, tuple(ACTV_OBJ_DICT.values())))
        if not len(_LENS)==0 and min(_LENS)!=max(_LENS):   # WILL EXCEPT IF TRY TO GET min OR max OF EMPTY
            print(f'GIVEN INDIVIDUAL SUPPORT OBJECTS MUST ALL BE THE SAME LENGTH:')
            self._exception(f', '.join([f'{name}({_len})' for name,_len in zip(ACTV_OBJ_DICT.keys(), _LENS)]), fxn=fxn)


        FULL_SUPPORT_OBJECT = ldv.list_dict_validater(FULL_SUPPORT_OBJECT, 'FULL_SUPPORT_OBJECT')[1]
        # DTYPE VALIDATED BY dsof

        if FULL_SUPPORT_OBJECT is None:
            FULL_SUPPORT_OBJECT = msod.build_empty_support_object(_LENS[0])
        elif not dsof.diagnose_support_object_format(FULL_SUPPORT_OBJECT):
            self._exception(f'FULL_SUPPORT_OBJECT, IF PASSED, MUST BE PASSED AS A FULL', fxn=fxn)
        else:
            if len(FULL_SUPPORT_OBJECT[0]) != _LENS[0]:
                self._exception(f'len FULL_SUPPORT_OBJECT DOES NOT EQUAL lens OF PASSED INDIV SUP OBJS', fxn=fxn)

        del NAMES, SUP_OBJS, _LENS    # EVERYTHING IS IN ACTV_OBJ_DICT NOW

        # IF FULL_SUPPORT_OBJECT PASSED & FULL, PRESERVE WHATEVER INFORMATION MAY BE IN IT AND ONLY
        # OVERWRITE W INDIV SUP OBJS GIVEN AS KWARG

        for name, VALUES in ACTV_OBJ_DICT.items():
            FULL_SUPPORT_OBJECT[msod.QUICK_POSN_DICT()[name]] = VALUES

        self.SUPPORT_OBJECTS = FULL_SUPPORT_OBJECT



    def _exception(self, words, fxn=None):
        fxn = f'.{fxn}()' if not fxn is None else inspect.stack()[0][3]
        raise Exception(f'{self.this_module}{fxn} >>> {words}')


    def build(self):
        return self.SUPPORT_OBJECTS




if __name__ == '__main__':
    pass




