import sys, inspect, warnings
import numpy as np
from MLObjects.SupportObjects import master_support_object_dict as msod, validate_full_support_object as vfso
from debug import get_module_name as gmn
from data_validation import arg_kwarg_validater as akv


class FullSupObjSplitter:
    '''Access individual support objects as attribute of class or their respective method.'''
    def __init__(self, FULL_SUPPORT_OBJECT, bypass_validation=False):

        self.this_module = gmn.get_module_name(str(sys.modules[__name__]))
        fxn = '__init__'

        bypass_validation = akv.arg_kwarg_validater(bypass_validation, 'bypass_validation', [True, False, None],
                                                    self.this_module, fxn, return_if_none=False)

        if not bypass_validation:

            # RUNNING INTO DIFFICULTY USING vfso TO VALIDATE FULL_SUPPORT_OBJECT WHEN OBJECT IS NOT PASSED
            # vfso.validate_full_support_object(FULL_SUPPORT_OBJECT, OBJECT=None, object_given_orientation=None,
            #                                   OBJECT_HEADER=None, allow_override=False)
            # 3/13/23 JUST VALIDATE FULL_SUPPORT_OBJECT IS NOT RAGGED

            _LENS = list(map(len, FULL_SUPPORT_OBJECT))
            if min(_LENS) != max(_LENS):
                NAMES, POSNS = np.fromiter(msod.QUICK_POSN_DICT().keys(), dtype='<U20'), np.fromiter(msod.QUICK_POSN_DICT().values(), dtype=np.int8)
                POSN_ARGSORT = np.argsort(POSNS)
                NAMES = NAMES[..., POSN_ARGSORT]   # GET THE CORRECT NAMES TO GO WITH THE CORRECT LENS
                del POSN_ARGSORT
                self._exception(f'FULL_SUPPORT_OBJECT IS RAGGED:\n' + f', '.join([f'{NAMES[_]} ({_LENS[_]})'
                                                                      for _ in range(len(FULL_SUPPORT_OBJECT))]), fxn=fxn)
            del _LENS


        self.SUPPORT_OBJECTS = FULL_SUPPORT_OBJECT
        self.OBJECT_HEADER = FULL_SUPPORT_OBJECT[msod.QUICK_POSN_DICT()['HEADER']].reshape((1,-1))
        self.VALIDATED_DATATYPES = FULL_SUPPORT_OBJECT[msod.QUICK_POSN_DICT()['VALIDATEDDATATYPES']]
        self.MODIFIED_DATATYPES = FULL_SUPPORT_OBJECT[msod.QUICK_POSN_DICT()['MODIFIEDDATATYPES']]
        self.FILTERING = FULL_SUPPORT_OBJECT[msod.QUICK_POSN_DICT()['FILTERING']]
        self.MIN_CUTOFFS = FULL_SUPPORT_OBJECT[msod.QUICK_POSN_DICT()['MINCUTOFFS']]
        self.USE_OTHER = FULL_SUPPORT_OBJECT[msod.QUICK_POSN_DICT()['USEOTHER']]
        self.START_LAG = FULL_SUPPORT_OBJECT[msod.QUICK_POSN_DICT()['STARTLAG']]
        self.END_LAG = FULL_SUPPORT_OBJECT[msod.QUICK_POSN_DICT()['ENDLAG']]
        self.SCALING = FULL_SUPPORT_OBJECT[msod.QUICK_POSN_DICT()['SCALING']]
        # 3/13/23 IF PASSED SUPOBJ IS FOR EXPANDED DATA, self.KEEP IN THE CALLING PLACE WOULD BE OVERWRIT WITH THIS self.KEEP
        # self.KEEP = FULL_SUPPORT_OBJECT[msod.QUICK_POSN_DICT()['HEADER']][0]


    def _exception(self, words, fxn=None):
        fxn = f'.{fxn}()' if not fxn is None else inspect.stack()[0][3]
        raise Exception(f'{self.this_module}{fxn} >>> {words}')

    def object_header(self):
        return self.OBJECT_HEADER

    def validated_datatypes(self):
        return self.VALIDATED_DATATYPES

    def modified_datatypes(self):
        return self.MODIFIED_DATATYPES

    def filtering(self):
        return self.FILTERING

    def min_cutoffs(self):
        return self.MIN_CUTOFFS

    def use_other(self):
        return self.USE_OTHER

    def start_lag(self):
        return self.START_LAG

    def end_lag(self):
        return self.END_LAG

    def scaling(self):
        return self.SCALING

    # def keep(self):    # SEE NOTE AT END OF init
    #     return self.KEEP
















