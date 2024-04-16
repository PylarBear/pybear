import numpy as np
import sys, warnings, time
import sparse_dict as sd
from data_validation import arg_kwarg_validater as akv
from ML_PACKAGE._data_validation import list_dict_validater as ldv
from debug import get_module_name as gmn
from MLObjects import MLObject as mlo, MLTargetTransposeObject as mltto, MLObjectSymmetric as mlos, \
    MLTargetObject as mlto


class MLObjectOrienter:
    '''Yield DATA, DATA_TRANSPOSE, XTX, XTX_INV, TARGET, TARGET_TRANSPOSE, TARGET_AS_LIST as attributes of the class.'''

    def __init__(self,
                 DATA=None,
                 data_given_orientation=None,
                 data_return_orientation='AS_GIVEN',
                 data_return_format='AS_GIVEN',

                 DATA_TRANSPOSE=None,
                 data_transpose_given_orientation=None,
                 data_transpose_return_orientation='AS_GIVEN',
                 data_transpose_return_format='AS_GIVEN',

                 XTX=None,
                 xtx_return_format='AS_GIVEN',   # XTX IS SYMMETRIC SO DONT NEED ORIENTATION

                 XTX_INV=None,
                 xtx_inv_return_format='AS_GIVEN',  # XTX IS SYMMETRIC SO DONT NEED ORIENTATION

                 target_is_multiclass=None,
                 TARGET=None,
                 target_given_orientation=None,
                 target_return_orientation='AS_GIVEN',
                 target_return_format='AS_GIVEN',

                 TARGET_TRANSPOSE=None,
                 target_transpose_given_orientation=None,
                 target_transpose_return_orientation='AS_GIVEN',
                 target_transpose_return_format='AS_GIVEN',

                 TARGET_AS_LIST=None,
                 target_as_list_given_orientation=None,
                 target_as_list_return_orientation='AS_GIVEN',

                 RETURN_OBJECTS=None,

                 bypass_validation=False,
                 calling_module=None,
                 calling_fxn=None):


        this_module = gmn.get_module_name(str(sys.modules[__name__]))
        fxn = '__init__'

        bypass_validation = akv.arg_kwarg_validater(bypass_validation, 'bypass_validation',
                                                    [True, False, None], this_module, fxn, return_if_none=False)

        # raise Exception template for this module
        def _exception(fxn, _text):
            raise Exception(f'\n*** {this_module}.{fxn}() ' +
                            [f'called by {calling_module}.{calling_fxn}() >>> ' if not calling_module is None
                                   and not calling_fxn is None else '>>> '][0] +
                            f'{_text} ***\n')

        # THIS MUST BE FIRST FOR ACCESS OF dtypes BY SHORT-CIRCUIT MODE
        # VERIFY GIVEN OBJECT IS LIST, DICT, OR None

        ''' OBJECTS THAT ARE VALIDATED FOR ARRAY/DICT/None IN THIS SECTION WILL BE EVALUATED FOR CONGRUENCY 
            LATER, IF APPLICABLE
            DATA, DATA_TRANSPOSE
            XTX, XTX_INV
            TARGET, TARGET_TRANSPOSE, TARGET_AS_LIST'''


        data_given_format, DATA = ldv.list_dict_validater(DATA, 'DATA')
        data_transpose_given_format, DATA_TRANSPOSE = ldv.list_dict_validater(DATA_TRANSPOSE, 'DATA_TRANSPOSE')
        xtx_given_format, XTX = ldv.list_dict_validater(XTX, 'XTX')
        xtx_inv_given_format, XTX_INV = ldv.list_dict_validater(XTX_INV, 'XTX_INV')
        target_given_format, TARGET = ldv.list_dict_validater(TARGET, 'TARGET')
        target_transpose_given_format, TARGET_TRANSPOSE = ldv.list_dict_validater(TARGET_TRANSPOSE, 'TARGET_TRANSPOSE')
        target_as_list_given_format, TARGET_AS_LIST = ldv.list_dict_validater(TARGET_AS_LIST, 'TARGET_AS_LIST')  # MUST BE ARRAY

        if target_as_list_given_format not in [None, 'ARRAY']: _exception(fxn, f'TARGET_AS_LIST MUST BE A LIST-TYPE!')

        while True:    # ONLY TO ALLOW break ON SHORT CIRCUIT TO BYPASS ALL THE HEAVY MACHINERY. SHOULD ONLY MAKE ONE PASS.

            self.DATA = None
            self.data_given_format = data_given_format
            self.data_current_format = None
            self.data_return_format = None
            self.data_given_orientation = data_given_orientation
            self.data_current_orientation = None
            self.data_return_orientation = None

            self.DATA_TRANSPOSE = None
            self.data_transpose_given_format = data_transpose_given_format
            self.data_transpose_current_format = None
            self.data_transpose_return_format = None
            self.data_transpose_given_orientation = data_transpose_given_orientation
            self.data_transpose_current_orientation = None
            self.data_transpose_return_orientation = None

            self.XTX = None
            self.xtx_given_format = xtx_given_format
            self.xtx_current_format = None
            self.xtx_return_format = None

            self.XTX_INV = None
            self.xtx_inv_given_format = xtx_inv_given_format
            self.xtx_inv_current_format = None
            self.xtx_inv_return_format = None

            self.target_is_multiclass = None
            self.TARGET = None
            self.target_given_format = target_given_format
            self.target_current_format = None
            self.target_return_format = None
            self.target_given_orientation = target_given_orientation
            self.target_current_orientation = None
            self.target_return_orientation = None

            self.TARGET_TRANSPOSE = None
            self.target_transpose_given_format = target_transpose_given_format
            self.target_transpose_current_format = None
            self.target_transpose_return_format = None
            self.target_transpose_given_orientation = target_transpose_given_orientation
            self.target_transpose_current_orientation = None
            self.target_transpose_return_orientation = None

            self.TARGET_AS_LIST = None
            self.target_as_list_given_format = target_as_list_given_format
            self.target_as_list_current_format = None
            self.target_as_list_given_orientation = target_as_list_given_orientation
            self.target_as_list_current_orientation = None
            self.target_as_list_return_orientation = None

            #####################################################################################################################
            #####################################################################################################################
            #####################################################################################################################
            # AS OF 12/15/22, THIS IS SETUP UP TO SHORT-CIRCUIT ONLY IF ALL OF THE REQUIRED RETURNS ARE SATISFIED (BY HAVING THE
            # OBJECT, RETURN FORMAT == GIVEN FORMAT, AND RETURN ORIENTATION == GIVEN ORIENTATION.)  HAVE NOT YET THOUGHT OF A
            # FEASIBLE WAY TO SET THIS UP THAT IT CAN BYPASS TO RUN JUST ONE. IF TRY TO RUN JUST ONE, ALL
            # OF THE OTHER BUILDS WILL BE TRIGGERED, REBUILDING THEM ALL ANYWAY. SO IT'S ALL OR NONE FOR NOW.

            if bypass_validation:
                for _ in RETURN_OBJECTS:
                    if _ == 'DATA':
                        if not DATA is None and data_return_orientation in [data_given_orientation, 'AS_GIVEN'] and \
                                data_return_format in [data_given_format, 'AS_GIVEN']: pass
                        else: break

                    elif _ == 'DATA_TRANSPOSE':
                        if not DATA_TRANSPOSE is None and \
                                data_transpose_return_orientation in [data_transpose_given_orientation, 'AS_GIVEN'] and \
                                data_transpose_return_format in [data_transpose_given_format, 'AS_GIVEN']: pass
                        else: break

                    elif _ == 'XTX':
                        if not XTX is None and xtx_return_format in [xtx_given_format, 'AS_GIVEN']: pass
                        else: break

                    elif _ == 'XTX_INV':
                        if not XTX_INV is None and xtx_inv_return_format in [xtx_inv_given_format, 'AS_GIVEN']: pass
                        else: break

                    elif _ == 'TARGET':
                        if not TARGET is None and target_return_orientation in [target_given_orientation, 'AS_GIVEN'] and \
                                target_return_format in [target_given_format, 'AS_GIVEN']: pass
                        else: break

                    elif _ == 'TARGET_TRANSPOSE':
                        if not TARGET_TRANSPOSE is None and \
                                target_transpose_return_orientation in [target_transpose_given_orientation, 'AS_GIVEN'] and \
                                target_transpose_return_format in [target_transpose_given_format, 'AS_GIVEN']: pass
                        else: break

                    elif _ == 'TARGET_AS_LIST':
                        if not TARGET_AS_LIST is None and not isinstance(TARGET_AS_LIST, (np.ndarray, list, tuple)):
                            _exception(fxn, f'TARGET_AS_LIST MUST BE A LIST-TYPE!')

                        if not TARGET_AS_LIST is None and \
                            target_as_list_return_orientation in [target_as_list_given_orientation, 'AS_GIVEN']: pass
                        else: break

                else:    # IF GET THRU for LOOP W/O A BREAK

                    self.RETURN_OBJECTS = RETURN_OBJECTS

                    for _ in RETURN_OBJECTS:
                        if _ == 'DATA':
                            self.DATA = DATA
                            self.data_given_format = data_given_format
                            self.data_current_format = data_given_format
                            self.data_return_format = data_given_format
                            self.data_given_orientation = data_given_orientation
                            self.data_current_orientation = data_given_orientation
                            self.data_return_orientation = data_given_orientation
                        elif _ == 'DATA_TRANSPOSE':
                            self.DATA_TRANSPOSE = DATA_TRANSPOSE
                            self.data_transpose_given_format = data_transpose_given_format
                            self.data_transpose_current_format = data_transpose_given_format
                            self.data_transpose_return_format = data_transpose_given_format
                            self.data_transpose_given_orientation = data_transpose_given_orientation
                            self.data_transpose_current_orientation = data_transpose_given_orientation
                            self.data_transpose_return_orientation = data_transpose_given_orientation
                        elif _ == 'XTX':
                            self.XTX = XTX
                            self.xtx_given_format = xtx_given_format
                            self.xtx_current_format = xtx_given_format
                            self.xtx_return_format = xtx_given_format
                        elif _ == 'XTX_INV':
                            self.XTX_INV = XTX_INV
                            self.xtx_inv_given_format = xtx_inv_given_format
                            self.xtx_inv_current_format = xtx_inv_given_format
                            self.xtx_inv_return_format = xtx_inv_given_format
                        elif _ == 'TARGET':
                            self.TARGET = TARGET
                            self.target_given_format = target_given_format
                            self.target_current_format = target_given_format
                            self.target_return_format = target_given_format
                            self.target_given_orientation = target_given_orientation
                            self.target_current_orientation = target_given_orientation
                            self.target_return_orientation = target_given_orientation
                        elif _ == 'TARGET_TRANSPOSE':
                            self.TARGET_TRANSPOSE = TARGET_TRANSPOSE
                            self.target_transpose_given_format = target_transpose_given_format
                            self.target_transpose_current_format = target_transpose_given_format
                            self.target_transpose_return_format = target_transpose_given_format
                            self.target_transpose_given_orientation = target_transpose_given_orientation
                            self.target_transpose_current_orientation = target_transpose_given_orientation
                            self.target_transpose_return_orientation = target_transpose_given_orientation
                        elif _ == 'TARGET_AS_LIST':
                            self.TARGET_AS_LIST = TARGET_AS_LIST
                            self.target_as_list_given_format = target_as_list_given_format
                            self.target_as_list_current_format = target_as_list_given_format
                            self.target_as_list_given_orientation = target_as_list_given_orientation
                            self.target_as_list_current_orientation = target_as_list_given_orientation
                            self.target_as_list_return_orientation = target_as_list_given_orientation

                    break  # IF GET THRU for LOOP W/O A BREAK, CLASS ATTRIBUTES ARE SET AND THIS BREAK IS HIT AND
                            # BYPASSES THE HEAVY MACHINERY TO END init
            #####################################################################################################################
            #####################################################################################################################
            #####################################################################################################################

            if isinstance(TARGET, (np.ndarray, list, tuple)) and len(np.array(TARGET).shape)==1: TARGET = TARGET.reshape((1,-1))

            RETURN_OBJECTS = np.char.upper(RETURN_OBJECTS) if len(RETURN_OBJECTS) != 0 else RETURN_OBJECTS

            ##################################################################################################################################
            # FIRST ROUND OF CONDITIONAL VALIDATION ####################################################################################################
            if not bypass_validation:
                # KWARG VALIDATION #####################################################################################################
                data_given_orientation = akv.arg_kwarg_validater(data_given_orientation, 'data_given_orientation',
                        ['COLUMN', 'ROW', None], this_module, fxn)
                data_return_orientation = akv.arg_kwarg_validater(data_return_orientation, 'data_return_orientation',
                        ['COLUMN', 'ROW', 'AS_GIVEN', None], this_module, fxn)
                data_return_format = akv.arg_kwarg_validater(data_return_format, 'data_return_format',
                        ['ARRAY', 'SPARSE_DICT', 'AS_GIVEN', None], this_module, fxn)
                data_transpose_given_orientation = akv.arg_kwarg_validater(data_transpose_given_orientation, 'data_transpose_given_orientation',
                        ['COLUMN', 'ROW', None], this_module, fxn)
                data_transpose_return_orientation = akv.arg_kwarg_validater(data_transpose_return_orientation, 'data_transpose_return_orientation',
                        ['COLUMN', 'ROW', 'AS_GIVEN', None], this_module, fxn)
                data_transpose_return_format = akv.arg_kwarg_validater(data_transpose_return_format, 'data_transpose_return_format',
                        ['ARRAY', 'SPARSE_DICT', 'AS_GIVEN', None], this_module, fxn)
                xtx_return_format = akv.arg_kwarg_validater(xtx_return_format, 'xtx_return_format',
                        ['ARRAY', 'SPARSE_DICT', 'AS_GIVEN', None], this_module, fxn)
                xtx_inv_return_format = akv.arg_kwarg_validater(xtx_inv_return_format, 'xtx_inv_return_format',
                        ['ARRAY', 'SPARSE_DICT', 'AS_GIVEN', None], this_module, fxn)
                target_is_multiclass = akv.arg_kwarg_validater(target_is_multiclass, 'target_is_multiclass',
                        [True, False, None], this_module, fxn, return_if_none=False)
                target_given_orientation = akv.arg_kwarg_validater(target_given_orientation, 'target_given_orientation',
                        ['COLUMN', 'ROW', None], this_module, fxn)
                target_return_orientation = akv.arg_kwarg_validater(target_return_orientation, 'target_return_orientation',
                        ['COLUMN', 'ROW', 'AS_GIVEN', None], this_module, fxn)
                target_return_format = akv.arg_kwarg_validater(target_return_format, 'target_return_format',
                        ['ARRAY', 'SPARSE_DICT', 'AS_GIVEN', None], this_module, fxn)
                target_transpose_given_orientation = akv.arg_kwarg_validater(target_transpose_given_orientation, 'target_transpose_given_orientation',
                        ['COLUMN', 'ROW', None], this_module, fxn)
                target_transpose_return_orientation = akv.arg_kwarg_validater(target_transpose_return_orientation, 'target_transpose_return_orientation',
                        ['COLUMN', 'ROW', 'AS_GIVEN', None], this_module, fxn)
                target_transpose_return_format = akv.arg_kwarg_validater(target_transpose_return_format, 'target_transpose_return_format',
                        ['ARRAY', 'SPARSE_DICT', 'AS_GIVEN', None], this_module, fxn)
                target_as_list_given_orientation = akv.arg_kwarg_validater(target_as_list_given_orientation, 'target_as_list_given_orientation',
                        ['COLUMN', 'ROW', None], this_module, fxn)
                target_as_list_return_orientation = akv.arg_kwarg_validater(target_as_list_return_orientation, 'target_as_list_return_orientation',
                        ['COLUMN', 'ROW', 'AS_GIVEN', None], this_module, fxn)
                # END KWARG VALIDATION #####################################################################################################

                '''   ARGS/KWARGS VALIDATED TO THIS POINT
                data_given_orientation
                data_return_orientation
                data_return_format
                data_transpose_given_orientation
                data_transpose_return_orientation
                data_transpose_return_format
                xtx_return_format
                xtx_inv_return_format
                target_is_multiclass
                target_given_orientation
                target_return_orientation
                target_return_format
                target_transpose_given_orientation
                target_transpose_return_orientation
                target_transpose_return_format
                target_as_list_given_orientation
                target_as_list_return_orientation
                RETURN_OBJECTS'''

            # END FIRST ROUND OF CONDITIONAL VALIDATION ####################################################################################################
            ##################################################################################################################################

            ##################################################################################################################################
            # ASSIGN ALL REMAINING args/kwargs BASED ON THOSE GIVEN ABOVE. SHOULD BE FAST SO JUST DO ALL OF THEM WHETHER USED OR NOT #########
            # THESE WILL ALL FEED THE OBJECT BUILDERS/ORIENTERS, WHETHER THE OBJECTS ARE RETURNED OR NOT #####################################

            # Return given state if return state is 'AS_GIVEN'
            is_given_flipper = lambda x, y: x if y == 'AS_GIVEN' else y

            # MAKE DATA GIVEN/RETURN DEFAULT TO DATA_TRANSPOSE SETTINGS IF DATA IS GIVEN BUT SETTINGS ARE NOT
            # data_given_orientation = itself
            if not DATA is None:
                if data_given_orientation is None: data_given_orientation = data_transpose_given_orientation
                if data_return_orientation is None: data_return_orientation = data_transpose_return_orientation
                if data_return_format is None: data_return_format = data_transpose_return_format

            data_return_orientation = is_given_flipper(data_given_orientation if not data_given_orientation is None else data_transpose_given_orientation, data_return_orientation)
            data_return_format = is_given_flipper(data_given_format if not data_given_format is None else data_transpose_given_format, data_return_format)

            # MAKE DATA_TRANSPOSE GIVEN/RETURN DEFAULT TO DATA SETTINGS IF DATA_TRANSPOSE IS GIVEN BUT SETTINGS ARE NOT
            if not DATA_TRANSPOSE is None:
                if data_transpose_given_orientation is None: data_transpose_given_orientation = data_given_orientation
                if data_transpose_return_orientation is None: data_transpose_return_orientation = data_return_orientation
                if data_transpose_return_format is None: data_transpose_return_format = data_return_format

            data_transpose_return_orientation = is_given_flipper(data_transpose_given_orientation if not data_transpose_given_orientation is None else data_given_orientation, data_transpose_return_orientation)
            data_transpose_return_format = is_given_flipper(data_transpose_given_format if not data_transpose_given_format is None else data_given_format, data_transpose_return_format)

            # XTX IS SYMMETRIC SO DONT NEED ORIENTATION
            xtx_return_format = is_given_flipper(xtx_given_format if not xtx_given_format is None else \
                 xtx_inv_given_format if not XTX_INV is None else data_given_format, xtx_return_format)

            # XTX_INV IS SYMMETRIC SO DONT NEED ORIENTATION
            xtx_inv_return_format = is_given_flipper(xtx_inv_given_format if not xtx_inv_given_format is None else
                 xtx_given_format if not xtx_given_format is None else data_given_format, xtx_inv_return_format)

            if not TARGET is None:
                if target_given_orientation is None: target_given_orientation = target_as_list_given_orientation if not target_as_list_given_orientation is None else target_transpose_given_orientation
                if target_return_orientation is None: target_return_orientation = target_as_list_return_orientation if not target_as_list_return_orientation is None else target_transpose_return_orientation
                if target_return_format is None: target_return_format = target_transpose_return_format if not target_transpose_return_format is None else 'ARRAY'

            target_return_orientation = is_given_flipper(target_given_orientation if not target_given_orientation is None else target_as_list_given_orientation if not target_as_list_given_orientation is None else target_transpose_given_orientation, target_return_orientation)
            target_return_format = is_given_flipper(target_given_format if not target_given_format is None else 'ARRAY' if not target_as_list_given_format is None else target_transpose_given_format, target_return_format)
            # target_is_multiclass = itself

            if not TARGET_TRANSPOSE is None:
                if target_transpose_given_orientation is None: target_transpose_given_orientation = target_given_orientation if not target_given_orientation is None else target_as_list_given_orientation
                if target_transpose_return_orientation is None: target_transpose_return_orientation = target_return_orientation if not target_return_orientation is None else target_as_list_return_orientation
                if target_transpose_return_format is None: target_transpose_return_format = target_return_format if not target_return_format is None else 'ARRAY'

            target_transpose_return_orientation = is_given_flipper(target_transpose_given_orientation if not target_transpose_given_orientation is None else target_given_orientation if not target_given_orientation is None else target_as_list_given_orientation, target_transpose_return_orientation)
            target_transpose_return_format = is_given_flipper(target_transpose_given_format if not target_transpose_given_format is None else target_given_format if not target_given_format is None else 'ARRAY' if not TARGET_AS_LIST is None else None, target_transpose_return_format)

            if not TARGET_AS_LIST is None:
                if target_as_list_given_format is None: target_as_list_given_format = target_given_format if not target_given_format is None else target_transpose_given_format
                if target_as_list_given_orientation is None: target_as_list_given_orientation = target_given_orientation if not target_given_orientation is None else target_transpose_given_orientation
                if target_as_list_return_orientation is None: target_as_list_return_orientation = target_return_orientation if not target_return_orientation is None else target_as_list_return_orientation

            target_as_list_return_orientation = is_given_flipper(target_as_list_given_orientation if not target_as_list_given_orientation is None else target_given_orientation if not target_given_orientation is None else target_transpose_given_orientation, target_as_list_return_orientation)
            target_as_list_return_format = 'ARRAY'

            del is_given_flipper
            # END ASSIGN REMAINING ARGS/KWARGS #############################################################################################
            ##################################################################################################################################

            ##################################################################################################################################
            # SECOND ROUND OF CONDITIONAL VALIDATION ################################################################################################
            if not bypass_validation:

                # KWARG VALIDATION #####################################################################################################
                data_return_orientation = akv.arg_kwarg_validater(data_return_orientation, 'data_return_orientation',
                        ['COLUMN', 'ROW', None], this_module, fxn)
                data_given_format = akv.arg_kwarg_validater(data_given_format, 'data_given_format',
                        ['ARRAY', 'SPARSE_DICT', None], this_module, fxn)
                data_return_format = akv.arg_kwarg_validater(data_return_format, 'data_return_format',
                        ['ARRAY', 'SPARSE_DICT', None], this_module, fxn)
                data_transpose_given_orientation = akv.arg_kwarg_validater(data_transpose_given_orientation, 'data_transpose_given_orientation',
                        ['COLUMN', 'ROW', None], this_module, fxn)
                data_transpose_return_orientation = akv.arg_kwarg_validater(data_transpose_return_orientation, 'data_transpose_return_orientation',
                        ['COLUMN', 'ROW', None], this_module, fxn)
                data_transpose_given_format = akv.arg_kwarg_validater(data_transpose_given_format, 'data_transpose_given_format',
                        ['ARRAY', 'SPARSE_DICT', None], this_module, fxn)
                data_transpose_return_format = akv.arg_kwarg_validater(data_transpose_return_format, 'data_transpose_return_format',
                        ['ARRAY', 'SPARSE_DICT', None], this_module, fxn)
                xtx_given_format = akv.arg_kwarg_validater(xtx_given_format, 'xtx_given_format',
                        ['ARRAY', 'SPARSE_DICT', None], this_module, fxn)
                xtx_return_format = akv.arg_kwarg_validater(xtx_return_format, 'xtx_return_format',
                        ['ARRAY', 'SPARSE_DICT', None], this_module, fxn)
                xtx_inv_given_format = akv.arg_kwarg_validater(xtx_inv_given_format, 'xtx_inv_given_format',
                        ['ARRAY', 'SPARSE_DICT', None], this_module, fxn)
                xtx_inv_return_format = akv.arg_kwarg_validater(xtx_inv_return_format, 'xtx_inv_return_format',
                        ['ARRAY', 'SPARSE_DICT', None], this_module, fxn)
                target_given_format = akv.arg_kwarg_validater(target_given_format, 'target_given_format',
                        ['ARRAY', 'SPARSE_DICT', None], this_module, fxn)
                target_return_orientation = akv.arg_kwarg_validater(target_return_orientation, 'target_return_orientation',
                        ['COLUMN', 'ROW', 'AS_GIVEN', None], this_module, fxn)
                target_return_format = akv.arg_kwarg_validater(target_return_format, 'target_return_format',
                        ['ARRAY', 'SPARSE_DICT', 'AS_GIVEN', None], this_module, fxn)
                target_transpose_given_orientation = akv.arg_kwarg_validater(target_transpose_given_orientation, 'target_transpose_given_orientation',
                        ['COLUMN', 'ROW', None], this_module, fxn)
                target_transpose_return_orientation = akv.arg_kwarg_validater(target_transpose_return_orientation, 'target_transpose_return_orientation',
                        ['COLUMN', 'ROW', 'AS_GIVEN', None], this_module, fxn)
                target_transpose_given_format = akv.arg_kwarg_validater(target_transpose_given_format, 'target_transpose_given_format',
                        ['ARRAY', 'SPARSE_DICT', None], this_module, fxn)
                target_transpose_return_format = akv.arg_kwarg_validater(target_transpose_return_format, 'target_transpose_return_format',
                        ['ARRAY', 'SPARSE_DICT', 'AS_GIVEN', None], this_module, fxn)
                target_as_list_given_orientation = akv.arg_kwarg_validater(target_as_list_given_orientation, 'target_as_list_given_orientation',
                        ['COLUMN', 'ROW', None], this_module, fxn)
                target_as_list_return_orientation = akv.arg_kwarg_validater(target_as_list_return_orientation, 'target_as_list_return_orientation',
                        ['COLUMN', 'ROW', 'AS_GIVEN', None], this_module, fxn)
                target_as_list_given_format = akv.arg_kwarg_validater(target_as_list_given_format, 'target_as_list_given_format',
                        ['ARRAY', 'SPARSE_DICT', None], this_module, fxn)
                target_as_list_return_format = akv.arg_kwarg_validater(target_as_list_return_format, 'target_as_list_return_format',
                        ['ARRAY'], this_module, fxn)

                # END KWARG VALIDATION #####################################################################################################
            # END SECOND ROUND OF CONDITIONAL VALIDATION ################################################################################################
            ##################################################################################################################################

            ################################################################################################################################
            # THIRD ROUND OF CONDITIONAL VALIDATION #########################################################################################
            if not bypass_validation:
                # VALIDATE RETURN_OBJECTS ####################################################################################################
                RETURN_OBJECTS = akv.arg_kwarg_validater(RETURN_OBJECTS, 'RETURN_OBJECTS',
                      ['DATA', 'DATA_TRANSPOSE', 'XTX', 'XTX_INV', 'TARGET', 'TARGET_TRANSPOSE', 'TARGET_AS_LIST', None],
                      this_module, 'ObjectOrienter.__init__()',
                      return_if_none=['DATA', 'DATA_TRANSPOSE', 'XTX', 'XTX_INV', 'TARGET', 'TARGET_TRANSPOSE', 'TARGET_AS_LIST'])
                # END VALIDATE RETURN_OBJECTS ####################################################################################################

                # VERIFY RETURN_OBJECTS CAN BE SATISFIED BY GIVEN OBJECTS ##################################################################

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")

                    if ('TARGET' in RETURN_OBJECTS or 'TARGET_TRANSPOSE' in RETURN_OBJECTS or 'TARGET_AS_LIST' in RETURN_OBJECTS) and \
                            (TARGET is None and TARGET_TRANSPOSE is None and TARGET_AS_LIST is None):
                        _exception(fxn, f'AT LEAST ONE TARGET OBJECT MUST BE GIVEN TO RETURN A TARGET OBJECT')

                    if ('XTX' in RETURN_OBJECTS or 'XTX_INV' in RETURN_OBJECTS) and \
                            (DATA is None and DATA_TRANSPOSE is None and XTX is None and XTX_INV is None):
                        _exception(fxn, f'AT LEAST ONE OF THE DATA OBJECTS OR XTX OBJECTS MUST BE GIVEN TO RETURN AN XTX OBJECT')

                    if ('DATA' in RETURN_OBJECTS or 'DATA_TRANSPOSE' in RETURN_OBJECTS) and \
                            (DATA is None and DATA_TRANSPOSE is None):
                        _exception(fxn, f'AT LEAST ONE DATA OBJECT MUST BE GIVEN TO RETURN A DATA OBJECT')
                # END VERIFY RETURN_OBJECTS CAN BE SATISFIED BY GIVEN OBJECTS ##################################################################
            # END THIRD ROUND OF CONDITIONAL VALIDATION #########################################################################################
            ################################################################################################################################


            ################################################################################################################################
            # FOURTH ROUND OF CONDITIONAL VALIDATION - VERIFY CONGRUENCY OF GIVEN OBJECTS ##################################################
            ### PUT GIVEN OBJECTS IN MLObject class AND CROSS-VERIFY CONGRUENCE OF ALL OBJECTS ############################################

            if not bypass_validation:

                # REMEMBER OBJECTS HAVE NOT BEEN MODIFIED IN ANY WAY AT THIS POINT, STILL IN AS-GIVEN STATE

                # CREATE TEMPORARY MLObjects FOR VALIDATION, TO BE MADE FROM ORIGINAL args. DO NOT DO THE VALIDATION
                # DIRECTLY ON THE ORIGINAL args, THEY WOULD BE CHANGED. ORIGINAL args MUST BE PRESERVED FOR FINAL BUILDS.
                self.TARGET, self.TARGET_TRANSPOSE, self.TARGET_AS_LIST, self.DATA, self.DATA_TRANSPOSE, self.XTX, self.XTX_INV = \
                    None, None, None, None, None, None, None

                ########################################################################################################################
                # TARGET INPUT VALIDATION ##############################################################################################
                # THIS CAN STAND ALONE, NO CROSS-VALIDATION W/ DATA &/OR XTX

                # IF ONE OBJECT IS GIVEN, CANNOT VALIDATE. IF NO TARGET RETURN(S) REQUIRED, DO NOT NEED TO VALIDATE
                # IF THERE ARE OBJECTS, MAKE TEMPORARY MLObject CLASSES FOR THEM AND USE CLASS INTERNALS TO VERIFY EQUALITY

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")

                    more_than_one_target_object_given = \
                                        sum([1 for _ in (TARGET, TARGET_TRANSPOSE, TARGET_AS_LIST) if not _ is None]) > 1

                    num_target_objects_to_be_returned = \
                                        sum([1 for _ in ('TARGET', 'TARGET_TRANSPOSE', 'TARGET_AS_LIST') if _ in RETURN_OBJECTS])

                    more_than_one_data_object_given = sum([1 for _ in (DATA, DATA_TRANSPOSE) if not _ is None]) > 1

                    num_data_objects_to_be_returned = sum([1 for _ in ('DATA', 'DATA_TRANSPOSE') if _ in RETURN_OBJECTS])

                    more_than_one_xtx_object_given = sum([1 for _ in (XTX, XTX_INV) if not _ is None]) > 1

                    num_xtx_objects_to_be_returned = sum([1 for _ in ('XTX', 'XTX_INV') if _ in RETURN_OBJECTS])

                if not more_than_one_target_object_given or num_target_objects_to_be_returned == 0:
                    pass
                else:
                    # MAKE CLASSES ########################################################################################
                    if not TARGET is None:   # RETURN AS GIVEN
                        self.TARGET = mlto.MLTargetObject(TARGET,
                                                          target_given_orientation,
                                                          is_multiclass=target_is_multiclass,
                                                          bypass_validation=bypass_validation,
                                                          calling_module=this_module,
                                                          calling_fxn=fxn)

                    if not TARGET_TRANSPOSE is None:   # RETURN AS GIVEN
                        self.TARGET_TRANSPOSE = mltto.MLTargetTransposeObject(TARGET_TRANSPOSE,
                                                                              target_transpose_given_orientation,
                                                                              is_multiclass=target_is_multiclass,
                                                                              bypass_validation=bypass_validation,
                                                                              calling_module=this_module,
                                                                              calling_fxn=fxn)

                    if not TARGET_AS_LIST is None:   # RETURN AS GIVEN
                        self.TARGET_AS_LIST = mlto.MLTargetObject(TARGET_AS_LIST,
                                                                  target_as_list_given_orientation,
                                                                  is_multiclass=target_is_multiclass,
                                                                  bypass_validation=bypass_validation,
                                                                  calling_module=this_module,
                                                                  calling_fxn=fxn)
                    # END MAKE CLASSES ########################################################################################

                    # TEST ####################################################################################################
                    if not self.TARGET is None and not self.TARGET_TRANSPOSE is None:

                        if self.TARGET.current_orientation == self.TARGET_TRANSPOSE.current_orientation and not \
                            self.TARGET.is_equiv(self.TARGET_TRANSPOSE.get_transpose()):
                            _exception(fxn, f'GIVEN "TARGET" AND "TARGET_TRANSPOSE" OBJECTS ARE NOT CUT FROM THE SAME CLOTH')
                        elif self.TARGET.current_orientation != self.TARGET_TRANSPOSE.current_orientation and not \
                            self.TARGET.is_equiv(self.TARGET_TRANSPOSE.OBJECT):
                            _exception(fxn, f'GIVEN "TARGET" AND "TARGET_TRANSPOSE" OBJECTS ARE NOT CUT FROM THE SAME CLOTH')

                    if not self.TARGET is None and not self.TARGET_AS_LIST is None:
                        if self.TARGET.current_orientation == self.TARGET_AS_LIST.current_orientation and not \
                            self.TARGET.is_equiv(self.TARGET_AS_LIST.OBJECT):
                            _exception(fxn, f'GIVEN "TARGET" AND "TARGET_AS_LIST" OBJECTS ARE NOT CUT FROM THE SAME CLOTH')
                        elif self.TARGET.current_orientation != self.TARGET_AS_LIST.current_orientation and not \
                            self.TARGET.is_equiv(self.TARGET_AS_LIST.get_transpose()):
                            _exception(fxn, f'GIVEN "TARGET" AND "TARGET_AS_LIST" OBJECTS ARE NOT CUT FROM THE SAME CLOTH')

                    if self.TARGET is None and not self.TARGET_TRANSPOSE is None and not self.TARGET_AS_LIST is None:
                        if self.TARGET_TRANSPOSE.current_orientation == self.TARGET_AS_LIST.current_orientation and not \
                            self.TARGET_TRANSPOSE.is_equiv(self.TARGET_AS_LIST.get_transpose()):
                            _exception(fxn, f'GIVEN "TARGET_TRANSPOSE" AND "TARGET_AS_LIST" OBJECTS ARE NOT CUT FROM THE SAME CLOTH')
                        elif self.TARGET_TRANSPOSE.current_orientation != self.TARGET_AS_LIST.current_orientation and not \
                            self.TARGET_TRANSPOSE.is_equiv(self.TARGET_AS_LIST.OBJECT):
                            _exception(fxn, f'GIVEN "TARGET_TRANSPOSE" AND "TARGET_AS_LIST" OBJECTS ARE NOT CUT FROM THE SAME CLOTH')
                    # END TEST ################################################################################################

                # END TARGET INPUT VALIDATION #################################################################################
                ###############################################################################################################

                ###############################################################################################################
                # DATA INPUT VALIDATION #######################################################################################

                if not more_than_one_data_object_given or (num_data_objects_to_be_returned == 0 and
                       (num_xtx_objects_to_be_returned == 0 or (not XTX is None or not XTX_INV is None))):
                    pass
                else:
                    # MAKE CLASSES ############################################################################################
                    if not DATA is None:   # RETURN AS GIVEN
                        self.DATA = mlo.MLObject(DATA,
                                                 data_given_orientation,
                                                 name='DATA',
                                                 bypass_validation=bypass_validation,
                                                 calling_module=this_module,
                                                 calling_fxn=fxn)

                    elif not DATA_TRANSPOSE is None:   # RETURN AS GIVEN
                        self.DATA_TRANSPOSE = mlo.MLObject(DATA_TRANSPOSE,
                                                           data_transpose_given_orientation,
                                                           name='DATA_TRANSPOSE',
                                                           bypass_validation=bypass_validation,
                                                           calling_module=this_module,
                                                           calling_fxn=fxn)
                    # END MAKE CLASSES #########################################################################################

                    # TEST ####################################################################################################
                    if not self.DATA is None and not self.DATA_TRANSPOSE is None:
                        if not self.DATA.is_equiv(self.DATA_TRANSPOSE.get_transpose(), test_as=self.DATA.current_format):
                            _exception(fxn, f'GIVEN "DATA" AND "DATA_TRANSPOSE" OBJECTS ARE NOT CUT FROM THE SAME CLOTH')
                    # END TEST ################################################################################################

                # END DATA INPUT VALIDATION ###################################################################################
                ###############################################################################################################

                ###############################################################################################################
                # XTX / XTX_INV INPUT VALIDATION ##############################################################################

                if not more_than_one_xtx_object_given or num_xtx_objects_to_be_returned == 0:
                    pass
                else:
                    # MAKE CLASSES ############################################################################################
                    if not XTX is None:   # RETURN AS GIVEN
                        self.XTX = mlos.MLObjectSymmetric(XTX, calling_module=this_module, calling_fxn=fxn)

                    if not XTX_INV is None:   # RETURN AS GIVEN
                        self.XTX_INV = mlos.MLObjectSymmetric(XTX_INV, calling_module=this_module, calling_fxn=fxn)
                    # END MAKE CLASSES ########################################################################################

                    # TEST ####################################################################################################
                    if not np.allclose(np.linalg.inv(self.XTX.return_as_array()),
                                       self.XTX_INV.return_as_array(),
                                       atol=1e-10,
                                       rtol=1e-10):
                        _exception(fxn, f'INVERSE OF GIVEN "XTX" AND GIVEN "XTX_INV" OBJECTS ARE NOT EQUAL')
                    # END TEST ################################################################################################
                # END XTX / XTX_INV INPUT VALIDATION ##########################################################################
                ###############################################################################################################


                # #############################################################################################################
                # DATA / XTX CROSS VALIDATION #################################################################################
                # DATA & XTX OBJECT CLASSES SHOULD BE CREATED AT THIS POINT, IF NEEDED
                # DATA SHOULD BE VERIFIED AGAINST DATA_TRANSPOSE, XTX SHOULD BE VERIFIED AGAINST XTX_INV
                # SO SHOULD ONLY HAVE TO VERIFY ONE OF [DATA, DATA_TRANSPOSE] AGAINST ONE OF [XTX, XTX_INV]

                if not self.DATA is None:
                    if not self.XTX is None:
                        if not np.allclose(self.DATA.return_XTX(return_format='ARRAY'),
                                            self.XTX.return_as_array(),
                                            atol=1e-10,
                                            rtol=1e-10):
                            _exception(fxn, f'GIVEN DATA OBJECT AND GIVEN XTX OBJECT ARE NOT CUT FROM THE SAME CLOTH')
                    elif not self.XTX_INV is None:
                        if not np.allclose(self.DATA.return_XTX_INV(return_format='ARRAY'),
                                          self.XTX_INV.return_as_array(),
                                          atol=1e-10,
                                          rtol=1e-10):
                            _exception(fxn, f'GIVEN DATA OBJECT AND GIVEN XTX_INV OBJECT ARE NOT CUT FROM THE SAME CLOTH')

                elif not self.DATA_TRANSPOSE is None:
                    if not self.XTX is None:
                        if not np.allclose(self.DATA_TRANSPOSE.return_XTX(return_format='ARRAY'),
                                           self.XTX.return_as_array(),
                                           atol=1e-10,
                                           rtol=1e-10):
                            _exception(fxn, f'GIVEN DATA_TRANSPOSE OBJECT AND GIVEN XTX OBJECT ARE NOT CUT FROM THE SAME CLOTH')
                    elif not self.XTX_INV is None:
                        if not np.allclose(self.DATA_TRANSPOSE.return_XTX_INV(return_format='ARRAY'),
                                              self.XTX_INV.return_as_array(),
                                              atol=1e-10,
                                              rtol=1e-10):
                            _exception(fxn, f'GIVEN DATA_TRANSPOSE OBJECT AND GIVEN XTX_INV OBJECT ARE NOT CUT FROM THE SAME CLOTH')

                # END DATA / XTX CROSS VALIDATION #################################################################################
                # #############################################################################################################

                # CANT USE THESE CLASSES TO FINALIZE THE OBJECTS, THESE WOULDNT BE CREATED IF bypass_validation is True
                del self.TARGET, self.TARGET_TRANSPOSE, self.TARGET_AS_LIST, self.DATA, self.DATA_TRANSPOSE, self.XTX, self.XTX_INV

            # END FOURTH ROUND OF CONDITIONAL VALIDATION - VERIFY CONGRUENCY OF GIVEN OBJECTS ##################################################
            ################################################################################################################################

            ###############################################################################################################################
            ###############################################################################################################################
            # BUILD OBJECTS FROM AVAILABLE OBJECTS ########################################################################################
            # EXCEPTIONS FOR INSUFFICIENT OBJECTS TO BUILD FROM IS REDUNDANT IF validation IS ON, BUT IS SOME PROTECTION IF validation IS OFF

            ###### DATA ##################################################################################################################
            self.DATA = None

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                if not (DATA is None and DATA_TRANSPOSE is None) and ('DATA' in RETURN_OBJECTS or
                    ('DATA_TRANSPOSE' in RETURN_OBJECTS and DATA_TRANSPOSE is None) or
                    (('XTX' in RETURN_OBJECTS or 'XTX_INV' in RETURN_OBJECTS) and (XTX is None and XTX_INV is None))):
                    # DATA CAN ONLY BE BUILT 2 WAYS, FROM GIVEN OR FROM DATA TRANSPOSE
                    if not DATA is None:   # FROM GIVEN
                        self.DATA = mlo.MLObject(DATA,
                                                 data_given_orientation,
                                                 name='DATA',
                                                 return_orientation=data_return_orientation,
                                                 return_format=data_return_format,
                                                 bypass_validation=bypass_validation,
                                                 calling_module=this_module,
                                                 calling_fxn=fxn)

                    elif DATA is None and not DATA_TRANSPOSE is None:
                        self.DATA = mlo.MLObject(DATA_TRANSPOSE.transpose() if data_transpose_given_format == 'ARRAY' else sd.core_sparse_transpose(DATA_TRANSPOSE),
                                                 data_transpose_given_orientation,
                                                 name='DATA',
                                                 return_orientation=data_return_orientation,
                                                 return_format=data_return_format,
                                                 bypass_validation=bypass_validation,
                                                 calling_module=this_module,
                                                 calling_fxn=fxn)
                # MANAGE OF self.DATA CLASS HANDLED AFTER VALIDATION
            ###### END DATA ##################################################################################################################

            ###### DATA TRANSPOSE ################################################################################################
            self.DATA_TRANSPOSE = None

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                if 'DATA_TRANSPOSE' in RETURN_OBJECTS:
                    # DATA_TRANSPOSE CAN ONLY BE BUILT 2 WAYS, FROM GIVEN OR FROM DATA

                    if not DATA_TRANSPOSE is None:
                        self.DATA_TRANSPOSE = mlo.MLObject(DATA_TRANSPOSE,
                                                            data_transpose_given_orientation,
                                                            name='DATA_TRANSPOSE',
                                                            return_orientation=data_transpose_return_orientation,
                                                            return_format=data_transpose_return_format,
                                                            bypass_validation=bypass_validation,
                                                            calling_module=this_module,
                                                            calling_fxn=fxn)

                    elif DATA_TRANSPOSE is None and not DATA is None:
                        self.DATA_TRANSPOSE = mlo.MLObject(DATA.transpose() if data_given_format == 'ARRAY' else sd.core_sparse_transpose(DATA),
                                                            data_given_orientation,
                                                            name='DATA_TRANSPOSE',
                                                            return_orientation=data_transpose_return_orientation,
                                                            return_format=data_transpose_return_format,
                                                            bypass_validation=bypass_validation,
                                                            calling_module=this_module,
                                                            calling_fxn=fxn)

                # MANAGE OF self.DATA_TRANSPOSE CLASS HANDLED AFTER VALIDATION
            ###### END DATA TRANSPOSE #############################################################################################

            ###### XTX #########################################################################################################
            self.XTX = None

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                if 'XTX' in RETURN_OBJECTS or ('XTX_INV' in RETURN_OBJECTS and XTX_INV is None):
                    # BUILD IF DIRECTLY CALLED FOR OR IF XTX_INV NEEDED (SAVES TIME AND CODE FOR XTX_INV)

                    # XTX CAN BE BUILT 5 WAYS, FROM GIVEN XTX, XTX_INV, (DATA & DATA_TRANSPOSE), JUST DATA, OR JUST DATA_TRANSPOSE
                    ##### TIME TESTS 12/7/22, COMPARE SPEED FOR GETTING XTX AS ARRAY FROM MATMUL OF X AS ARRAY OR linalg.inv(XTX_INV) #######
                    # NOT EVEN CLOSE.  FOR NUMPY, linalg.inv(XTX_INV) BEATS matmul(XT,X) EASILY
                    # FOR SD, zip_list(linalg.inv(unzip(SD_XTX_INV))) EASILY BEATS core_matmul(XT, X)

                    # np.linalg_inv on XTX_INV          average, sdev: time = 0.212 sec, 0.003; mem = 8.000, 0.000
                    # np.matmul on XT and X             average, sdev: time = 26.648 sec, 0.967; mem = 4.000, 0.000
                    # sd.unzip --> np.inv --> sd.zip    average, sdev: time = 0.948 sec, 0.065; mem = 88.333, 0.471
                    # sd.core_matmul(XT,X)              average, sdev: time = 35.395 sec, 1.441; mem = 65.000, 0.000

                    # ALSO VERIFIED INDEPENDENTLY THAT matmul REALLY IS THAT SLOW :(

                    if not XTX is None:
                        t0 = time.time()
                        self.XTX = mlos.MLObjectSymmetric(XTX,
                                                          return_format=xtx_return_format,
                                                          bypass_validation=bypass_validation,
                                                          calling_module=this_module,
                                                          calling_fxn=fxn
                                                          )

                    elif XTX is None and not XTX_INV is None:
                        self.XTX = mlos.MLObjectSymmetric(XTX_INV,
                                                          return_format=xtx_return_format,
                                                          bypass_validation=bypass_validation,
                                                          calling_module=this_module,
                                                          calling_fxn=fxn
                                                          )
                        self.XTX.invert()

                    elif XTX is None and XTX_INV is None:
                        # BUILD AN XTX OBJECT USING DATA
                        if self.DATA is None:
                            _exception(fxn, f'CANNOT BUILD XTX / XTX_INV WHEN DATA OR DATA_TRANSPOSE IS NOT GIVEN')
                            # DATA SHOULD HAVE BEEN BUILT ABOVE IF DATA OR DATA_TRANSPOSE WAS GIVEN AND XTX OR
                            # XTX_INV NEEDED AND XTX AND XTX_INV WERE NOT GIVEN. ONLY WAY THIS SHOULD ERROR IS IF BYPASSING
                            # VALIDATION AND NEEDING XTX OR XTX_INV WHEN DATA, DATA_TRANSPOSE, XTX AND XTX_INV WERE NOT GIVEN

                        # IF XTX_INV IS IN RETURN_OBJECTS, STILL HAVE TO MAKE IT FROM XTX OR WOULD HAVE ALREADY USED
                        # IT ABOVE TO MAKE XTX. SO MAKE XTX AS ARRAY, THEN IF GOING TO SD, DO IT AFTER MAKING XTX_INV.

                        self.XTX = mlos.MLObjectSymmetric(
                            self.DATA.return_XTX(return_format='ARRAY' if 'XTX_INV' in RETURN_OBJECTS else xtx_return_format),
                            'ARRAY' if 'XTX_INV' in RETURN_OBJECTS else xtx_return_format,
                            bypass_validation=bypass_validation,
                            calling_module=this_module,
                            calling_fxn=fxn
                        )

                # MANAGE OF self.XTX CLASS HANDLED AFTER VALIDATION
            ###### END XTX #########################################################################################################

            ###### XTX_INV #########################################################################################################
            self.XTX_INV = None

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                if 'XTX_INV' in RETURN_OBJECTS:

                    if not XTX_INV is None:
                        self.XTX_INV = mlos.MLObjectSymmetric(XTX_INV,
                                                              return_format=xtx_inv_return_format,
                                                              bypass_validation=bypass_validation,
                                                              calling_module=this_module,
                                                              calling_fxn=fxn)
                    elif not self.XTX is None:
                        self.XTX_INV = mlos.MLObjectSymmetric(self.XTX.OBJECT,
                                                              return_format=xtx_inv_return_format,
                                                              bypass_validation=bypass_validation,
                                                              calling_module=this_module,
                                                              calling_fxn=fxn)

                        self.XTX_INV.invert()

                    elif XTX_INV is None and self.DATA is None and self.DATA_TRANSPOSE is None and self.XTX is None:
                        _exception(fxn, f'CANNOT BUILD XTX_INV FROM SCRATCH WHEN DATA, DATA_TRANSPOSE, OR XTX ARE NOT GIVEN')
                        # XTX SHOULD HAVE BEEN BUILT ABOVE IF XTX_INV IN RETURN_OBJECTS AND XTX AND XTX_INV WERE NOT GIVEN AND
                        # DATA OR DATA_TRANSPOSE WERE GIVEN. ONLY WAY THIS SHOULD ERROR IS IF BYPASSING VALIDATION AND NEEDING
                        # XTX_INV WHEN DATA, DATA_TRANSPOSE, XTX AND XTX_INV WERE NOT GIVEN. IF XTX NOT AVAILABLE AT THIS POINT
                        # IF DATA/DATA_TRANSPOSE WERE GIVEN THEN BIG DISASTER.

                # MANAGE OF self.XTX_INV CLASS AFTER VALIDATION
            ###### END XTX_INV ###################################################################################################

            #### TARGET ###########################################################################################################
            self.TARGET = None

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                if 'TARGET' in RETURN_OBJECTS or (not TARGET is None and (('TARGET_TRANSPOSE' in RETURN_OBJECTS and TARGET_TRANSPOSE is None) or
                                                  ('TARGET_AS_LIST' in RETURN_OBJECTS and TARGET_AS_LIST is None))):

                    if not TARGET is None:
                        self.TARGET = mlto.MLTargetObject(
                                                      TARGET,
                                                      target_given_orientation,
                                                      return_orientation=target_return_orientation,
                                                      return_format=target_return_format,
                                                      is_multiclass=target_is_multiclass,
                                                      bypass_validation=bypass_validation,
                                                      calling_module=this_module,
                                                      calling_fxn=fxn
                                                      )

                    elif TARGET is None:
                        # 12/8/22 NOT RACKING MY HEAD ANYMORE TO OPTIMIZE WHAT TO USE FIRST FOR TARGET IF TARGET NOT GIVEN, COMES DOWN
                        # TO WHAT MIGHT NEED A ZIP/UNZIP OR A TRANSPOSE... DONT HAVE TIME FOR A PHD DISSERTATION, REVISIT IF NECESSARY
                        # TARGET FAR MORE OFTEN WILL BE ARRAY, SO GO WITH GETTING TARGET_AS_LIST FIRST, THEN TRY TARGET_TRANSPOSE
                        if not TARGET_AS_LIST is None:
                            self.TARGET = mlto.MLTargetObject(
                                                      TARGET_AS_LIST,
                                                      target_as_list_given_orientation,
                                                      return_orientation=target_return_orientation,
                                                      return_format=target_return_format,
                                                      is_multiclass=target_is_multiclass,
                                                      bypass_validation=bypass_validation,
                                                      calling_module=this_module,
                                                      calling_fxn=fxn
                                                      )

                        elif not TARGET_TRANSPOSE is None:
                            self.TARGET = mlto.MLTargetObject(
                                                      TARGET_TRANSPOSE,
                                                      'ROW' if target_transpose_given_orientation == 'COLUMN' else 'COLUMN',
                                                      return_orientation=target_return_orientation,
                                                      return_format=target_return_format,
                                                      is_multiclass=target_is_multiclass,
                                                      bypass_validation=bypass_validation,
                                                      calling_module=this_module,
                                                      calling_fxn=fxn
                                                      )

                        else: _exception(fxn, f'AT LEAST ONE TARGET OBJECT MUST BE GIVEN TO RETURN TARGET OBJECT(S).')

                # MANAGE self.TARGET class AFTER VALIDATION
            #### END TARGET ######################################################################################################

            #### TARGET_TRANSPOSE ################################################################################################
            self.TARGET_TRANSPOSE = None

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                if 'TARGET_TRANSPOSE' in RETURN_OBJECTS or (not TARGET_TRANSPOSE is None and
                                                              ('TARGET_AS_LIST' in RETURN_OBJECTS and TARGET_AS_LIST is None)):
                    if not TARGET_TRANSPOSE is None:
                        self.TARGET_TRANSPOSE = mltto.MLTargetTransposeObject(TARGET_TRANSPOSE,
                                                        target_transpose_given_orientation,
                                                        return_orientation=target_transpose_return_orientation,
                                                        return_format=target_transpose_return_format,
                                                        is_multiclass=target_is_multiclass,
                                                        bypass_validation=bypass_validation,
                                                        calling_module=this_module,
                                                        calling_fxn=fxn
                                                        )

                    elif TARGET_TRANSPOSE is None:
                        # GO WITH TARGET FIRST, LIKELY JUST A TRANSPOSE, WHERE TARGET_AS_LIST COULD BE ZIP/UNZIP AND A TRANSPOSE
                        if not self.TARGET is None:

                            self.TARGET_TRANSPOSE = mltto.MLTargetTransposeObject(
                                                        self.TARGET.OBJECT,
                                                        'ROW' if target_return_orientation == 'COLUMN' else 'COLUMN',
                                                        return_orientation=target_transpose_return_orientation,
                                                        return_format=target_transpose_return_format,
                                                        is_multiclass=target_is_multiclass,
                                                        bypass_validation=bypass_validation,
                                                        calling_module=this_module,
                                                        calling_fxn=fxn
                                                        )
                        elif not TARGET_AS_LIST is None:
                            self.TARGET_TRANSPOSE = mltto.MLTargetTransposeObject(
                                                        TARGET_AS_LIST,
                                                        'ROW' if target_as_list_given_orientation == 'COLUMN' else 'COLUMN',
                                                        return_orientation=target_transpose_return_orientation,
                                                        return_format=target_transpose_return_format,
                                                        is_multiclass=target_is_multiclass,
                                                        bypass_validation=bypass_validation,
                                                        calling_module=this_module,
                                                        calling_fxn=fxn
                                                        )

                        else: _exception(fxn, f'AT LEAST ONE TARGET OBJECT MUST BE GIVEN TO RETURN TARGET OBJECT(S).')

                # MANAGE self.TARGET_TRANSPOSE class AFTER VALIDATION
            #### END TARGET_TRANSPOSE ############################################################################################

            ###### TARGET_AS_LIST ################################################################################################
            self.TARGET_AS_LIST = None

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                if 'TARGET_AS_LIST' in RETURN_OBJECTS:
                    if not TARGET_AS_LIST is None:
                        self.TARGET_AS_LIST = mlto.MLTargetObject(TARGET_AS_LIST,
                                                        target_as_list_given_orientation,
                                                        return_orientation=target_as_list_return_orientation,
                                                        return_format=target_as_list_return_format,
                                                        is_multiclass=target_is_multiclass,
                                                        bypass_validation=bypass_validation,
                                                        calling_module=this_module,
                                                        calling_fxn=fxn
                                                        )
                    elif TARGET_AS_LIST is None:
                        # GET FROM TARGET FIRST, JUST A POSSIBLE ZIP/UNZIP, WHEREAS TARGET_TRANSPOSE COULD BE ZIP/UNZIP AND A TRANSPOSE
                        if not self.TARGET is None:
                            self.TARGET_AS_LIST = mlto.MLTargetObject(
                                                          self.TARGET.OBJECT,
                                                          target_return_orientation,
                                                          return_orientation=target_as_list_return_orientation,
                                                          return_format=target_as_list_return_format,
                                                          is_multiclass=target_is_multiclass,
                                                          bypass_validation=bypass_validation,
                                                          calling_module=this_module,
                                                          calling_fxn=fxn
                                                          )

                        elif not self.TARGET_TRANSPOSE is None:
                            self.TARGET_AS_LIST = mlto.MLTargetObject(
                                                        self.TARGET_TRANSPOSE.OBJECT,
                                                        'ROW' if target_transpose_return_orientation == 'COLUMN' else 'COLUMN',
                                                        return_orientation=target_as_list_return_orientation,
                                                        return_format=target_as_list_return_format,
                                                        is_multiclass=target_is_multiclass,
                                                        bypass_validation=bypass_validation,
                                                        calling_module=this_module,
                                                        calling_fxn=fxn
                                                        )
                # MANAGE self.TARGET_AS_LIST class AFTER VALIDATION
            ###### END TARGET_AS_LIST #############################################################################################

            # END BUILD OBJECTS FROM AVAILABLE OBJECTS ########################################################################################
            ###############################################################################################################################
            ###############################################################################################################################

            ###############################################################################################################################
            # FIFTH CONDITIONAL VALIDATION, OUTPUT OBJECTS #################################################################################
            if not bypass_validation:

                # ABOVE, IN ROUND FOUR OF VALIDATION, THE CLASSES CREATED FOR THE OBJECTS WERE JUST TEMPORARY HOLDERS TO EXPEDITE VALIDATION.
                # HERE, THESE ARE THE REAL MLObject CLASSES FOR THE FINAL OBJECTS, SO DO NOT MODIFY AND DO NOT DELETE!

                ########################################################################################################################
                # TARGET OUTPUT VALIDATION ######################################################################################################
                # THIS CAN STAND ALONE, NO CROSS-VALIDATION W/ DATA &/OR XTX

                # IF ONE OBJECT IS RETURNED, CANNOT VALIDATE. IF NO TARGET RETURN(S) REQUIRED, DO NOT NEED TO VALIDATE
                if num_target_objects_to_be_returned < 2:
                    pass
                else:
                    # TEST ####################################################################################################
                    if not self.TARGET is None and not self.TARGET_TRANSPOSE is None:
                        if self.TARGET.current_orientation == self.TARGET_TRANSPOSE.current_orientation and not \
                            self.TARGET.is_equiv(self.TARGET_TRANSPOSE.get_transpose()):
                            _exception(fxn, f'RETURNED "TARGET" AND "TARGET_TRANSPOSE" OBJECTS ARE NOT CUT FROM THE SAME CLOTH')
                        elif self.TARGET.current_orientation != self.TARGET_TRANSPOSE.current_orientation and not \
                            self.TARGET.is_equiv(self.TARGET_TRANSPOSE.OBJECT):
                            _exception(fxn, f'RETURNED "TARGET" AND "TARGET_TRANSPOSE" OBJECTS ARE NOT CUT FROM THE SAME CLOTH')

                    if not self.TARGET is None and not self.TARGET_AS_LIST is None:
                        if self.TARGET.current_orientation == self.TARGET_AS_LIST.current_orientation and not \
                            self.TARGET.is_equiv(self.TARGET_AS_LIST.OBJECT):
                            _exception(fxn, f'RETURNED "TARGET" AND "TARGET_AS_LIST" OBJECTS ARE NOT CUT FROM THE SAME CLOTH')
                        elif self.TARGET.current_orientation != self.TARGET_AS_LIST.current_orientation and not \
                            self.TARGET.is_equiv(self.TARGET_AS_LIST.get_transpose()):
                            _exception(fxn, f'RETURNED "TARGET" AND "TARGET_AS_LIST" OBJECTS ARE NOT CUT FROM THE SAME CLOTH')

                    if self.TARGET is None and not self.TARGET_TRANSPOSE is None and not self.TARGET_AS_LIST is None:
                        if self.TARGET_TRANSPOSE.current_orientation == self.TARGET_AS_LIST.current_orientation and not \
                            self.TARGET_TRANSPOSE.is_equiv(self.TARGET_AS_LIST.get_transpose()):
                            _exception(fxn, f'RETURNED "TARGET_TRANSPOSE" AND "TARGET_AS_LIST" OBJECTS ARE NOT CUT FROM THE SAME CLOTH')
                        elif self.TARGET_TRANSPOSE.current_orientation != self.TARGET_AS_LIST.current_orientation and not \
                            self.TARGET_TRANSPOSE.is_equiv(self.TARGET_AS_LIST.OBJECT):
                            _exception(fxn, f'RETURNED "TARGET_TRANSPOSE" AND "TARGET_AS_LIST" OBJECTS ARE NOT CUT FROM THE SAME CLOTH')
                    # END TEST ################################################################################################

                del more_than_one_target_object_given, num_target_objects_to_be_returned

                # END TARGET INPUT VALIDATION #################################################################################
                ###############################################################################################################

                ###############################################################################################################
                # DATA INPUT VALIDATION #######################################################################################

                # IF ONE OBJECT IS RETURNED, CANNOT VALIDATE. IF NO DATA RETURN(S) REQUIRED, DO NOT NEED TO VALIDATE
                if num_data_objects_to_be_returned < 2:
                    pass
                else:
                    # TEST ####################################################################################################
                    if not self.DATA is None and not self.DATA_TRANSPOSE is None:
                        # HAVE TO STANDARDIZE ORIENTATION BEFORE DOING equiv TEST
                        if self.DATA.return_orientation == self.DATA_TRANSPOSE.return_orientation and not \
                                self.DATA.is_equiv(self.DATA_TRANSPOSE.get_transpose()):
                            _exception(fxn, f'GIVEN "DATA" AND "DATA_TRANSPOSE" OBJECTS ARE NOT CUT FROM THE SAME CLOTH')
                        elif self.DATA.return_orientation != self.DATA_TRANSPOSE.return_orientation and not \
                                self.DATA.is_equiv(self.DATA_TRANSPOSE.OBJECT):
                            _exception(fxn, f'GIVEN "DATA" AND "DATA_TRANSPOSE" OBJECTS ARE NOT CUT FROM THE SAME CLOTH')
                    # END TEST ################################################################################################

                del more_than_one_data_object_given, num_data_objects_to_be_returned

                # END DATA INPUT VALIDATION ###################################################################################
                ###############################################################################################################

                ###############################################################################################################
                # XTX / XTX_INV INPUT VALIDATION ##############################################################################
                # IF ONE OBJECT IS RETURNED, CANNOT VALIDATE. IF NO XTX RETURN(S) REQUIRED, DO NOT NEED TO VALIDATE
                if num_xtx_objects_to_be_returned < 2:
                    pass
                else:
                    # TEST ####################################################################################################
                    if not np.allclose(np.linalg.inv(self.XTX.return_as_array()),
                                          self.XTX_INV.return_as_array(),
                                          atol=1e-10,
                                          rtol=1e-10):
                        _exception(fxn, f'INVERSE OF GIVEN "XTX" AND GIVEN "XTX_INV" OBJECTS ARE NOT EQUAL')
                    # END TEST ################################################################################################

                del more_than_one_xtx_object_given, num_xtx_objects_to_be_returned

                # END XTX / XTX_INV INPUT VALIDATION ##########################################################################
                ###############################################################################################################

                # #############################################################################################################
                # DATA / XTX CROSS VALIDATION #################################################################################
                # DATA SHOULD ALREADY BE VERIFIED AGAINST DATA_TRANSPOSE, XTX SHOULD ALREADY BE VERIFIED AGAINST XTX_INV
                # SO SHOULD ONLY HAVE TO VERIFY ONE OF [DATA, DATA_TRANSPOSE] AGAINST ONE OF [XTX, XTX_INV]

                if not self.DATA is None:
                    if not self.XTX is None:
                        if not np.allclose(self.DATA.return_XTX(return_format='ARRAY'),
                                          self.XTX.return_as_array(),
                                          atol=1e-10,
                                          rtol=1e-10
                                          ):
                            _exception(fxn, f'GIVEN DATA OBJECT AND GIVEN XTX OBJECT ARE NOT CUT FROM THE SAME CLOTH')
                    elif not self.XTX_INV is None:
                        if not np.allclose(self.DATA.return_XTX_INV(return_format='ARRAY'),
                                           self.XTX_INV.return_as_array(),
                                           atol=1e-10,
                                           rtol=1e-10):
                            _exception(fxn, f'GIVEN DATA OBJECT AND GIVEN XTX_INV OBJECT ARE NOT CUT FROM THE SAME CLOTH')

                elif not self.DATA_TRANSPOSE is None:
                    if not self.XTX is None:
                        if not np.allclose(self.DATA_TRANSPOSE.return_XTX(return_format='ARRAY'),
                                           self.XTX.return_as_array(),
                                           atol=1e-10,
                                           rtol=1e-10):
                            _exception(fxn, f'GIVEN DATA_TRANSPOSE OBJECT AND GIVEN XTX OBJECT ARE NOT CUT FROM THE SAME CLOTH')
                    elif not self.XTX_INV is None:
                        if not np.allclose(self.DATA_TRANSPOSE.return_XTX_INV(return_format='ARRAY'),
                                           self.XTX_INV.return_as_array(),
                                           atol=1e-10,
                                           rtol=1e-10):
                            _exception(fxn, f'GIVEN DATA_TRANSPOSE OBJECT AND GIVEN XTX_INV OBJECT ARE NOT CUT FROM THE SAME CLOTH')

                # END DATA / XTX CROSS VALIDATION #################################################################################
                # #############################################################################################################

            # END FIFTH CONDITIONAL VALIDATION, OUTPUT OBJECTS #################################################################################
            ###############################################################################################################################


            # **** THIS IS ACCESSED WHETHER IN SHORT-CIRCUIT MORE OR FULL PROCESS MODE *********************************************
            # IF DATA AND DATA_TRANSPOSE FULL CLASSES WERE KEPT AROUND, MANAGE THAT NOW ###############################################
            # DONT MOVE THESE TO AFTER "if DATA:" OR AFTER "if DATA_TRANSPOSE:" ABOVE, CANT SET THESE TO None UNTIL AFTER XTX IS DONE
            if not self.DATA is None:
                if 'DATA' not in RETURN_OBJECTS:
                    self.DATA = None
                elif 'DATA' in RETURN_OBJECTS:
                    self.data_current_format = self.DATA.current_format
                    self.data_return_format = self.DATA.return_format
                    self.data_current_orientation = self.DATA.current_orientation
                    self.data_return_orientation = self.DATA.return_orientation
                    self.DATA = self.DATA.OBJECT


            if not self.DATA_TRANSPOSE is None:
                if 'DATA_TRANSPOSE' not in RETURN_OBJECTS:
                    self.DATA_TRANSPOSE = None
                elif 'DATA_TRANSPOSE' in RETURN_OBJECTS:
                    self.data_transpose_current_format = self.DATA_TRANSPOSE.current_format
                    self.data_transpose_return_format = self.DATA_TRANSPOSE.return_format
                    self.data_transpose_current_orientation = self.DATA_TRANSPOSE.current_orientation
                    self.data_transpose_return_orientation = self.DATA_TRANSPOSE.return_orientation
                    self.DATA_TRANSPOSE = self.DATA_TRANSPOSE.OBJECT

            # END FINAL MANAGE OF DATA AND DATA_TRANSPOSE #############################################################################

            # IF XTX FULL CLASS WAS KEPT AROUND TO BUILD XTX_INV, MANAGE THAT NOW, MAKE SD IF WAS LEFT AS ARRAY FOR XTX_INV ######
            # MAKE FINAL CONVERSION OF XTX WHEN WITHHELD AS ARRAY TO BUILD XTX_INV
            if not self.XTX is None:
                if 'XTX' not in RETURN_OBJECTS:
                    self.XTX = None
                elif 'XTX' in RETURN_OBJECTS:
                    if xtx_return_format == 'SPARSE_DICT':
                        self.XTX.to_dict()
                    self.xtx_current_format = self.XTX.current_format
                    self.xtx_return_format = self.XTX.return_format
                    if self.XTX.current_format != xtx_return_format:
                        self.XTX.to_dict()
                        self.XTX = self.XTX.OBJECT
                    else: self.XTX =  self.XTX.OBJECT
            # END FINAL MANAGE OF XTX ############################################################################################

            # MANAGE XTX_INV ###################################################################################################
            if not self.XTX_INV is None:
                if 'XTX_INV' not in RETURN_OBJECTS:
                    self.XTX_INV = None
                elif 'XTX_INV' in RETURN_OBJECTS:
                    self.xtx_inv_current_format = self.XTX_INV.current_format
                    self.xtx_inv_return_format = self.XTX_INV.return_format
                    self.XTX_INV = self.XTX_INV.OBJECT
            # END FINAL MANAGE OF XTX_INV ######################################################################################

            # MANAGE TARGET ###################################################################################################
            self.target_is_multiclass = target_is_multiclass
            if not self.TARGET is None:
                if 'TARGET' not in RETURN_OBJECTS:
                    self.TARGET = None
                elif 'TARGET' in RETURN_OBJECTS:
                    self.target_current_format = self.TARGET.current_format
                    self.target_return_format = self.TARGET.return_format
                    self.target_current_orientation = self.TARGET.current_orientation
                    self.target_return_orientation = self.TARGET.return_orientation
                    self.TARGET = self.TARGET.OBJECT
            # END FINAL MANAGE OF TARGET ######################################################################################

            # MANAGE TARGET_TRANSPOSE #########################################################################################
            if not self.TARGET_TRANSPOSE is None:
                if 'TARGET_TRANSPOSE' not in RETURN_OBJECTS:
                    self.TARGET_TRANSPOSE = None
                elif 'TARGET_TRANSPOSE' in RETURN_OBJECTS:
                    self.target_transpose_current_format = self.TARGET_TRANSPOSE.current_format
                    self.target_transpose_return_format = self.TARGET_TRANSPOSE.return_format
                    self.target_transpose_current_orientation = self.TARGET_TRANSPOSE.current_orientation
                    self.target_transpose_return_orientation = self.TARGET_TRANSPOSE.return_orientation
                    self.TARGET_TRANSPOSE =  self.TARGET_TRANSPOSE.OBJECT
            # END FINAL MANAGE OF TARGET_TRANSPOSE ######################################################################################

            # MANAGE TARGET_AS_LIST ############################################################################################
            if not self.TARGET_AS_LIST is None:
                if 'TARGET_AS_LIST' not in RETURN_OBJECTS:
                    self.TARGET_AS_LIST = None
                elif 'TARGET_AS_LIST' in RETURN_OBJECTS:
                    self.target_as_list_current_format = self.TARGET_AS_LIST.current_format
                    self.target_as_list_return_format = self.TARGET_AS_LIST.return_format
                    self.target_as_list_current_orientation = self.TARGET_AS_LIST.current_orientation
                    self.target_as_list_return_orientation = self.TARGET_AS_LIST.return_orientation
                    self.TARGET_AS_LIST = self.TARGET_AS_LIST.OBJECT
            # END FINAL MANAGE OF TARGET_AS_LIST ######################################################################################

            self.RETURN_OBJECT = RETURN_OBJECTS

            del _exception

            break





















































if __name__ == '__main__':

    # TESTING IS DONE IN THREE SEPARATE MODULES IN MLObjectOrienter__test
    # THAT TEST DATA/DATA_TRANSPOSE/XTX/XTX_INV, XTX/XTX_INV, & TARGET/TARGET_TRANSPOSE/TARGET_AS_LIST SEPARATELY.

    pass

























































