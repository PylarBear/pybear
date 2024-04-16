import sys
import numpy as np
import sparse_dict as sd
from MLObjects.SupportObjects import ApexSupportObjectHandling as asoh
from data_validation import arg_kwarg_validater as akv
from ML_PACKAGE._data_validation import list_dict_validater as ldv
from debug import get_module_name as gmn
from MLObjects.SupportObjects.master_support_object_dict import val_text_dtypes as vtd, val_num_dtypes as vnd, \
                    mod_text_dtypes as mtd, mod_num_dtypes as mnd, empty_value as ev


# CHILD OF Apex, (WAS CHILD OF Header, BUT WANT TO INGEST OBJECT_HEADER AND Header DOESNT ALLOW IT (INGEST HEADER AS SUPOBJ IN Header))
class Filtering(asoh.ApexSupportObjectHandle):

    def __init__(self,
                 OBJECT=None,
                 object_given_orientation='COLUMN',
                 columns=None,
                 OBJECT_HEADER=None,
                 SUPPORT_OBJECT=None,
                 # prompt_to_override=False,  DONT ALLOW OVERRIDE, ONLY ALLOW ENTRIES BY FILTERING MECHANISM
                 return_support_object_as_full_array=True,
                 bypass_validation=False,
                 calling_module=None,
                 calling_fxn=None
                 ):

        self.this_module = gmn.get_module_name(str(sys.modules[__name__]))
        fxn = '__init__'
        self.calling_module = calling_module if not calling_module is None else self.this_module
        self.calling_fxn = calling_fxn if not calling_fxn is None else fxn

        # REQUIRING THAT FILTERING_SUP_OBJ BE PASSED TO super().__init__() AS PART OF FULLY SIZED SUPPORT_OBJECT.

        # 1/3/23 AFTER HRS OF TRYING, THERE IS NO WAY TO GET ndarray([list[]), list([]),...]) TO GO TO
        # ndarray([[list[]), list([]),...]]) WHICH WOULD BE NECESSARY FOR Apex init TO DEAL WITH FILTERING IN THE SAME
        # WAY AS ALL THE OTHERS.  MUST BE [[]] BECAUSE OF THE WAY Apex ASSIGNS AN INDEX TO THE SUPPORT_OBJECT FOR EDITING,
        # SO THAT THE EDITING OF A SINGLE SUPPORT_OBJECT IS STANDARDIZED WITH THE WAY FOR A FULL SUPPORT_OBJECT.
        # IF PASSED A [], ASSIGNING AN INDEX WOULD CAUSE IT TO EDIT INSIDE THE [].  🤬

        if not SUPPORT_OBJECT is None and not isinstance(SUPPORT_OBJECT, (np.ndarray, list, tuple)):
            raise Exception(f'{self.this_module}.{fxn}() >>> GIVEN SUPPORT_OBJECT MUST BE A LIST-TYPE THAT '
                            f'CAN BE CONVERTED TO AN NP ARRAY, OR None')

        while True:
            # 1) IF SUPPORT_OBJECT IS None, JUST LET PASS THRU TO super(), Apex WILL BUILD EMPTY SUPOBJ BASED ON OBJECT, HEADER, OR columns
            if SUPPORT_OBJECT is None: turn_to_full = False; break
            # 2) IF SUPOBJ IS FULL, ALL INNER SUPOBJS MUST ALWAYS BE EQUAL LEN, SO IF OBJECT DOES NOT HAVE EQUAL INNER LENS,
            # MUST BE A SINGLE FILTERING OBJ, WHICH MUST BE TURNED TO FULL
            _supobj_lens = list(map(len, SUPPORT_OBJECT))
            if min(_supobj_lens) != max(_supobj_lens): turn_to_full = True; del _supobj_lens; break
            # 3) IF ALL INNERS HAVE EQUAL LEN, AND INNER LEN IS > 0, THEN IF A FULL SUPOBJ SHOULD HAVE DTYPES LIKE A FULL SUPOBJ, OR
            #       BE FILLED W self.empty_value
            if min(_supobj_lens) >= 1:
                # JUMP THRU HOOPS TO GET VAL_ALLOWED AND MOD_ALLOWED BEFORE FULL init OF PARENT AT THE END OF THIS init
                self.VAL_ALLOWED_LIST = list(vtd().values()) + list(vnd().values())
                self.MOD_ALLOWED_LIST = list(mtd().values()) + list(mnd().values())
                self.empty_value = ev()

                DICT = {key: self.master_dict()[key]['position'] for key in self.master_dict()}
                # THIS IS JUST AN ARBITRARY SAMPLE OF SUPOBJS
                _start_lag_idx = DICT['STARTLAG']
                _end_lag_idx = DICT['ENDLAG']
                _scaling_idx = DICT['SCALING']
                _useother_idx = DICT['USEOTHER']
                # tm = _type_matcher       ALLOW FOR FILLED dtype OR empty_value IF NOT FILLED.
                # A SINGLE FILTERING WHERE INNER LISTS ARE EQUAL len AND len >= 1 COULD ONLY HAVE A str type
                _tm = lambda _idx, expected_type: int(type(SUPPORT_OBJECT[_idx][0])==expected_type) + int(SUPPORT_OBJECT[_idx][0]==self.empty_value)

                if _tm(_start_lag_idx, int) and _tm(_end_lag_idx, int) and _tm(_scaling_idx, str) and _tm(_useother_idx, str):
                    # IF TYPES MATCH, THEN THIS IS A FULLY SIZED SUPPORT OBJECT
                    turn_to_full = False; del _supobj_lens, DICT, _start_lag_idx, _end_lag_idx, _scaling_idx, _useother_idx, _tm
                    break
            # NOW KNOW SUPPORT_OBJECT not None, SUPOBJ INNER LENS EQUAL, COMPOSITION IF INNER LEN >= 1
            # WHAT IF INNER LEN == 0 COULD BE AN EMPTY SINGLE FILTERING OR FULL SUPOBJ FOR ZERO COLUMNS

            # 4) IF SUPOBJ INNERS ALL ZERO len
            # SEE IF CAN GET EXPECTED shape OF SUPOBJ FROM OBJECT, IF GIVEN
            given_format, OBJECT = ldv.list_dict_validater(OBJECT, 'OBJECT')

            if OBJECT is None: pass
            else: # IF OBJECT IS GIVEN, GET COLUMNS TO FIND OUT WHAT SIZE SUPPORT_OBJECT SHOULD BE
                object_given_orientation = akv.arg_kwarg_validater(object_given_orientation, f'object_given_orientation',
                                       ['ROW', 'COLUMN'], self.this_module, fxn)
                _ = object_given_orientation
                if given_format == 'ARRAY':
                    if _ == 'COLUMN': _columns = len(OBJECT)
                    elif _ == 'ROW': _columns = len(OBJECT[0])
                elif given_format == 'SPARSE_DICT':
                    if _ == 'COLUMN': _columns = sd.outer_len(OBJECT)
                    elif _ == 'ROW': _columns = sd.inner_len_quick(OBJECT)
                del _

                if len(SUPPORT_OBJECT) == _columns:  # THEN MUST BE AN EMPTY SINGLE FILTERING
                    turn_to_full = True; del given_format, _columns; break
                elif len(SUPPORT_OBJECT) == len(self.master_dict()):   # THEN MUST BE A FULLY SIZED SUPOBJ FOR ZERO COLUMNS
                    turn_to_full = False; del given_format, _columns; break
                else:  # IF DOESNT MATCH ANY OF THE DIMENSIONS, SUPOBJ HAS INVALID SIZE
                    raise Exception(f'{self.this_module}.{fxn}() >>> GIVEN SUPPORT_OBJECT HAS INVALID DIMENSIONS W-R-T GIVEN OBJECT')

            # 5) SUPOBJ INNERS ALL ZERO len
            # IF OBJECT NOT GIVEN, SEE IF CAN GET EXPECTED shape OF SUPOBJ FROM HEADER, IF GIVEN
            header_format, OBJECT_HEADER = ldv.list_dict_validater(OBJECT_HEADER, 'HEADER')
            if not header_format in [None, 'ARRAY']:
                raise Exception(f'{self.this_module}.{fxn}() >>> GIVEN OBJECT_HEADER MUST BE A LIST-TYPE THAT '
                                        f'CAN BE CONVERTED TO AN NP ARRAY, OR None')
            if not OBJECT_HEADER is None:
                _columns = len(OBJECT_HEADER[0])
                if len(SUPPORT_OBJECT) == _columns:  # THEN MUST BE AN EMPTY SINGLE FILTERING
                    turn_to_full = True; del header_format, _columns; break
                elif len(SUPPORT_OBJECT) == len(self.master_dict()) and _columns==0:   # THEN MUST BE A FULLY SIZED SUPOBJ FOR ZERO COLUMNS
                    turn_to_full = False; del header_format, _columns; break
                else:  # IF DOESNT MATCH ANY OF THE DIMENSIONS, SUPOBJ HAS INVALID SIZE
                    raise Exception(f'{self.this_module}.{fxn}() >>> GIVEN SUPPORT_OBJECT HAS INVALID DIMENSIONS W-R-T GIVEN HEADER')

            elif OBJECT_HEADER is None:  # A STICKY WICKET, NO OBJECT, NO HEADER, INNERS ARE ALL EQUAL LEN
                if len(SUPPORT_OBJECT) != len(self.master_dict()):
                    # FOR ALL CASES WHERE len([[],[],[]...]) != len(master_dict()) THE INTENT MUST BE AN EMPTY SINGLE WHERE [] REPRESENTS COLUMNS
                    turn_to_full = True; break
                elif len(SUPPORT_OBJECT) == len(self.master_dict()):
                    # FOR THE OBSCURE CASE THAT len([[],[],[]...]) == len(master_dict()), WITH NO OTHER OBJECTS GIVEN, ASSUME
                    # THAT INTENT IS A SINGLE FILTERING SUPOBJ WITH len(master_dict()) COLUMNS. WHAT ARE THE CHANCES A FULLY
                    # SIZED SUPOBJ IS PASSED WITH NO OTHER OBJECTS, WITH THE INTENT TO HAVE ZERO COLUMNS? SLIM.
                    turn_to_full = True; break

            # IF GET HERE, DEALING WITH A DEGENERATE SITUTATION
            raise Exception(f'{self.this_module}.{fxn}() >>> degenerate situation trying to determine if given SUPOBJ is'
                            f'a fully sized SUPOBJ or a single Filtering vector.')

        if not turn_to_full:  # SUPOBJ IS ALREADY FULL OR IS None, SO JUST PASS TO SUPER
            pass
        elif turn_to_full:
            EMPTY_FULL = np.full((len(self.master_dict()), len(SUPPORT_OBJECT)))
            EMPTY_FULL[self.master_dict()['FILTERING']['position']] = SUPPORT_OBJECT
            SUPPORT_OBJECT = EMPTY_FULL
            del EMPTY_FULL

        del turn_to_full

        super().__init__(
                OBJECT=OBJECT,
                object_given_orientation=object_given_orientation,
                columns=columns,
                OBJECT_HEADER=OBJECT_HEADER,
                SUPPORT_OBJECT=SUPPORT_OBJECT,
                prompt_to_override=False,   # DONT ALLOW OVERRIDE, ONLY ALLOW ENTRIES BY FILTERING MECHANISM
                return_support_object_as_full_array=return_support_object_as_full_array,
                bypass_validation=bypass_validation,
                calling_module=self.calling_module,
                calling_fxn=self.calling_fxn
                )

    # END __init__ ###########################################################################################################
    ##########################################################################################################################
    ##########################################################################################################################
    ##########################################################################################################################

    # INHERITS
    # _exception()
    # len_validation()
    # validate_allowed()
    # empty()
    # single_edit()
    # full_edit()
    # delete_idx()
    # insert_idx()
    # manual_fill_all()
    # default_fill()

    ##########################################################################################################################
    # OVERWRITTEN ############################################################################################################

    # # INHERITED FROM Apex
    # def set_active_index(self):
    #     '''Allows support object to be uniquely set in children to a single object or row of a larger object. '''
    #     self.actv_idx = self.master_dict()[self.this_module.upper()]['position']

    def support_name(self):
        '''Name of child's support object.'''
        # OVERWRITE IN CHILD
        return 'FILTERING'

    def default_value(self):
        '''Default value to fill support object.'''
        # OVERWRITE IN CHILD
        return []

    def allowed_values(self):
        'Allowed values for validation.'''
        return (np.ndarray, list, tuple)

    def autofill(self):
        '''Unique method to fill particular support object.'''
        self.default_fill()


    # INHERITED FROM Apex
    # def validate_allowed(self, kill=None, fxn=None):


    # INHERITED FROM Apex
    # def validate_against_objects(self, OBJECT=None, kill=None, fxn=None):
    #     # FILTERING CAN HAPPEN ON ANY COLUMN. CANT THINK OF ANY OTHER VALIDATION BESIDES validate_allowed ABOVE.
    #     pass

    # END OVERWRITTEN ########################################################################################################
    ##########################################################################################################################
    ##########################################################################################################################























































