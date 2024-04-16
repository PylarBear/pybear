import sys, inspect
import numpy as np
from MLObjects.SupportObjects import ApexSupportObjectHandling as asoh, ValidatedDatatypes as vd
from data_validation import arg_kwarg_validater as akv, validate_user_input as vui
from ML_PACKAGE._data_validation import list_dict_validater as ldv
from debug import get_module_name as gmn



# CHILD OF Apex, CANNOT BE CHILD OF ValidatedDatatypes, IS CALLED THERE TO VALIDATE KWARG MODIFIED_DATATYPES IF PASSED, WOULD BE CIRCULAR
class ModifiedDatatypes(asoh.ApexSupportObjectHandle):

    def __init__(self,
                 OBJECT=None,
                 object_given_orientation=None,
                 columns=None,
                 OBJECT_HEADER=None,
                 SUPPORT_OBJECT=None,
                 VALIDATED_DATATYPES=None,
                 prompt_to_override=False,
                 return_support_object_as_full_array=True,
                 bypass_validation=False,
                 calling_module=None,
                 calling_fxn=None
                 ):

        self.this_module = gmn.get_module_name(str(sys.modules[__name__]))
        fxn = '__init__'
        self.calling_module = calling_module if not calling_module is None else self.this_module
        self.calling_fxn = calling_fxn if not calling_fxn is None else fxn

        self.bypass_validation = akv.arg_kwarg_validater(bypass_validation, 'bypass_validation', [True, False, None],
                                                         self.this_module, fxn, return_if_none=False)

        self.VDTYPES_is_empty=True    # self.VDTYPES_is_empty IS FOR VALIDATED_DATATYPES, self.vdtypes_is_empty IS FOR
        # VDTYPES GIVEN VIA FULL SUPPORT_OBJECT ---- self.vdtypes_is_empty WILL BE GOTTEN BY super().__init__() LATER
        if self.bypass_validation:
            if VALIDATED_DATATYPES is None:
                self.VALIDATED_DATATYPES = VALIDATED_DATATYPES
                # self.VDTYPES_is_empty STAYS True
            if not VALIDATED_DATATYPES is None:
                self.VALIDATED_DATATYPES = np.array(VALIDATED_DATATYPES).reshape((1,-1))[0]
                self.VDTYPES_is_empty = self.is_empty_getter(self.VALIDATED_DATATYPES)

        elif not self.bypass_validation:
            vdtypes_type, self.VALIDATED_DATATYPES = ldv.list_dict_validater(VALIDATED_DATATYPES, 'VALIDATED_DATATYPES')
            if vdtypes_type not in [None, 'ARRAY']: self._exception(fxn,
                                        f'VALIDATED_DATATYPES MUST BE A LIST-TYPE THAT CAN BE CONVERTED TO AN NP ARRAY, OR None')

            if vdtypes_type is None:
                # self.VDTYPES_is_empty STAYS True
                # self.VALIDATED_DATATYPES STAYS None
                pass
            elif not vdtypes_type is None:
                self.VALIDATED_DATATYPES = self.VALIDATED_DATATYPES[0]

                if len(self.VALIDATED_DATATYPES) != self.columns:
                    self._exception(fxn, f'len(VALIDATED_DATATYPES) != columns')

                VDTYPE_VALIDATOR = vd.ValidatedDatatypes(OBJECT=OBJECT,
                                         object_given_orientation=object_given_orientation,
                                         columns=columns,
                                         OBJECT_HEADER=OBJECT_HEADER,
                                         SUPPORT_OBJECT=VALIDATED_DATATYPES,
                                         prompt_to_override=prompt_to_override,
                                         return_support_object_as_full_array=False,
                                         bypass_validation=self.bypass_validation,
                                         calling_module=gmn.get_module_name(str(sys.modules[__name__])),
                                         calling_fxn=fxn)

                VDTYPE_VALIDATOR.validate_allowed(kill=not prompt_to_override, fxn=fxn)

                self.VDTYPES_is_empty = VDTYPE_VALIDATOR.vdtypes_is_empty

                del VDTYPE_VALIDATOR

            del vdtypes_type


        # MUST BE LAST BECAUSE OF autofill()
        super().__init__(
                        OBJECT=OBJECT,
                        object_given_orientation=object_given_orientation,
                        columns=columns,
                        OBJECT_HEADER=OBJECT_HEADER,
                        SUPPORT_OBJECT=SUPPORT_OBJECT,
                        prompt_to_override=prompt_to_override,
                        return_support_object_as_full_array=return_support_object_as_full_array,
                        bypass_validation=bypass_validation,
                        calling_module=self.calling_module,
                        calling_fxn=self.calling_fxn
                        )

    # END __init__ ###########################################################################################################
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
    ##########################################################################################################################
    # OVERWRITTEN IN CHILDREN ################################################################################################

    # INHERITED FROM Apex
    # def set_active_index(self):
    #     '''Allows support object to be uniquely set in children to a single object or row of a larger object. '''
    #     self.actv_idx = self.master_dict()[self.this_module.upper()]['position']

    def support_name(self):
        '''Name of child's support object.'''
        # OVERWRITE IN CHILD
        return 'MODIFIEDDATATYPES'

    def default_value(self):
        '''Default value to fill support object.'''
        # OVERWRITE IN CHILD
        return 'STR'



    def allowed_values(self):
        'Allowed values for validation.'''
        # OVERWROTE IN CHILD
        self.TEXT_DTYPES = self.MOD_TEXT_DTYPES
        self.NUM_DTYPES = self.MOD_NUM_DTYPES
        self.NUM_DTYPES_LIST = self.MOD_NUM_DTYPES_LIST
        self.ALLOWED_DICT = self.MOD_ALLOWED_DICT
        self.ALLOWED_LIST = self.MOD_ALLOWED_LIST
        self.menu_allowed_cmds = self.mod_menu_allowed_cmds
        self.MENU_OPTIONS_PROMPT = self.MOD_MENU_OPTIONS_PROMPT
        self.REVERSE_LOOKUP = self.VAL_REVERSE_LOOKUP


    # INHERITED FROM Apex
    # def validate_against_objects(self, OBJECT=None, kill=None, fxn=None):
    #     pass


    def autofill(self):
        '''Unique method to fill particular support object.'''

        fxn = inspect.stack()[0][3]

        # ONLY WAY COULD GET HERE IS OBJECT IS GIVEN AND SUPPORT_OBJECT IS FULL OF empty_value OR IS None

        # START - MAKE MODDTYPES IN SUPPORT_OBJECT HOLD VALDTYPES, SO THAT CAN BE OVERWRITTEN W USER MODDTYPES ############################

        # IF USER GIVES VALIDATED_DATATYPES AS KWARG, ASSUME INTENT IS TO OVERWRITE ANY EXISTING VDTYPES ALREADY IN SUPPORT_OBJECT (IF FULL)
        if (not self.VDTYPES_is_empty and not self.vdtypes_is_empty) or \
                (not self.VDTYPES_is_empty and self.vdtypes_is_empty):

            self.SUPPORT_OBJECT[self.actv_idx] = self.VALIDATED_DATATYPES.copy()
            self.mdtypes_is_empty = False

            if self.supobj_is_full:
                self.SUPPORT_OBJECT[self.master_dict()['VALIDATEDDATATYPES']['position']] = self.VALIDATED_DATATYPES.copy()
                self.vdtypes_is_empty = False

        elif self.VDTYPES_is_empty and not self.vdtypes_is_empty:
            # VALIDATED_DATATYPES NOT GIVEN AS A KWARG, BUT VDTYPES GIVEN IN SUPPORT_OBJECT
            self.SUPPORT_OBJECT[self.actv_idx] = self.SUPPORT_OBJECT[self.master_dict()['VALIDATEDDATATYPES']['position']].copy()

        elif self.vdtypes_is_empty and self.VDTYPES_is_empty:
            # vdtypes() AUTOMATICALLY WRITES TO self.active_idx (FOR THIS MODULE IS MODIFIEDDATATYPES idx
            self.vdtypes(quick=False, print_notes=False)  # not self.prompt_to_override)
            if self.supobj_is_full and self.vdtypes_is_empty:
                self.SUPPORT_OBJECT[self.master_dict()['VALIDATEDDATATYPES']['position']] = self.SUPPORT_OBJECT[self.actv_idx].copy()

        # END - MAKE MODDTYPES IN SUPPORT_OBJECT HOLD VALDTYPES, SO THAT CAN BE OVERWRITTEN W USER MODDTYPES ############################

        if self.prompt_to_override:
            if vui.validate_user_str(f'Override MOD DTYPES (currently defaulted to VAL DTYPES)? (y/n) > ', 'YN') == 'Y':

                # ITERATE THRU OBJECT, FIND TYPE OF DATA, THEN ALLOW USER TO DECLARE HOW TO DEAL WITH A STR, INT, & FLOAT

                SUPPORT_OBJECT_RESTORE = self.SUPPORT_OBJECT.copy()

                while True:
                    for col_idx in range(self.columns):
                        type = self.SUPPORT_OBJECT[self.actv_idx][col_idx]
                        if type in ['STR']:
                            new_type = self.ALLOWED_DICT[
                                vui.validate_user_str(
                                    f'CURRENT VALIDATED DATATYPE FOR {self.QUICK_HEADER[0][col_idx][:50]} IS '
                                    f'{type}. TREAT AS {", ".join([f"{v}({k})" for k,v in self.TEXT_DTYPES.items()])} > ',
                                    "".join(list(self.TEXT_DTYPES.keys()))
                                )
                            ]
                            self.SUPPORT_OBJECT[self.actv_idx][col_idx] = new_type; del new_type
                            self.OBJECT[col_idx] = self.OBJECT[col_idx].astype(str)
                        elif type in ['INT', 'FLOAT', 'BIN']:
                            while True:
                                try:
                                    new_type = self.ALLOWED_DICT[
                                        vui.validate_user_str(
                                            f'CURRENT VALIDATED DATATYPE FOR {self.QUICK_HEADER[0][col_idx][:50]} IS {type}. '
                                            f'TREAT AS {", ".join([f"{v}({k})" for k,v in self.NUM_DTYPES.items()]+["STR(S}"])} > ',
                                            "".join(list(self.NUM_DTYPES.keys())+["S"])
                                        )
                                    ]

                                    if new_type == 'STR': self.OBJECT[col_idx] = self.OBJECT[col_idx].astype(str)
                                    elif new_type == 'FLOAT': self.OBJECT[col_idx] = self.OBJECT[col_idx].astype(np.float64)
                                    elif new_type == 'INT': self.OBJECT[col_idx] = self.OBJECT[col_idx].astype(np.int32)
                                    elif new_type == 'BIN': self.OBJECT[col_idx] = self.OBJECT[col_idx].astype(np.int8)

                                    break

                                except:
                                    print(f'\n{self.QUICK_HEADER[0][col_idx]} raised exception with {new_type}.  Try again.')
                                    continue

                            self.SUPPORT_OBJECT[self.actv_idx][col_idx] = new_type; del new_type

                    self.view_support_objects()

                    if vui.validate_user_str(f'\nAccept MODIFIED TYPES? (y/n) > ', 'YN') == 'Y':
                        del SUPPORT_OBJECT_RESTORE
                        break
                    else:
                        print(f'\nRETURNING USER DATATYPES TO PRIOR STATE AND RESTARTING')
                        self.SUPPORT_OBJECT = SUPPORT_OBJECT_RESTORE.copy()


    # END OVERWRITTEN ########################################################################################################
    ##########################################################################################################################
    ##########################################################################################################################












