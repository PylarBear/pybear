import sys, inspect
import numpy as np
from MLObjects.SupportObjects import ApexSupportObjectHandling as asoh, ModifiedDatatypes as md, master_support_object_dict as msod
from data_validation import arg_kwarg_validater as akv
from ML_PACKAGE._data_validation import list_dict_validater as ldv
from debug import get_module_name as gmn


# CHILD OF Apex
# PARENT OF UseOther
class ValidatedDatatypes(asoh.ApexSupportObjectHandle):

    def __init__(self,
                 OBJECT=None,
                 object_given_orientation='COLUMN',
                 columns=None,
                 OBJECT_HEADER=None,
                 SUPPORT_OBJECT=None,
                 MODIFIED_DATATYPES=None,   # CAN REVERSE ENGINEER VALIDATED_DATATYPES IF MOD_DTYPES IS GIVEN
                 quick_vdtypes=False,
                 prompt_to_override=False,
                 return_support_object_as_full_array=True,
                 bypass_validation=False,
                 print_notes=False,
                 calling_module=None,
                 calling_fxn=None
                 ):

        self.this_module = gmn.get_module_name(str(sys.modules[__name__]))
        fxn = '__init__'
        self.calling_module = calling_module if not calling_module is None else self.this_module
        self.calling_fxn = calling_fxn if not calling_fxn is None else fxn

        self.bypass_validation = akv.arg_kwarg_validater(bypass_validation, 'bypass_validation', [True, False, None],
                                                         self.this_module, fxn, return_if_none=False)

        # self.MDTYPES_is_empty IS FOR MODIFIED_DATATYPES, self.mdtypes_is_empty IS FOR MDTYPES GIVEN AS PART OF SUPPORT_OBJECT
        self.MDTYPES_is_empty = True    # self.mdtypes_is_empty WILL BE GOTTEN BY super().__init__()
        if self.bypass_validation:
            self.quick_vdtypes = quick_vdtypes
            self.print_notes = print_notes

            if MODIFIED_DATATYPES is None:
                self.MODIFIED_DATATYPES = MODIFIED_DATATYPES
                # self.MDTYPES_is_empty STAYS True
            elif not MODIFIED_DATATYPES is None:
                self.MODIFIED_DATATYPES = np.array(MODIFIED_DATATYPES).reshape((1, -1))[0]
                UNIQUES = np.unique(self.MODIFIED_DATATYPES)
                self.MDTYPES_is_empty = True if len(UNIQUES) == 1 and UNIQUES[0] == msod.empty_value() else False
                del UNIQUES

        elif not self.bypass_validation:

            self.quick_vdtypes = akv.arg_kwarg_validater(quick_vdtypes, 'quick_vdtypes', [True, False, None], self.this_module,
                                                        fxn, return_if_none=False)
            self.print_notes = akv.arg_kwarg_validater(print_notes, 'print_notes', [True, False, None], self.this_module,
                                                        fxn, return_if_none=False)

            mdtypes_type, self.MODIFIED_DATATYPES = ldv.list_dict_validater(MODIFIED_DATATYPES, 'MODIFIED_DATATYPES')
            if mdtypes_type not in [None, 'ARRAY']: self._exception(fxn,
                                f'MODIFIED_DATATYPES MUST BE A LIST-TYPE THAT CAN BE CONVERTED TO AN NP ARRAY, OR None')

            if mdtypes_type is None:
                # self.MDTYPES_is_empty STAYS True
                # self.MODIFIED_DATATYPES STAYS None
                pass
            elif not mdtypes_type is None:
                self.MODIFIED_DATATYPES = self.MODIFIED_DATATYPES[0]

                MDTYPE_VALIDATOR = md.ModifiedDatatypes(OBJECT=OBJECT,
                                         object_given_orientation=object_given_orientation,
                                         columns=columns,
                                         OBJECT_HEADER=OBJECT_HEADER,
                                         SUPPORT_OBJECT=self.MODIFIED_DATATYPES,
                                         VALIDATED_DATATYPES=None,
                                         prompt_to_override=prompt_to_override,
                                         return_support_object_as_full_array=False,
                                         bypass_validation=False,
                                         calling_module=self.this_module,
                                         calling_fxn=fxn)

                MDTYPE_VALIDATOR.validate_allowed(kill=not prompt_to_override, fxn=fxn)

                self.MDTYPES_is_empty = MDTYPE_VALIDATOR.mdtypes_is_empty

                del MDTYPE_VALIDATOR

            del mdtypes_type

        # MUST BE LAST BECAUSE OF autofill()
        super().__init__(
                         OBJECT=OBJECT,
                         object_given_orientation=object_given_orientation,
                         columns=columns,
                         OBJECT_HEADER=OBJECT_HEADER,
                         SUPPORT_OBJECT=SUPPORT_OBJECT,
                         prompt_to_override=prompt_to_override,
                         return_support_object_as_full_array=return_support_object_as_full_array,
                         bypass_validation=self.bypass_validation,
                         calling_module=self.calling_module,
                         calling_fxn=self.calling_fxn
                        )

        if not self.MODIFIED_DATATYPES is None and self.return_support_object_as_full_array:
            self.SUPPORT_OBJECT[self.master_dict()['MODIFIEDDATATYPES']['position']] = self.MODIFIED_DATATYPES

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
    # OVERWRITTEN ############################################################################################################

    # INHERITED FROM Apex
    # def set_active_index(self):
    #     """Allows support object to be uniquely set in children as row of a larger object. """
    #     self.actv_idx = self.master_dict()[self.this_module.upper()]['position']

    def support_name(self):
        '''Name of child's support object.'''
        # OVERWRITE IN CHILD
        return 'VALIDATEDDATATYPES'

    def default_value(self):
        '''Default value to fill support object.'''
        # OVERWRITE IN CHILD
        return 'STR'


    def allowed_values(self):
        '''Allowed values for validation.'''
        # OVERWROTE IN CHILD
        self.TEXT_DTYPES = self.VAL_TEXT_DTYPES
        self.NUM_DTYPES = self.VAL_NUM_DTYPES
        self.NUM_DTYPES_LIST = self.VAL_NUM_DTYPES_LIST
        self.ALLOWED_DICT = self.VAL_ALLOWED_DICT
        self.ALLOWED_LIST = self.VAL_ALLOWED_LIST
        self.menu_allowed_cmds = self.val_menu_allowed_cmds
        self.MENU_OPTIONS_PROMPT = self.VAL_MENU_OPTIONS_PROMPT
        self.REVERSE_LOOKUP = self.VAL_REVERSE_LOOKUP


    # INHERITED FROM Apex
    # def validate_against_objects(self, OBJECT=None, kill=None, fxn=None):
    #     # FOR V DTYPES, VALIDATE ACCURACY AGAINST OBJECT (IF GIVEN)
    #     pass


    # INHERITED FROM Apex
    # def validate_allowed(self):


    def special_supobj_handling(self, fill_function, args, kwargs):
        # 1/17/23 BEAR LOOK AT ModifiedDatatypes HANDLING OF "VALIDATED_DATATYPES" KWARG, SEE WHAT CAN BE TAKEN FROM HERE

        # FOR SPEED'S SAKE, IF MODIFIED_DATATYPES IS AVAILABLE FROM "MODIFIED_DATATYPES" KWARG OR A FULL SUPPORT OBJECT,
        # THEN REVERSE-LOOKUP VAL DTYPES.  IF NEITHER IS AVAILABLE:
        # FOR autofill() DO THE HARDCORE WAY WITH "quick" OR "full" SET BY "quick_val" KWARG
        # FOR default_fill() USE default

        ######################################################################################################################
        # BUILD VALIDATED DATATYPES FROM MOD DTYPES IF GIVEN #################################################################
        if not self.MDTYPES_is_empty:  # DOESNT MATTER IF self.mdtypes_is_empty True OR False, GOING TO OVERWRITE
            # IF MODIFIED_DATATYPES WAS GIVEN AS A KWARG AND WAS ALSO GIVEN AS PART OF A FULL SUPPORT_OBJECT, ASSUME USER INTENDS
            # TO OVERWRITE SUPPORT_OBJECT M_DTYPES W GIVEN MODIFIED_DATATYPES

            # FOR EITHER if self.supobj_is_full OR not self.supobj_is_full ###############################################
            for _idx in range(self.columns):
                self.SUPPORT_OBJECT[self.actv_idx][_idx] = self.REVERSE_LOOKUP[self.MODIFIED_DATATYPES[_idx]]

            self.vdtypes_is_empty = False
            # END FOR EITHER ################################################################################################

            if self.supobj_is_full:
                self.SUPPORT_OBJECT[self.master_dict()['MODIFIEDDATATYPES']['position']] = self.MODIFIED_DATATYPES.copy()
                self.mdtypes_is_empty = False   # STAYS False IF WAS ALREADY FILLED, BECOMES False IF NEWLY FILLED

        elif self.MDTYPES_is_empty and not self.mdtypes_is_empty:
            # IF MODIFIED_DATATYPES WAS NOT PASSED VIA MODIFIED_DATATYPES KWARG, SEE IF ITS AVAILABLE IN A FULL SUPPORT OBJECT
            __ = self.master_dict()['MODIFIEDDATATYPES']['position']

            for _idx in range(self.columns):
                self.SUPPORT_OBJECT[self.actv_idx][_idx] = self.REVERSE_LOOKUP[self.SUPPORT_OBJECT[__][_idx]]
            del __
            self.vdtypes_is_empty = False
            # self.mdtypes_is_empty STAYS False

        # END BUILD VALIDATED DATATYPES FROM MOD DTYPES IF GIVEN #############################################################
        ######################################################################################################################

        ###################################################################################################################
        # BUILD VALIDATED DATATYPES FROM DATA (MOD DTYPES IS NOT GIVEN) ###################################################
        # ONLY WAY TO GET INTO autofill WAS self.vdtypes_is_empty AND OBJECT IS GIVEN (IF vdtypes NOT EMPTY, WOULD HAVE JUST
        # prompt_override>edit IN Apex)
        # ONLY WAY VAL DTYPES IS STILL EMPTY IS IF self.mdtypes_is_empty AND self.MDTYPES_is_empty WERE BOTH True
        elif self.MDTYPES_is_empty and self.mdtypes_is_empty:
            # vdtypes() AND sub_default_fill() AUTOMATICALLY FILLS SUPPORT_OBJECT[actv_idx] AT IDX FOR VAL
            fill_function(*args, **kwargs)
            self.vdtypes_is_empty = False


    def default_fill(self):
        # THE ONLY WAY TO GET HERE IS OBJECT IS NOT GIVEN AND self.vdtypes_is_empty IS True #################################
        # VALIDATED SLOT CAN BE FILLED EITHER BY VALIDATED_DATATYPES GIVEN BY "VALIDATED_DATATYPES" KWARG, BY REVERSE-LOOKUP
        # OF MOD DTYPES IN A FULL SUPPORT_OBJECT, OR W default_value.

        fxn = inspect.stack()[0][3]

        def sub_default_fill():
            for _idx in range(self.columns):
                self.SUPPORT_OBJECT[self.actv_idx] = self.default_value()

        self.special_supobj_handling(sub_default_fill, (), {})

        del sub_default_fill


    def autofill(self):
        '''Unique method to fill particular support object.'''
        '''Unique method to fill validated datatypes in ValidatedDatatypes.'''

        # 12/29/22 THE ONLY WAY THIS CODE CAN BE REACHED IS IF OBJECT IS GIVEN AND VALIDATED_DATATYPES IS NOT GIVEN
        # VIA SUPPORT_OBJECT (SUPPORT_OBJECT IS None OR IS FILLED W empty_value)

        fxn = inspect.stack()[0][3]

        self.special_supobj_handling(self.vdtypes, (), {'quick':self.quick_vdtypes, 'print_notes':self.print_notes})

        # AFTER GETTING REAL VAL DTYPES, OVERWRITE ANY QUICK RESULTS THAT MAY BE IN QUICK_VAL_DTYPES
        self.QUICK_VAL_DTYPES = self.SUPPORT_OBJECT[self.actv_idx].copy()
        self.qvdtypes_is_empty = False


    def fill_from_kwarg(self):
        fxn = inspect.stack()[0][3]

        # LAMBDA IS A DUMMY TO DO NOTHING, BUT FIT THE FORM OF special_supobj_handling()
        self.special_supobj_handling(lambda x: x, (1,), {})


    # END OVERWRITTEN ########################################################################################################
    ##########################################################################################################################
    ##########################################################################################################################
















