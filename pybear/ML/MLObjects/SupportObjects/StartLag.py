import sys, inspect
import numpy as np
from MLObjects.SupportObjects import MinCutoffs as mc
from debug import get_module_name as gmn
from data_validation import arg_kwarg_validater as akv


# CHILD OF MinCutoffs
# PARENT OF EndLag
class StartLag(mc.MinCutoffs):

    def __init__(self,
                OBJECT = None,
                object_given_orientation = 'COLUMN',
                columns=None,
                OBJECT_HEADER=None,
                SUPPORT_OBJECT = None,
                prompt_to_override = False,
                return_support_object_as_full_array = True,
                bypass_validation = False,
                calling_module = None,
                calling_fxn = None
                ):

        self.this_module = gmn.get_module_name(str(sys.modules[__name__]))
        fxn = '__init__'
        self.calling_module = calling_module if not calling_module is None else self.this_module
        self.calling_fxn = calling_fxn if not calling_fxn is None else fxn


        super().__init__(
                        OBJECT = OBJECT,
                        object_given_orientation = object_given_orientation,
                        columns=columns,
                        OBJECT_HEADER = OBJECT_HEADER,
                        SUPPORT_OBJECT = SUPPORT_OBJECT,
                        prompt_to_override = prompt_to_override,
                        return_support_object_as_full_array = return_support_object_as_full_array,
                        bypass_validation = bypass_validation,
                        calling_module = self.calling_module,
                        calling_fxn = self.calling_fxn
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

    # INHERITED FROM Apex
    # def set_active_index(self):
    #     '''Allows support object to be uniquely set in children to a single object or row of a larger object. '''
    #     self.actv_idx = self.master_dict()[self.this_module.upper()]['position']

    def support_name(self):
        '''Name of child's support object.'''
        return 'STARTLAG'

    # INHERITED FROM MinCutoffs
    # def default_value(self):
    #     '''Default value to fill support object.'''

    def allowed_values(self):
        'Allowed values for validation.'''
        self.ALLOWED_LIST = range(int(-1e7),int(1e7),1)


    # INHERITED FROM Apex
    # def autofill(self):
    #     '''Unique method to fill particular support object.'''


    # INHERITED FROM Apex VIA MinCutoffs
    # def validate_allowed(self, value=None):


    # INHERITED FROM Apex
    # def validate_against_objects(self, OBJECT=None, kill=None, fxn=None):
    #     pass


    # END OVERWRITTEN ########################################################################################################
    ##########################################################################################################################
    ##########################################################################################################################


















