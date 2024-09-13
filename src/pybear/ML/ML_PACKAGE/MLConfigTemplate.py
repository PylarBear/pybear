import numpy as np
import sys, inspect
from data_validation import validate_user_input as vui
from debug import get_module_name as gmn
from general_list_ops import list_select as ls
from MLObjects.SupportObjects import master_support_object_dict as msod
from ML_PACKAGE.GENERIC_PRINT import DictMenuPrint as dmp
from MLObjects import MLObject as mlo


# module_specific_config_cmds()              SPECIFIED IN CHILDREN
# standard_config_module()                   return standard config module
# module_specific_operations()               parameter selection operations specific to a child module
# print_parameters()                         print parameter state for child module
# return_fxn()                               return from child module
# config()                                   exe


class MLConfigTemplate:
    def __init__(self, standard_config, sub_config, SUPER_WORKING_NUMPY_LIST, WORKING_SUPOBJS, data_run_orientation,
                 conv_kill, pct_change, conv_end_method, rglztn_type, rglztn_fctr, bypass_validation, module):

        self.standard_config = standard_config
        self.sub_config = sub_config
        self.SUPER_WORKING_NUMPY_LIST = SUPER_WORKING_NUMPY_LIST
        self.WORKING_SUPOBJS = WORKING_SUPOBJS
        self.data_run_orientation = data_run_orientation
        self.conv_kill = conv_kill
        self.pct_change = pct_change
        self.conv_end_method = conv_end_method
        self.rglztn_type = rglztn_type
        self.rglztn_fctr = rglztn_fctr
        self.bypass_validation = bypass_validation
        self.this_module = gmn.get_module_name(str(sys.modules[module]))
        fxn = '__init__'

        # 'Z' MUST BE RESERVED AS A BYPASS (OTHER BYPASS TERMS LIKE 'BYPASS' MIGHT CONFLICT WITH SOME CHILD SETUP LOGIC)

        # AEIQZ RESERVED FOR GENERIC CMDS, AS OF 5/13/23 OUXY ARE ALSO AVAILABLE BUT NOT USED
        self.GENERIC_CONFIG_CMDS = {
                                    'a': 'accept config / continue',
                                    'e': 'compulsory manual setup sequence',
                                    'i': 'select standard config',
                                    'o': 'select columns',
                                    'q': 'quit'
        }

        self.ALL_CMDS = self.GENERIC_CONFIG_CMDS | self.module_specific_config_cmds()

        # CHECK THAT MERGED DICTS DID NOT HAVE ANY EQUAL KEYS
        if len(self.ALL_CMDS) != len(self.GENERIC_CONFIG_CMDS) + len(self.module_specific_config_cmds()):
            raise Exception(f'*** {self.this_module}.{fxn}() >>> GENERIC_CONFIG_CMDS AND module_specific_config_cmds '
                            f'HAVE AT LEAST 1 EQUAL KEY ***')



    def module_specific_config_cmds(self):
        return {}           # SPECIFIED IN CHILDREN


    def standard_config_module(self):
        pass # OVERWROTE IN CHILD


    def module_specific_operations(self):
        pass   # SPECIFIED IN CHILDREN


    def print_parameters(self):
        pass  # OVERWROTE IN CHILD


    def return_fxn_base(self):
        return self.SUPER_WORKING_NUMPY_LIST, self.WORKING_SUPOBJS


    def return_fxn(self):
        pass
        # REMEMBER TO PUT return_fxn_base() INTO CHILD return_fxn()
        #return ---variables that are unique to specific ML run() program---


    def config(self):
        # CREATE A 'Z' SETTING THAT IS USED TO DISTINGUISH BETWEEN THE 1ST PASS & ALL SUBSEQNT PASSES

        fxn = inspect.stack()[0][3]

        while True:
            ################################################################################################################
            # CHOOSE A STANDARD CONFIG OR MANUAL############################################################################

            if self.sub_config in 'IZ':  # load standard config(i), perform compulsory setup sequence(e), other manual config(z)
                if self.sub_config == 'Z':
                    self.sub_config = vui.validate_user_str(f'\nLoad {self.this_module} standard configs(i) or compulsory manual setup ' + \
                                 f'sequence(e) or proceed with other manual config(z)? > ', 'IEZ')

                if self.sub_config in 'EZ':
                    pass

                elif self.sub_config == 'I':
                    self.standard_config_module()

            # END STANDARD CONFIG OR MANUAL #################################################################################
            ################################################################################################################

            if self.sub_config in "".join(self.module_specific_config_cmds().keys()).upper() + 'EIA':
                # NOT "Z", SEND "I" THRU TO GET AON BUILT AFTER GETTING CONFIG INFO, SEND "A" THRU IN CASE THERE IS SPECIAL FINAL CODE FOR "A"
                self.module_specific_operations()

            #################################################################################################################
            # DONT DELETE OR MOVE THESE #####################################################################################
            if self.sub_config == 'A':  # 'accept config / continue(a)',
                break

            ####################################################################################################################
            ####################################################################################################################

            # CHOP COLUMNS IN MIConfig AND RETURN SWNL AND SUPOBJS BACK TO MLConfigRunTemplate.  PUTTING A COLUMN CHOP
            # OPERATION DIRECTLY LEADING INTO MLRunTemplate (EITHER AT THE TOP OF MLRunTemplate OR BEFORE MLRunTemplate()
            # CALL IN MLConfigRunTemplate) WILL CAUSE CHOPS EVERYTIME AFTER EXITING MLRunTemptate BACK TO MLConfigRun AND
            # GO BACK INTO MLRunTemplate

            if self.sub_config == 'O':  # select columns(o)

                while True:
                    print(f'\nSELECT COLUMNS TO USE IN ANALYSIS: \n')
                    ACTV_HDR = self.WORKING_SUPOBJS[0][msod.QUICK_POSN_DICT()["HEADER"]]
                    SELECTED_COLUMNS = ls.list_custom_select(ACTV_HDR, 'idx')

                    print(f'\nUSER SELECTED TO INCLUDE ONLY THE FOLLOWING COLUMNS FOR ANALYSIS:')
                    [print(_) for _ in ACTV_HDR[SELECTED_COLUMNS]]
                    __ = vui.validate_user_str(f'\nAccept(a), restart(r), abort(b)? > ', 'ARB')
                    if __ == 'A': pass
                    elif __ == 'R': continue
                    elif __ == 'B': break

                    del ACTV_HDR

                    # CHOP DATA & SUPOBJ TO SELECTED COLUMNS
                    SelectorClass = mlo.MLObject(self.SUPER_WORKING_NUMPY_LIST[0],
                                                 self.data_run_orientation,
                                                 name = 'DATA',
                                                 return_orientation = self.data_run_orientation,
                                                 return_format = 'AS_GIVEN',
                                                 bypass_validation = self.bypass_validation,
                                                 calling_module = self.this_module,
                                                 calling_fxn = fxn
                    )

                    self.SUPER_WORKING_NUMPY_LIST[0] = SelectorClass.return_columns(SELECTED_COLUMNS,
                                                                                    return_orientation='AS_GIVEN',
                                                                                    return_format='AS_GIVEN')

                    self.WORKING_SUPOBJS[0] = self.WORKING_SUPOBJS[0][..., SELECTED_COLUMNS]

                    del SELECTED_COLUMNS

                    break



            ####################################################################################################################
            ####################################################################################################################

            if self.sub_config == 'Q':  # quit(q)
                raise Exception(f'USER TERMINATED.')
            # END DONT DELETE OR MOVE THESE #################################################################################
            #################################################################################################################

            # PRINT CURRENT STATE OF PARAMETERS NEXT TO MENU OPTIONS
            print(); self.print_parameters(); print()

            self.sub_config = dmp.DictMenuPrint(self.ALL_CMDS, disp_len=140).select(f'')

        return self.return_fxn()













