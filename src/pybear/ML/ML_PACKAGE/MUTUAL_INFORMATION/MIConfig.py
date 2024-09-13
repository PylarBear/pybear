from ML_PACKAGE import MLConfigTemplate as mlct
from ML_PACKAGE.MUTUAL_INFORMATION import MICoreConfigCode as miccc



# OVERWRITTEN
# module_specific_config_cmds()              unique config commands for specific ML package
# standard_config_module()                   return standard config module
# module_specific_operations()               parameter selection operations specific to a child module
# print_parameters()                         print parameter state for child module
# return_fxn()                               return from child module

# INHERITED
# config()                                   exe


class MIConfig(mlct.MLConfigTemplate):
    def __init__(self, standard_config, mi_config, SUPER_WORKING_NUMPY_LIST, WORKING_SUPOBJS, data_run_orientation,
                batch_method, batch_size, mi_int_or_bin_only, mi_max_columns, mi_bypass_agg, intcpt_col_idx,
                bypass_validation):

        # NOTES 4-17-22 --- REMEMBER WE ARE TAKING IN mi_conig, BUT IT'S BEING __init__ED AS sub_config in
        # THE PARENT (PARENT ONLY TALKS IN sub_config) SO MUST INVOKE mi_config THRU sub_config HERE

        # MODULE SPECIFIC __init__   # 11/14/22 THIS MUST BE BEFORE SUPER, SUPER __init__ NEEDS int_or_bin DECLARATION
        # TO HANDLE module_specific_config_cmds()
        self.batch_method = batch_method
        self.batch_size = batch_size
        self.int_or_bin_only = mi_int_or_bin_only
        self.max_columns = mi_max_columns
        self.bypass_agg = mi_bypass_agg
        self.intcpt_col_idx = intcpt_col_idx

        super().__init__(standard_config, mi_config, SUPER_WORKING_NUMPY_LIST, WORKING_SUPOBJS, data_run_orientation,
                         'dum_conv_kill', 'dum_pct_change', 'dum_conv_end_method', 'dum_rglztn_type', 'dum_rglztn_fctr',
                         bypass_validation, __name__)


    def module_specific_config_cmds(self):
        return {
                'b': 'batch method ',
                'c': 'batch size ',
                'd': f'toggle int_or_bin_only ({self.int_or_bin_only})',
                'f': 'max columns',
                'g': f'bypass agglomerative MLR ({self.bypass_agg})'
        }           # SPECIFIED IN CHILDREN   CANT USE AEIQZ


    def standard_config_module(self):
        #return sc.MI_standard_configs(self.standard_config, self.su_config)
        pass


    def module_specific_operations(self):
        self.batch_method, self.batch_size, self.int_or_bin_only, self.max_columns, self.bypass_agg = \
            miccc.MICoreConfigCode(self.sub_config, self.SUPER_WORKING_NUMPY_LIST[0], self.data_run_orientation,
                                   self.batch_method, self.batch_size, self.int_or_bin_only, self.max_columns,
                                   self.bypass_agg).config()


        # TO UPDATE FIELDS IN DYNAMIC MENU
        self.ALL_CMDS = self.GENERIC_CONFIG_CMDS | self.module_specific_config_cmds()


    def print_parameters(self):
        _width = 30
        print()
        print(f'BATCH METHOD = '.ljust(_width) + f'{self.batch_method}')
        print(f'BATCH SIZE = '.ljust(_width) + f'{self.batch_size}')
        print(f'INTEGER OR BINARY ONLY = '.ljust(_width) + f'{self.int_or_bin_only}')
        print(f'MAX COLUMNS = '.ljust(_width) + f'{self.max_columns}')
        print(f'BYPASS AGGLOMERATIVE MLR = '.ljust(_width) + f'{self.bypass_agg}')
        print(f'INTERCEPT COLUMN INDEX = '.ljust(_width) + f'{self.intcpt_col_idx}')


    def return_fxn(self):
        return *self.return_fxn_base(), self.batch_method, self.batch_size, self.int_or_bin_only, self.max_columns, \
            self.bypass_agg, self.intcpt_col_idx







if __name__ == '__main__':

    import numpy as np
    from MLObjects.TestObjectCreators.SXNL import CreateSXNL as csxnl
    from MLObjects.TestObjectCreators import test_header as th

    _rows = 6
    _columns = 4
    DATA = np.arange(0, _rows*_columns, dtype=np.int32).reshape((_columns, _rows))
    data_run_orientation = 'COLUMN'
    bypass_validation = False

    SWNLClass = csxnl.CreateSXNL(rows=_rows,
                                 bypass_validation=bypass_validation,
                                 ########################################################################################
                                 # DATA #################################################################################
                                 data_return_format='ARRAY',
                                 data_return_orientation='AS_GIVEN',
                                 DATA_OBJECT=DATA,
                                 DATA_OBJECT_HEADER=th.test_header(_columns),
                                 DATA_FULL_SUPOBJ_OR_SINGLE_MDTYPES=None,
                                 data_override_sup_obj=False,
                                 data_given_orientation=data_run_orientation,
                                 data_columns=None,
                                 DATA_BUILD_FROM_MOD_DTYPES=None,
                                 DATA_NUMBER_OF_CATEGORIES=None,
                                 DATA_MIN_VALUES=None,
                                 DATA_MAX_VALUES=None,
                                 DATA_SPARSITIES=None,
                                 DATA_WORD_COUNT=None,
                                 DATA_POOL_SIZE=None,
                                 # END DATA #############################################################################
                                 ########################################################################################

                                 ########################################################################################
                                 # TARGET ###############################################################################
                                 target_return_format='ARRAY',
                                 target_return_orientation='COLUMN',
                                 TARGET_OBJECT=None,
                                 TARGET_OBJECT_HEADER=[['TARGET']],
                                 TARGET_FULL_SUPOBJ_OR_SINGLE_MDTYPES=None,
                                 target_type='BINARY',  # MUST BE 'BINARY','FLOAT', OR 'SOFTMAX'
                                 target_override_sup_obj=False,
                                 target_given_orientation=None,
                                 # END CORE TARGET_ARGS ########################################################
                                 # FLOAT AND BINARY
                                 target_sparsity=50,
                                 # FLOAT ONLY
                                 target_build_from_mod_dtype='INT',  # COULD BE FLOAT OR INT
                                 target_min_value=0,
                                 target_max_value=1,
                                 # SOFTMAX ONLY
                                 target_number_of_categories=None,

                                # END TARGET ############################################################################
                                #########################################################################################

                                #########################################################################################
                                # REFVECS ###############################################################################
                                refvecs_return_format='ARRAY',    # IS ALWAYS ARRAY (WAS, CHANGED THIS 4/6/23)
                                refvecs_return_orientation='COLUMN',
                                REFVECS_OBJECT=None,
                                REFVECS_OBJECT_HEADER=None,
                                REFVECS_FULL_SUPOBJ_OR_SINGLE_MDTYPES=None,
                                REFVECS_BUILD_FROM_MOD_DTYPES='STR',
                                refvecs_override_sup_obj=False,
                                refvecs_given_orientation=None,
                                refvecs_columns=3,
                                REFVECS_NUMBER_OF_CATEGORIES=5,
                                REFVECS_MIN_VALUES=None,
                                REFVECS_MAX_VALUES=None,
                                REFVECS_SPARSITIES=None,
                                REFVECS_WORD_COUNT=None,
                                REFVECS_POOL_SIZE=None
                                # END REFVECS ###########################################################################
                                #########################################################################################
    )

    SWNLClass.expand_data(expand_as_sparse_dict=False, auto_drop_rightmost_column=False)

    SUPER_WORKING_NUMPY_LIST = SWNLClass.SXNL
    WORKING_SUPOBJS = SWNLClass.SXNL_SUPPORT_OBJECTS
    del SWNLClass

    standard_config = 'BYPASS'
    mi_config = 'BYPASS'
    mi_batch_method = 'B'
    mi_batch_size = ...
    mi_int_or_bin_only = False
    mi_max_columns = 10
    mi_bypass_agg = False
    intcpt_col_idx = None



    SUPER_WORKING_NUMPY_LIST, WORKING_SUPOBJS, mi_batch_method, mi_batch_size, mi_int_or_bin_only, mi_max_columns, \
    mi_bypass_agg, intcpt_col_idx = \
        MIConfig(standard_config, mi_config, SUPER_WORKING_NUMPY_LIST, WORKING_SUPOBJS, data_run_orientation,
             mi_batch_method, mi_batch_size, mi_int_or_bin_only, mi_max_columns, mi_bypass_agg, intcpt_col_idx,
             bypass_validation).config()




















