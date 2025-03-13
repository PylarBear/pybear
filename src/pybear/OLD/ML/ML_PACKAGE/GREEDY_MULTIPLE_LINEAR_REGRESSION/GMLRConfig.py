from ML_PACKAGE import MLConfigTemplate as mlct
from ML_PACKAGE.GREEDY_MULTIPLE_LINEAR_REGRESSION import GMLRCoreConfigCode as gccc



# OVERWRITTEN
# module_specific_config_cmds()              unique config commands for specific ML package
# standard_config_module()                   return standard config module
# module_specific_operations()               parameter selection operations specific to a child module
# print_parameters()                         print parameter state for child module
# return_fxn()                               return from child module

# INHERITED
# config()                                   exe


class GMLRConfig(mlct.MLConfigTemplate):
    def __init__(self, standard_config, gmlr_config, SUPER_WORKING_NUMPY_LIST, WORKING_SUPOBJS, data_run_orientation,
                 conv_kill, pct_change, conv_end_method, rglztn_type, rglztn_fctr, gmlr_batch_method, gmlr_batch_size,
                 gmlr_type, gmlr_score_method, gmlr_float_only, gmlr_max_columns, gmlr_bypass_agg, intcpt_col_idx,
                 bypass_validation):

        # NOTES 2-6-22 --- REMEMBER WE ARE TAKING IN gmlr_config, BUT IT'S BEING __init__ED AS sub_config in
        # THE PARENT (PARENT ONLY TALKS IN sub_config) SO MUST INVOKE gmlr_config THRU sub_config HERE

        # MODULE SPECIFIC __init__   # 11/14/22 THIS MUST BE BEFORE SUPER, SUPER __init__ NEEDS float_only DECLARATION
        self.batch_method = gmlr_batch_method
        self.batch_size = gmlr_batch_size
        self.gmlr_type = gmlr_type
        self.score_method = gmlr_score_method   # LAZY OR FULL
        self.float_only = gmlr_float_only
        self.max_columns = gmlr_max_columns
        self.bypass_agg = gmlr_bypass_agg
        self.intcpt_col_idx = intcpt_col_idx

        super().__init__(standard_config, gmlr_config, SUPER_WORKING_NUMPY_LIST, WORKING_SUPOBJS, data_run_orientation,
                         conv_kill, pct_change, conv_end_method, rglztn_type, rglztn_fctr, bypass_validation, __name__)


    def module_specific_config_cmds(self):
        return {
                    'b': 'batch method',
                    'c': 'batch size',
                    'd': 'select type (lazy / forward / backward)',
                    'f': 'select score method (F, RSQ, RSQ-adj, r)',  # F-score(f), RSQ(q), RSQ-adj(a), r(r)
                    'g': f'bypass agglomerative MLR ({self.bypass_agg})',
                    'h': 'adjust convergence kill',
                    'j': 'regularization type',
                    'k': 'regularization factor',
                    'l': f'toggle float only ({self.float_only})',
                    'm': 'max columns',
                }           # SPECIFIED IN CHILDREN, CANT USE AEIOQUZ



    def standard_config_module(self):
        #return sc.GMLR_standard_configs(self.standard_config, self.su_config)
        pass


    def module_specific_operations(self):
        self.conv_kill, self.pct_change, self.conv_end_method, self.rglztn_type, self.rglztn_fctr, self.batch_method, \
        self.batch_size, self.gmlr_type, self.score_method, self.float_only, self.max_columns, self.bypass_agg = \
            gccc.GMLRCoreConfigCode(self.sub_config, self.SUPER_WORKING_NUMPY_LIST[0], self.data_run_orientation,
                self.conv_kill, self.pct_change, self.conv_end_method, self.rglztn_type, self.rglztn_fctr, self.batch_method,
                self.batch_size, self.gmlr_type, self.score_method, self.float_only, self.max_columns, self.bypass_agg,
                self.intcpt_col_idx).config()

        # TO UPDATE FIELDS IN DYNAMIC MENU & GIVE DYNAMIC MENU FOR OPTIONS BASED ON CURRENT SELECTIONS
        disallowed = ''
        if self.gmlr_type=='L':
            disallowed += 'f'
            if self.bypass_agg is True: disallowed += 'h'
        if self.gmlr_type in ['F', 'B']: disallowed += 'g'
        if self.rglztn_type=='NONE': disallowed += 'k'

        AVAILABLE_MOD_SPECIFID_CMDS = {k:v for k,v in self.module_specific_config_cmds().items() if k not in disallowed}

        self.ALL_CMDS = self.GENERIC_CONFIG_CMDS | AVAILABLE_MOD_SPECIFID_CMDS

        del disallowed, AVAILABLE_MOD_SPECIFID_CMDS


    def print_parameters(self):
        _width = 30
        print(f"CONV KILL = ".ljust(_width) + f"{self.conv_kill}")
        print(f"PCT CHANGE = ".ljust(_width) + f"{self.pct_change}")
        print(f"CONV END METHOD = ".ljust(_width) + f"{self.conv_end_method}")
        print(f"RGLZTN TYPE = ".ljust(_width) + f"{self.rglztn_type}")
        print(f"RGLZTN FACTOR = ".ljust(_width) + f"{self.rglztn_fctr}")
        print(f"BATCH METHOD = ".ljust(_width) + f"{dict({'B':'BATCH','M':'MINI-BATCH'})[self.batch_method]}")
        print(f"BATCH SIZE = ".ljust(_width) + f"{self.batch_size}")
        print(f"TYPE = ".ljust(_width) + f"{dict({'B': 'Backward', 'F': 'Forward', 'L': 'Lazy'})[self.gmlr_type]}")
        print(f"SCORE METHOD = ".ljust(_width) + f"{dict({'F': 'F-score', 'Q': 'RSQ', 'A': 'ADJ RSQ', 'R':'r'})[self.score_method]}")
        print(f"FLOAT ONLY = ".ljust(_width) + f"{self.float_only}")
        print(f"MAX COLUMNS = ".ljust(_width) + f"{self.max_columns}")
        print(f"BYPASS AGGLOMERATIVE MLR = ".ljust(_width) + f"{self.bypass_agg}")
        print(f"INTERCEPT COLUMN INDEX = ".ljust(_width) + f"{self.intcpt_col_idx}")


    def return_fxn(self):
        return self.conv_kill, self.pct_change, self.conv_end_method, self.rglztn_type, self.rglztn_fctr, self.batch_method, \
                self.batch_size, self.gmlr_type, self.score_method, self.float_only, self.max_columns, self.bypass_agg, \
                self.intcpt_col_idx









if __name__ == '__main__':

    # TEST MODULE  --- MODULE AND TEST VERIFIED GOOD 5/24/2023

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
    gmlr_config = 'BYPASS'
    intcpt_col_idx = None

    gmlr_conv_kill = None
    gmlr_pct_change = float('inf')
    gmlr_conv_end_method = 'KILL'
    gmlr_rglztn_type = 'RIDGE'
    gmlr_rglztn_fctr = 100
    gmlr_batch_method = 'B'
    gmlr_batch_size = int(1e12)
    gmlr_type = 'L'
    gmlr_score_method = 'Q'
    gmlr_float_only = True
    gmlr_max_columns = 10
    gmlr_bypass_agg = False


    gmlr_conv_kill, gmlr_pct_change, gmlr_conv_end_method, gmlr_rglztn_type, gmlr_rglztn_fctr, gmlr_batch_method, \
    gmlr_batch_size, gmlr_gmlr_type, gmlr_score_method, gmlr_float_only, gmlr_max_columns, gmlr_bypass_agg, \
    gmlr_intcpt_col_idx = \
        GMLRConfig(standard_config, gmlr_config, SUPER_WORKING_NUMPY_LIST, WORKING_SUPOBJS, data_run_orientation,
                   gmlr_conv_kill, gmlr_pct_change, gmlr_conv_end_method, gmlr_rglztn_type, gmlr_rglztn_fctr,
                   gmlr_batch_method, gmlr_batch_size, gmlr_type, gmlr_score_method, gmlr_float_only, gmlr_max_columns,
                   gmlr_bypass_agg, intcpt_col_idx, bypass_validation).config()

    # IF type IS LAZY, SHOULD FORCE method TO RSQ
    # IF BYPASSING AGG SHOULD NOT ALLOW CONV KILL ET AL INTO MENU,
    # SHOULD NOT ALLOW OPTIONS FOR AGG IF type IS FORWARD OR BACKWARD
    # SHOULD NOT ALLOW OPTION TO SET rglztn_fctr IF rglztn_type IS None











