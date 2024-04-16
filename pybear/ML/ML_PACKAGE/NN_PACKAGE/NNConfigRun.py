import numpy as np
from copy import deepcopy
from ML_PACKAGE import MLConfigRunTemplate as mlcrt
from ML_PACKAGE.NN_PACKAGE import NNConfig as nnc, NNRun as nnr
from MLObjects import MLObject as mlo
from MLObjects.ObjectOrienter import MLObjectOrienter as mloo
from MLObjects.SupportObjects import master_support_object_dict as msod



# INHERITS #############################################################################################################
# dataclass_mlo_loader()             Loads an instance of MLObject for DATA.
# intercept_manager()                Locate columns of constants in DATA & handle. As of 11/15/22 only for MLR, MI, and GMLR.
# insert_intercept()                 Insert a column of ones in the 0 index of DATA.
# delete_column()                    Delete a column from DATA and respective holder objects.
# run_module_input_tuple()           tuple of base params that pass into run_module for all ML packages
# return_fxn_base()                  values returned from all ML packages
# base_post_run_options_module()     holds post run options applied to all ML packages
# configrun()                        runs config_module() & run_module()

# INHERITS FOR NOW #####################################################################################################
# sub_post_run_cmds()                package-specific options available to modify WORKING_DATA after run
# sub_post_run_options_module()      holds post run options unique to particular ML package

# OVERWRITES ###########################################################################################################
# config_module()                    gets configuration source, returns configuration parameters for particular ML package
# run_module()                       returns run module for particular ML package
# return_fxn()                       returns user-specified output, in addition to return_fxn_base()


#CALLED BY ML
class NNConfigRun(mlcrt.MLConfigRunTemplate):
    def __init__(self, standard_config, nn_config, SUPER_RAW_NUMPY_LIST, RAW_SUPOBJS, SUPER_WORKING_NUMPY_LIST,
        WORKING_SUPOBJS, data_given_orientation, target_given_orientation, refvecs_given_orientation, WORKING_CONTEXT,
        WORKING_KEEP, split_method, LABEL_RULES, number_of_labels, event_value, negative_value, rglztn_type, rglztn_fctr,
        conv_kill, pct_change, conv_end_method,

        # NN SPECIFIC PARAMS
        ARRAY_OF_NODES, NEURONS, nodes, node_seed, activation_constant, aon_base_path, aon_filename, cost_fxn,
        SELECT_LINK_FXN, LIST_OF_NN_ELEMENTS, OUTPUT_VECTOR, batch_method, BATCH_SIZE, gd_method, conv_method, lr_method,
        LEARNING_RATE, momentum_weight, gd_iterations, non_neg_coeffs, allow_summary_print, summary_print_interval, iteration):

        data_run_format = 'AS_GIVEN'
        data_run_orientation = 'COLUMN'
        target_run_format = 'ARRAY'
        target_run_orientation = 'COLUMN'
        refvecs_run_format = 'ARRAY'
        refvecs_run_orientation = 'COLUMN'

        super().__init__(standard_config, nn_config, SUPER_RAW_NUMPY_LIST, RAW_SUPOBJS, SUPER_WORKING_NUMPY_LIST,
                WORKING_SUPOBJS, data_given_orientation, target_given_orientation, refvecs_given_orientation,
                data_run_orientation, target_run_orientation, refvecs_run_orientation, data_run_format, target_run_format,
                refvecs_run_format, WORKING_CONTEXT, WORKING_KEEP, split_method, LABEL_RULES, number_of_labels, event_value,
                negative_value, conv_kill, pct_change, conv_end_method, rglztn_type, rglztn_fctr, __name__)

        fxn = '__init__'

        # PLACEHOLDER FOR NN SPECIFIC __init__
        self.ARRAY_OF_NODES = ARRAY_OF_NODES
        self.NEURONS = NEURONS
        self.nodes = nodes
        self.node_seed = node_seed
        self.activation_constant = activation_constant
        self.aon_base_path = aon_base_path
        self.aon_filename = aon_filename
        self.cost_fxn = cost_fxn.upper()
        self.SELECT_LINK_FXN = SELECT_LINK_FXN
        self.LIST_OF_NN_ELEMENTS = LIST_OF_NN_ELEMENTS
        self.OUTPUT_VECTOR = OUTPUT_VECTOR
        self.batch_method = batch_method
        self.BATCH_SIZE = BATCH_SIZE
        self.gd_method = gd_method
        self.conv_method = conv_method
        self.lr_method = lr_method
        self.LEARNING_RATE = LEARNING_RATE
        self.momentum_weight = momentum_weight
        self.gd_iterations = gd_iterations
        self.non_neg_coeffs = non_neg_coeffs
        self.allow_summary_print = allow_summary_print
        self.summary_print_interval = summary_print_interval
        self.iteration = 0
        self.new_error_start = float('inf')


        ########################################################################################################################
        # ORIENT SWNL & MAKE OTHER OBJECTS #####################################################################################

        # SRNL IS IN super(), SWNL HANDLED IN CHILDREN FOR EASIER HANDLING OF DIFFERENT "RETURN_OBJECTS" FOR DIFFERENT PACKAGES

        print(f'\n    BEAR IN MIConfigRun Orienting WORKING DATA & TARGET IN __init__.  Patience...')

        # BREAK CONNECTIONS OF SUPER_WORKING_NUMPY_LIST WITH SUPER_WORKING_NUMPY_LIST IN OUTER SCOPE. DO NOT CHANGE THIS,
        # SEE NOTES IN MLRegressionConfigRun.
        self.SUPER_WORKING_NUMPY_LIST = [deepcopy(_) if isinstance(_, dict) else _.copy() for _ in SUPER_WORKING_NUMPY_LIST]

        SWNLOrienterClass = mloo.MLObjectOrienter(
                                                    DATA=self.SUPER_WORKING_NUMPY_LIST[0],
                                                    data_given_orientation=self.data_given_orientation,
                                                    data_return_orientation=self.data_run_orientation,
                                                    data_return_format=self.data_run_format,

                                                    target_is_multiclass=self.working_target_is_multiclass,
                                                    TARGET=self.SUPER_WORKING_NUMPY_LIST[1],
                                                    target_given_orientation=self.target_given_orientation,
                                                    target_return_orientation=self.target_run_orientation,
                                                    target_return_format=self.target_run_format,

                                                    RETURN_OBJECTS=['DATA', 'TARGET'],

                                                    bypass_validation=self.bypass_validation,
                                                    calling_module=self.this_module,
                                                    calling_fxn=fxn
        )

        # BECAUSE ObjectOrienter WASNT BUILT TO HANDLE REFVECS
        RefVecsClass = mlo.MLObject(self.SUPER_WORKING_NUMPY_LIST[2], self.refvecs_given_orientation,
            name='REFVECS', return_orientation=self.refvecs_run_orientation, return_format=self.refvecs_run_format,
            bypass_validation=self.bypass_validation, calling_module=self.this_module, calling_fxn=fxn)

        print(f'\n    BEAR IN MIConfigRun Orienting WORKING DATA & TARGET IN __init__.  Done')


        # BECAUSE ANY OF THESE COULD BE 'AS_GIVEN', GET ACTUALS
        self.SUPER_WORKING_NUMPY_LIST[0] = SWNLOrienterClass.DATA
        self.data_run_format = SWNLOrienterClass.data_return_format
        self.data_run_orientation = SWNLOrienterClass.data_return_orientation
        self.SUPER_WORKING_NUMPY_LIST[1] = SWNLOrienterClass.TARGET
        self.target_run_format = SWNLOrienterClass.target_return_format
        self.target_run_orientation = SWNLOrienterClass.target_return_orientation

        del SWNLOrienterClass

        self.SUPER_WORKING_NUMPY_LIST[2] = RefVecsClass.OBJECT
        self.refvecs_run_format = RefVecsClass.current_format
        self.refvecs_run_orientation = RefVecsClass.current_orientation

        del RefVecsClass

        # END ORIENT SWNL & MAKE OTHER OBJECTS #################################################################################
        ########################################################################################################################



    # INHERITS #############################################################################################################
    # intercept_finder()                 Not used here. Locate an intercept column in DATA and handle any anamolous columns of constants.
    #                                    As of 11/15/22 only for MLR, MI, and GMLR.
    # run_module_input_tuple()           tuple of base params that pass into run_module for all ML packages
    # return_fxn_base()                  values returned from all ML packages
    # base_post_run_options_module()     holds post run options applied to all ML packages
    # configrun()                        runs config_module() & run_module()


    # OVERWRITES #######################################################################################################
    def config_module(self):
        self.ARRAY_OF_NODES, self.NEURONS, self.nodes, self.node_seed, self.activation_constant, self.aon_base_path, \
        self.aon_filename, self.cost_fxn, self.SELECT_LINK_FXN, self.LIST_OF_NN_ELEMENTS, self.OUTPUT_VECTOR, \
        self.batch_method, self.BATCH_SIZE, self.gd_method, self.conv_method, self.lr_method, self.LEARNING_RATE, \
        self.momentum_weight, self.rglztn_type, self.rglztn_fctr, self.conv_kill, self.pct_change, self.conv_end_method, \
        self.gd_iterations, self.non_neg_coeffs, self.allow_summary_print, self.summary_print_interval, self.iteration = \
            nnc.NNConfig(self.standard_config, self.sub_config, self.SUPER_WORKING_NUMPY_LIST, self.WORKING_SUPOBJS,
                self.data_run_orientation, self.target_run_orientation, self.ARRAY_OF_NODES, self.NEURONS, self.nodes,
                self.node_seed, self.activation_constant, self.aon_base_path, self.aon_filename, self.cost_fxn,
                self.SELECT_LINK_FXN, self.LIST_OF_NN_ELEMENTS, self.OUTPUT_VECTOR, self.batch_method, self.BATCH_SIZE,
                self.gd_method, self.conv_method, self.lr_method, self.LEARNING_RATE, self.momentum_weight, self.rglztn_type,
                self.rglztn_fctr, self.conv_kill, self.pct_change, self.conv_end_method, self.gd_iterations,
                self.non_neg_coeffs, self. allow_summary_print, self.summary_print_interval, self.iteration,
                self.bypass_validation).config()


    def run_module(self):

        # *self.run_module_input_tuple(), \
        self.TRAIN_SWNL, self.DEV_SWNL, self.TEST_SWNL, self.PERTURBER_RESULTS, self.ARRAY_OF_NODES, self.OUTPUT_VECTOR, \
        self.iteration, self.rglztn_fctr = \
        nnr.NNRun(*self.run_module_input_tuple(), self.ARRAY_OF_NODES, self.NEURONS, self.LIST_OF_NN_ELEMENTS,
            self.SELECT_LINK_FXN, self.BATCH_SIZE, self.LEARNING_RATE, self.OUTPUT_VECTOR, self.nodes, self.node_seed,
            self.new_error_start, self.aon_base_path, self.aon_filename, self.activation_constant, self.gd_iterations,
            self.cost_fxn, self.allow_summary_print, self.summary_print_interval, self.batch_method, self.gd_method,
            self.conv_method, self.lr_method, self.momentum_weight, self.non_neg_coeffs, self.iteration).run()


    def return_fxn(self):
    # returns user-specified output, in addition to return_fxn_base()
        return *self.return_fxn_base(), self.ARRAY_OF_NODES, self.NEURONS, self.nodes, self.node_seed, \
            self.activation_constant, self.aon_base_path, self.aon_filename, self.cost_fxn, self.SELECT_LINK_FXN, \
            self.LIST_OF_NN_ELEMENTS, self.OUTPUT_VECTOR, self.batch_method, self.BATCH_SIZE, self.gd_method, \
            self.conv_method, self.lr_method, self.LEARNING_RATE, self.momentum_weight, self.gd_iterations, \
            self.non_neg_coeffs, self.allow_summary_print, self.summary_print_interval, self.iteration


    # INHERITS FOR NOW #####################################################################################################
    '''
    def sub_post_run_cmds(self):
    # package-specific options available to modify WORKING_DATA after run
        return []


    def sub_post_run_options_module(self):
    # holds post run options unique to particular ML package
        pass
    '''













if __name__ == '__main__':
    from MLObjects.TestObjectCreators.SXNL import CreateSXNL as csxnl
    import pandas as pd

    DATA = pd.read_csv(r'C:\Users\Bill\Documents\WORK STUFF\RESUME\1 - OTHER\SRP\beer_reviews.csv',
                      nrows=100,
                      header=0).dropna(axis=0)

    DATA = DATA[DATA.keys()[[3, 4, 5, 8, 9, 11]]]   # , 7

    TARGET = DATA['review_overall']
    TARGET_HEADER = [['review_overall']]
    TARGET = TARGET.to_numpy().reshape((1, -1))
    # TRANSFORM TARGET TO BINARY FOR LOGISTIC
    # TARGET[0] = TARGET[0] >= 3.5

    DATA = DATA.drop(columns=['review_overall'])

    RAW_DATA = DATA.copy()
    RAW_DATA_HEADER = np.fromiter(RAW_DATA.keys(), dtype='<U50').reshape((1, -1))
    RAW_DATA = RAW_DATA.to_numpy()

    SXNLClass = csxnl.CreateSXNL(rows=None,
                                 bypass_validation=False,
                                 data_return_format='ARRAY',
                                 data_return_orientation='COLUMN',
                                 DATA_OBJECT=RAW_DATA,
                                 DATA_OBJECT_HEADER=RAW_DATA_HEADER,
                                 DATA_FULL_SUPOBJ_OR_SINGLE_MDTYPES=None,
                                 data_override_sup_obj=False,
                                 data_given_orientation='ROW',
                                 data_columns=None,
                                 DATA_BUILD_FROM_MOD_DTYPES=None,
                                 DATA_NUMBER_OF_CATEGORIES=None,
                                 DATA_MIN_VALUES=None,
                                 DATA_MAX_VALUES=None,
                                 DATA_SPARSITIES=None,
                                 DATA_WORD_COUNT=None,
                                 DATA_POOL_SIZE=None,
                                 target_return_format='ARRAY',
                                 target_return_orientation='COLUMN',
                                 TARGET_OBJECT=TARGET,
                                 TARGET_OBJECT_HEADER=TARGET_HEADER,
                                 TARGET_FULL_SUPOBJ_OR_SINGLE_MDTYPES=None,
                                 target_type='FLOAT',
                                 target_override_sup_obj=False,
                                 target_given_orientation='COLUMN',
                                 target_sparsity=None,
                                 target_build_from_mod_dtype=None,
                                 target_min_value=None,
                                 target_max_value=None,
                                 target_number_of_categories=None,
                                 refvecs_return_format='ARRAY',
                                 refvecs_return_orientation='COLUMN',
                                 REFVECS_OBJECT=np.fromiter(range(len(RAW_DATA)), dtype=int).reshape((1, -1)),
                                 REFVECS_OBJECT_HEADER=[['ROW_ID']],
                                 REFVECS_FULL_SUPOBJ_OR_SINGLE_MDTYPES=None,
                                 REFVECS_BUILD_FROM_MOD_DTYPES=None,
                                 refvecs_override_sup_obj=False,
                                 refvecs_given_orientation='COLUMN',
                                 refvecs_columns=None,
                                 REFVECS_NUMBER_OF_CATEGORIES=None,
                                 REFVECS_MIN_VALUES=None,
                                 REFVECS_MAX_VALUES=None,
                                 REFVECS_SPARSITIES=None,
                                 REFVECS_WORD_COUNT=None,
                                 REFVECS_POOL_SIZE=None
                                 )

    SRNL = SXNLClass.SXNL.copy()
    RAW_SUPOBJS = SXNLClass.SXNL_SUPPORT_OBJECTS.copy()

    SXNLClass.expand_data(expand_as_sparse_dict=False, auto_drop_rightmost_column=False)
    SWNL = SXNLClass.SXNL
    WORKING_SUPOBJS = SXNLClass.SXNL_SUPPORT_OBJECTS
    data_given_orientation = SXNLClass.data_current_orientation
    target_given_orientation = SXNLClass.target_current_orientation
    refvecs_given_orientation = SXNLClass.refvecs_current_orientation

    del SXNLClass

    WORKING_CONTEXT = []
    WORKING_KEEP = RAW_SUPOBJS[0][msod.QUICK_POSN_DICT()["HEADER"]]

    standard_config = 'AA'
    nn_config = 'AA'

    split_method = 'None'
    LABEL_RULES = []
    number_of_labels = 1
    event_value = ''
    negative_value = ''

    from ML_PACKAGE.NN_PACKAGE import nn_default_config_params as ndcp
    ARRAY_OF_NODES, NEURONS, nodes, node_seed, activation_constant, aon_base_path, aon_filename, nn_cost_fxn, \
    SELECT_LINK_FXN, LIST_OF_NN_ELEMENTS, NN_OUTPUT_VECTOR, batch_method, BATCH_SIZE, gd_method, conv_method, \
    lr_method, gd_iterations, LEARNING_RATE, momentum_weight, nn_conv_kill, nn_pct_change, nn_conv_end_method, \
    nn_rglztn_type, nn_rglztn_fctr, non_neg_coeffs, allow_summary_print, summary_print_interval, iteration = \
        ndcp.nn_default_config_params()


    cost_fxn = 'minus log-likelihood(l)'
    OUTPUT_VECTOR = []


    NNConfigRun(standard_config, nn_config, SRNL, RAW_SUPOBJS, SWNL, WORKING_SUPOBJS, data_given_orientation,
                 target_given_orientation, refvecs_given_orientation, WORKING_CONTEXT, WORKING_KEEP,
                 split_method, LABEL_RULES, number_of_labels, event_value, negative_value,
                 nn_rglztn_type, nn_rglztn_fctr, nn_conv_kill, nn_pct_change, nn_conv_end_method,
                 ARRAY_OF_NODES, NEURONS, nodes, node_seed, activation_constant, aon_base_path, aon_filename, cost_fxn,
                 SELECT_LINK_FXN, LIST_OF_NN_ELEMENTS, OUTPUT_VECTOR, batch_method, BATCH_SIZE, gd_method, conv_method,
                 lr_method, LEARNING_RATE, momentum_weight, gd_iterations, non_neg_coeffs, allow_summary_print,
                 summary_print_interval, iteration).configrun()















