import numpy as np
from copy import deepcopy
from ML_PACKAGE import MLConfigRunTemplate as mlcrt
from ML_PACKAGE.SVM_PACKAGE import SVMConfig as sc, SVMRun as sr
from MLObjects.SupportObjects import master_support_object_dict as msod
from MLObjects import MLObject as mlo
from MLObjects.ObjectOrienter import MLObjectOrienter as mloo



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
class SVMConfigRun(mlcrt.MLConfigRunTemplate):

    def __init__(self, standard_config, svm_config, SUPER_RAW_NUMPY_LIST, RAW_SUPOBJS, SUPER_WORKING_NUMPY_LIST,
        WORKING_SUPOBJS, data_given_orientation, target_given_orientation, refvecs_given_orientation, WORKING_CONTEXT,
        WORKING_KEEP, split_method, LABEL_RULES, number_of_labels, event_value, negative_value, conv_kill, pct_change,
        conv_end_method,

        # SVM SPECIFIC PARAMS
        C, max_passes, tol, K, ALPHAS, b, margin_type, cost_fxn, kernel_fxn, constant, exponent, sigma, alpha_seed,
        alpha_selection_alg, SMO_a2_selection_method):

        data_run_format = 'AS_GIVEN'
        data_run_orientation = 'COLUMN'
        target_run_format = 'ARRAY'
        target_run_orientation = 'COLUMN'
        refvecs_run_format = 'ARRAY'
        refvecs_run_orientation = 'COLUMN'

        super().__init__(standard_config, svm_config, SUPER_RAW_NUMPY_LIST, RAW_SUPOBJS, SUPER_WORKING_NUMPY_LIST,
            WORKING_SUPOBJS, data_given_orientation, target_given_orientation, refvecs_given_orientation, data_run_orientation,
            target_run_orientation, refvecs_run_orientation, data_run_format, target_run_format, refvecs_run_format,
            WORKING_CONTEXT, WORKING_KEEP, split_method, LABEL_RULES, number_of_labels, event_value, negative_value,
            conv_kill, pct_change, conv_end_method, 'dummy rglztn type', C, __name__)


        fxn = '__init__'

        # SVM SPECIFIC __init__
        self.C = self.rglztn_fctr
        self.max_passes = max_passes
        self.tol = tol
        self.K = K
        self.ALPHAS = ALPHAS
        self.b = b
        self.margin_type = margin_type
        self.cost_fxn = cost_fxn
        self.kernel_fxn = kernel_fxn
        self.constant = constant
        self.exponent = exponent
        self.sigma = sigma
        self.alpha_seed = alpha_seed
        self.alpha_selection_alg = alpha_selection_alg
        self.SMO_a2_selection_method = SMO_a2_selection_method

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


        # PLACEHOLDERS
        self.SUPPORT_VECTORS = []
        self.SUPPORT_TARGETS = []
        self.SUPPORT_ALPHAS = []
        self.SUPPORT_KERNELS = []

        self.new_error_start = float('inf')



    # INHERITS #############################################################################################################
    # intercept_finder()                 Not used here. Locate an intercept column in DATA and handle any anamolous columns of constants.
    #                                    As of 11/15/22 only for MLR, MI, and GMLR.
    # run_module_input_tuple()           tuple of base params that pass into run_module for all ML packages
    # return_fxn_base()                  values returned from all ML packages
    # base_post_run_options_module()     holds post run options applied to all ML packages
    # configrun()                        runs config_module() & run_module()


    # OVERWRITES #######################################################################################################
    def config_module(self):
        self.SUPPORT_VECTORS, self.SUPPORT_TARGETS, self.SUPPORT_ALPHAS, self.b, self.margin_type, self.C, \
        self.cost_fxn, self.kernel_fxn, self.constant, self.exponent, self.sigma, self.alpha_seed, \
        self.alpha_selection_alg, self.max_passes, self.tol, self.SMO_a2_selection_method, self.conv_kill, self.pct_change,\
        self.conv_end_method = \
            sc.SVMConfig(self.standard_config, self.sub_config, self.SUPER_WORKING_NUMPY_LIST, self.WORKING_SUPOBJS,
                self.data_run_orientation, self.margin_type, self.cost_fxn, self.kernel_fxn, self.constant, self.exponent,
                self.sigma, self.alpha_seed, self.max_passes, self.tol, self.C, self.alpha_selection_alg,
                self.SMO_a2_selection_method, self.conv_kill, self.pct_change, self.conv_end_method, self.bypass_validation).config()


    def run_module(self):
        # *self.run_module_input_tuple(), \
        self.TRAIN_SWNL, self.DEV_SWNL, self.TEST_SWNL, self.PERTURBER_RESULTS, self.SUPPORT_VECTORS, self.SUPPORT_TARGETS, \
        self.SUPPORT_ALPHAS, self.SUPPORT_KERNELS, self.b, self.K, self.ALPHAS = \
            sr.SVMRun(*self.run_module_input_tuple(), self.SUPPORT_VECTORS, self.SUPPORT_TARGETS, self.SUPPORT_ALPHAS,
            self.SUPPORT_KERNELS, self.b, self.K, self.ALPHAS, self.margin_type, self.C, self.cost_fxn, self.kernel_fxn,
            self.constant, self.exponent, self.sigma, self.alpha_seed, self.alpha_selection_alg, self.max_passes,
            self.tol, self.SMO_a2_selection_method, self.new_error_start).run()

        # run_module_input_tuple()
        # self.standard_config, self.sub_config, self.SUPER_RAW_NUMPY_LIST, self.RAW_SUPOBJS, self.SUPER_WORKING_NUMPY_LIST, \
        # self.WORKING_SUPOBJS, self.data_run_orientation, self.target_run_orientation, self.refvecs_run_orientation, \
        # self.WORKING_CONTEXT, self.WORKING_KEEP, self.TRAIN_SWNL, self.DEV_SWNL, self.TEST_SWNL, self.split_method, \
        # self.LABEL_RULES, self.number_of_labels, self.event_value, self.negative_value, self.conv_kill, self.pct_change, \
        # self.conv_end_method, self.rglztn_type, self.rglztn_fctr, self.bypass_validation

        # ARGS FOR SVMRun
        # standard_config, svm_config, SUPER_RAW_NUMPY_LIST, RAW_SUPOBJS, SUPER_WORKING_NUMPY_LIST,
        # WORKING_SUPOBJS, data_run_orientation, target_run_orientation, refvecs_run_orientation, WORKING_CONTEXT,
        # WORKING_KEEP, TRAIN_SWNL, DEV_SWNL, TEST_SWNL, split_method, LABEL_RULES, number_of_labels, event_value,
        # negative_value, conv_kill, pct_change, conv_end_method, rglztn_fctr, bypass_validation, SUPPORT_VECTORS,
        # SUPPORT_TARGETS, SUPPORT_ALPHAS, SUPPORT_KERNELS, b, K, ALPHAS, margin_type, C, cost_fxn, kernel_fxn,
        # constant, exponent, sigma, alpha_seed, alpha_selection_alg, max_passes, tol, SMO_a2_selection_method,
        # new_error_start


    def return_fxn(self):
    # returns user-specified output, in addition to return_fxn_base()
        return *self.return_fxn_base(), self.C, self.max_passes, self.tol, self.K, self.ALPHAS, self.b, self.margin_type, \
               self.cost_fxn, self.kernel_fxn, self.constant, self.exponent, self.sigma, self.alpha_seed, \
               self.alpha_selection_alg, self.SMO_a2_selection_method

    # INHERITS FOR NOW ##############################################
    #######################################################
    '''
    def sub_post_run_cmds(self):
    # package-specific options available to modify WORKING_DATA after run
        return []


    def sub_post_run_options_module(self):
    # holds post run options unique to particular ML package
        pass
    '''



if __name__ == '__main__':
    import random
    from MLObjects.TestObjectCreators.SXNL import CreateSXNL as csxnl
    from MLObjects.TestObjectCreators import test_header as th

    rows = 200
    cols = 2

    DATA = [[], []]
    TARGET = [[]]
    gap = 5
    gap_slope = -1
    gap_y_int = -5
    length = 200
    while True:
        x = 10 * np.random.randn()
        y = 10 * np.random.randn()
        if y > gap_slope * x + (gap_y_int - 0.5 * gap) and y < gap_slope * x + (gap_y_int + 0.5 * gap): continue
        DATA[0].append(x)
        DATA[1].append(y)
        if y < gap_slope * x + (gap_y_int - 0.5 * gap): TARGET[0].append(-1)
        if y > gap_slope * x + (gap_y_int + 0.5 * gap): TARGET[0].append(1)
        if len(DATA[0]) == length: break


    DATA = np.array(DATA, dtype=object)  # [ [] = COLUMNS ]
    # DATA = np.random.randint(-9, 10, [rows, cols]).transpose()
    DATA_HEADER = th.test_header(cols)
    DATA_HEADER.reshape((1,-1))
    data_given_orientation = 'COLUMN'
    data_return_orientation = 'COLUMN'
    data_return_format = 'ARRAY'

    # TARGET = np.fromiter((-1 if _ < 0 else 1 for _ in np.random.randn(rows)), dtype=int)
    # TARGET.resize(1,TARGET.size)
    TARGET = np.array(TARGET, dtype=object)
    TARGET_HEADER = np.array(['TARGET'], dtype='<U6').reshape((1, 1))
    target_given_orientation = 'COLUMN'
    target_return_orientation = 'COLUMN'
    target_return_format = 'ARRAY'

    REFERENCE = np.fromiter(range(rows), dtype=np.int32)
    REFERENCE_HEADER = np.array(['REF'], dtype='<U3').reshape((1, 1))
    refvecs_given_orientation = 'COLUMN'
    refvecs_return_orientation = 'COLUMN'
    refvecs_return_format = 'ARRAY'


    SXNLClass = csxnl.CreateSXNL(rows=None,
                                 bypass_validation=False,
                                 data_return_format=data_return_format,
                                 data_return_orientation=data_return_orientation,
                                 DATA_OBJECT=DATA,
                                 DATA_OBJECT_HEADER=DATA_HEADER,
                                 data_override_sup_obj=False,
                                 data_given_orientation=data_given_orientation,

                                 target_return_format=target_return_format,
                                 target_return_orientation=target_return_orientation,
                                 TARGET_OBJECT=TARGET,
                                 TARGET_OBJECT_HEADER=TARGET_HEADER,
                                 target_type='FLOAT',
                                 target_override_sup_obj=False,
                                 target_given_orientation=target_given_orientation,

                                 refvecs_return_format=refvecs_return_format,
                                 refvecs_return_orientation=refvecs_return_orientation,
                                 REFVECS_OBJECT=REFERENCE,
                                 REFVECS_OBJECT_HEADER=REFERENCE_HEADER,
                                 refvecs_given_orientation=refvecs_given_orientation,
                                 )

    SUPER_RAW_NUMPY_LIST = SXNLClass.SXNL.copy()
    RAW_SUPOBJS = SXNLClass.SXNL_SUPPORT_OBJECTS.copy()

    # SXNLClass.expand_data(expand_as_sparse_dict={'P':True,'A':False}[vui.validate_user_str(f'\nExpand as sparse dict(p) or array(a) > ', 'AP')],
    #                       auto_drop_rightmost_column=False)
    SUPER_WORKING_NUMPY_LIST = SXNLClass.SXNL.copy()
    WORKING_SUPOBJS = SXNLClass.SXNL_SUPPORT_OBJECTS.copy()

    data_given_orientation = SXNLClass.data_current_orientation
    target_given_orientation = SXNLClass.target_current_orientation
    refvecs_given_orientation = SXNLClass.refvecs_current_orientation

    del SXNLClass

    WORKING_CONTEXT = []
    WORKING_KEEP = WORKING_SUPOBJS[0][msod.QUICK_POSN_DICT()["HEADER"]]


    standard_config = 'AA'
    svm_config = ''
    split_method = 'None'
    LABEL_RULES = ''
    number_of_labels = 1
    event_value = 1
    negative_value = -1
    rglztn_fctr = 0
    K = [[]]
    ALPHAS = []
    b = 0
    margin_type = 'SOFT'
    C = float('inf')
    cost_fxn = 'C'
    kernel_fxn = 'LINEAR'
    constant = 0
    exponent = 1
    sigma = 1
    alpha_seed = 0
    alpha_selection_alg = 'SMO'
    max_passes = 10000
    tol = .001
    SMO_a2_selection_method = 'RANDOM'
    conv_kill = 500
    pct_change = 0
    conv_end_method = 'PROMPT'





    SUPER_RAW_NUMPY_LIST, RAW_SUPOBJS, SUPER_WORKING_NUMPY_LIST, WORKING_SUPOBJS, WORKING_CONTEXT, WORKING_KEEP, \
    split_method, LABEL_RULES, number_of_labels, event_value, negative_value, svm_conv_kill, svm_pct_change, \
    svm_conv_end_method, dum_rglztn_type, dum_rglztn_fctr, C, max_passes, tol, K, ALPHAS, b, margin_type, \
    cost_fxn, kernel_fxn, constant, exponent, sigma, alpha_seed, alpha_selection_alg, SMO_a2_selection_method = \
        SVMConfigRun(standard_config, svm_config, SUPER_RAW_NUMPY_LIST, RAW_SUPOBJS, SUPER_WORKING_NUMPY_LIST,
        WORKING_SUPOBJS, data_given_orientation, target_given_orientation, refvecs_given_orientation, WORKING_CONTEXT,
        WORKING_KEEP, split_method, LABEL_RULES, number_of_labels, event_value, negative_value, conv_kill, pct_change,
        conv_end_method, C, max_passes, tol, K, ALPHAS, b, margin_type, cost_fxn, kernel_fxn, constant, exponent, sigma,
        alpha_seed, alpha_selection_alg, SMO_a2_selection_method).configrun()


















































