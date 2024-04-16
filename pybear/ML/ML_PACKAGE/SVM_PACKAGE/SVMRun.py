import numpy as n
from data_validation import validate_user_input as vui
from general_data_ops import get_shape as gs
from ML_PACKAGE import MLRunTemplate as mlrt
from ML_PACKAGE.SVM_PACKAGE import SVMCoreRunCode as scrc, SVMConfig as sc
from ML_PACKAGE.SVM_PACKAGE.print_results import boundary_excel_dump as bed, SVMSummaryStatistics as sss, svm_setup_dump as ssd
    # nn_generic_test_results_dump as fd

from ML_PACKAGE.SVM_PACKAGE import svm_output_calc as svmoc, svm_error_calc as sec, SVMConfig as svmc
from ML_PACKAGE.standard_configs import standard_configs as sc
from ML_PACKAGE.GENERIC_PRINT import general_dev_results_dump as gdrd, general_test_results_dump as gtrd
from MLObjects.SupportObjects import master_support_object_dict as msod

# INHERITED ####################################################################################################################
# MODULE-SPECIFIC MAIN MENU DECLARATIONS ############################################################################
# END MODULE-SPECIFIC MAIN MENU DECLARATIONS ############################################################################

# GENERIC FUNCTIONS ####################################################################################################################
# row_column_display_select()            # user selects rows and columns for display on screen and filedump
# filedump_path()                        # hard-coded directory, user-entered filename
# filedump_general_ml_setup_module()     # module for filedump of general setup for all ML packages, used for train, dev, & test filedump
# dev_or_test_draw_params()              # module that returns the sample size to pull from train data for dev or test data sets
# random_dev_or_test_draw()              # randomly select user-specified quantity of examples for dev & test sets from (remaining) examples in TRAIN_SWNL
# partition_dev_or_test_draw()           # partition (remaining) examples in TRAIN_SWNL for selection into dev or test
# category_dev_or_test_draw()            # select examples for dev & test sets using categories in TRAIN_SWNL
# output_calc()                          # return module for running user-selected data into trained ML algorithm and calculating and sorting results for all ML packages, overwritten in child
# END GENERIC FUNCTIONS ####################################################################################################################

# TRAIN BUILD DEFINITIONS ################################################################################################
# reset_train_data()                     # reset train data to ORIGINAL DATA
# END TRAIN BUILD DEFINITIONS ################################################################################################

# DEV BUILD DEFINITIONS ##################################################################################################
# base_dev_build_module()                # module with code for building dev objects for all ML packages
# dev_build()                            # module for building dev objects, train objects starts as original objects, then dev objects are extracted from train objects
# END DEV BUILD DEFINITIONS ##################################################################################################

# TEST BUILD DEFINITIONS ##################################################################################################
# base_test_build_module()               # module with code for building test objects for all ML packages
# test_build()                           # module for building test objects, train objects starts as original objects, then test objects are extracted from train objects
# END TEST BUILD DEFINITIONS ##################################################################################################

# TRAIN CALC DEFINITIONS ###################################################################################################
# END TRAIN CALC DEFINITIONS ###################################################################################################

# DEV CALC DEFINITIONS ##################################################################################################
# rglztn_partition_iterator()            # only used to save space in base_dev_calc_module()
# base_dev_calc_module()                 # module for performing dev calculations for all ML packages
# END DEV CALC DEFINITIONS ##################################################################################################

# TEST CALC DEFINITIONS ##################################################################################################
# base_test_calc_module()                # module for performing test calculations for all ML packages
# END TEST CALC DEFINITIONS ##################################################################################################

# TRAIN DATA DISPLAY ##############################################################################################################
# END TRAIN DATA DISPLAY ##############################################################################################################

# DEV DATA DISPLAY ###########################################################################################################
# dev_summary_statistics_module()        # returns module for printing summary statistics of dev data for all ML packages
# print_dev_results_module()             # returns module for printing dev results to screen for all ML packages
# dev_filedump_module()                  # returns module for filedump of dev results for all ML packages
# END DEV DATA DISPLAY ############################################################################################A##############

# CALC DATA DISPLAY ############################################################################################A##############
# END CALC DATA DISPLAY ############################################################################################A##############

# base_return_fxn()                      # specify what to return for all ML packages
# run()                                  # MLRunTemplate loop



# OVERWRITTEN ####################################################################################################################
# MODULE-SPECIFIC MAIN MENU DECLARATIONS ############################################################################
#Xmodule_specific_main_menu_cmds()       # returns module-specific top-level menu options, overwritten in child
#Xmodule_specific_main_menu_operations() # returns module-specific execution code for post-run cmds, overwritten in child
# END MODULE-SPECIFIC MAIN MENU DECLARATIONS ############################################################################

# GENERIC FUNCTIONS ####################################################################################################################
#Xhyperparameter_display_module()        # print hyperparameter settings to screen
#Xfiledump_package_specific_setup_module()  # module for filedump of setup for specific package, used for train, dev, & test filedump, overwritten in child
#Xgeneric_ml_core_calc_algorithm()       # return module for package-specific core output algorithm, returns final output vector, overwritten in child
#Xtrain_data_calc_algorithm()            # module for passing train data thru package-specific core algorithm, returns final output vector
#Xdev_data_calc_algorithm()              # module for passing dev data thru package-specific core algorithm, returns final output vector
#Xtest_data_calc_algorithm()             # module for passing test data thru package-specific core algorithm, returns final output vector
#Xgeneric_ml_core_error_algorithm()      # return module for package-specific core error algorithm, returns total error, overwritten in child
#Xtrain_data_error_algorithm()           # module for passing train data thru package-specific error algorithm, returns total error
#Xdev_data_error_algorithm()             # module for passing dev data thru package-specific error algorithm, returns total error
#Xtest_data_error_algorithm()            # module for passing test data th
# END GENERIC FUNCTIONS ####################################################################################################################

# TRAIN BUILD DEFINITIONS #####################################################################################################
# END TRAIN BUILD DEFINITIONS ################################################################################################

# DEV BUILD DEFINITIONS ##################################################################################################
#Xsub_dev_build_module()                 # return module with package-specific dev objects build code, overwritten in child
#Xsub_dev_build_cmds()                   # return list with package-specific dev objects build prompts, overwritten in child
# END DEV BUILD DEFINITIONS ##################################################################################################

# TEST BUILD DEFINITIONS ##################################################################################################
#Xsub_test_build_module()                # return module with package-specific test objects build code, overwritten in child
#Xsub_test_build_cmds()                  # return list with package-specific test objects build prompts, overwritten in child
# END TEST BUILD DEFINITIONS ##################################################################################################

# TRAIN CALC DEFINITIONS ###################################################################################################
#Xcore_training_code()                   # return unique core training algorithm for particular ML package
#Xkfold_core_training_code()             # run code with randomization of nn parameters for k-fold
# END TRAIN CALC DEFINITIONS ###################################################################################################

# DEV CALC DEFINITIONS ##################################################################################################
#Xsub_dev_calc_module()                  # return module with package-specific commands for performing dev calculations
#Xsub_dev_calc_cmds()                    # return list with package-specific dev calc prompts
# END DEV CALC DEFINITIONS ##################################################################################################

# TEST CALC DEFINITIONS ##################################################################################################
#Xsub_test_calc_module()                 # return module with package-specific commands for performing test calculations
#Xsub_test_calc_cmds()                   # return list with package-specific test calc prompts
# END TEST CALC DEFINITIONS ##################################################################################################

# TRAIN DATA DISPLAY ##############################################################################################################
#Xtrain_summary_statistics_module()      # returns module for printing summary statistics of train data for particular ML package
#Xprint_train_results_module()           # returns module for printing train results to screen for particular ML package
#Xtrain_filedump_module()                # returns module for filedump of train results for particular ML package
# END TRAIN DATA DISPLAY ##############################################################################################################

# DEV DATA DISPLAY ###########################################################################################################
# END DEV DATA DISPLAY ###########################################################################################################

# CALC DATA DISPLAY ###########################################################################################################
#Xtest_summary_statistics_module()       # returns module for printing summary statistics of test data for particular ML package
#Xprint_test_results_module()            # returns module for printing test results to screen for particular ML package
#Xtest_filedump_module()                 # returns module for filedump of test results for particular ML package
# END CALC DATA DISPLAY ###########################################################################################################

#Xreturn_fxn()                           # return self.base_return_fxn()




class SVMRun(mlrt.MLRunTemplate):
    def __init__(self, standard_config, svm_config, SUPER_RAW_NUMPY_LIST, RAW_SUPOBJS, SUPER_WORKING_NUMPY_LIST,
                 WORKING_SUPOBJS, data_run_orientation, target_run_orientation, refvecs_run_orientation, WORKING_CONTEXT,
                 WORKING_KEEP, TRAIN_SWNL, DEV_SWNL, TEST_SWNL, split_method, LABEL_RULES, number_of_labels, event_value,
                 negative_value, conv_kill, pct_change, conv_end_method, rglztn_type, rglztn_fctr, bypass_validation,
                 SUPPORT_VECTORS, SUPPORT_TARGETS, SUPPORT_ALPHAS, SUPPORT_KERNELS, b, K, ALPHAS, margin_type, C, cost_fxn,
                 kernel_fxn, constant, exponent, sigma, alpha_seed, alpha_selection_alg, max_passes, tol,
                 SMO_a2_selection_method, new_error_start):

        super().__init__(standard_config, svm_config, SUPER_RAW_NUMPY_LIST, RAW_SUPOBJS, SUPER_WORKING_NUMPY_LIST, WORKING_SUPOBJS,
                 data_run_orientation, target_run_orientation, refvecs_run_orientation, WORKING_CONTEXT, WORKING_KEEP, TRAIN_SWNL,
                 DEV_SWNL, TEST_SWNL, split_method, LABEL_RULES, number_of_labels, event_value, negative_value, conv_kill,
                 pct_change, conv_end_method, rglztn_type, rglztn_fctr, bypass_validation, __name__)


        # OVERWRITE SOME super().__init__()
        self.rglztn_fctr = C

        self.b = b
        self.K = K
        self.ALPHAS = ALPHAS
        self.margin_type = margin_type
        self.C = C
        self.cost_fxn = cost_fxn
        self.kernel_fxn = kernel_fxn
        self.constant = constant
        self.exponent = exponent
        self.sigma = sigma
        self.alpha_seed = alpha_seed
        self.alpha_selection_alg = alpha_selection_alg
        self.max_passes = max_passes
        self.tol = tol
        self.SMO_a2_selection_method = SMO_a2_selection_method

        self.new_error_start = new_error_start

        self.SUPPORT_VECTORS = SUPPORT_VECTORS
        self.SUPPORT_ALPHAS = SUPPORT_ALPHAS
        self.SUPPORT_TARGETS = SUPPORT_TARGETS
        self.SUPPORT_KERNELS = SUPPORT_KERNELS

        self.passes = 0
        self.early_stop_interval = 1e12

        # FOR MANAGING K RECONSTRUCTION DURING K-FOLD
        self.run_data_as = None
        self.return_kernel_as = None
        self.rebuild_kernel = True

        self.DATA_HEADER = self.WORKING_SUPOBJS[0][msod.QUICK_POSN_DICT()["HEADER"]].reshape((1,-1))
        self.TARGET_HEADER = self.WORKING_SUPOBJS[1][msod.QUICK_POSN_DICT()["HEADER"]].reshape((1,-1))


    #  INHERITS FOR NOW ####################################################################################################
    # MODULE-SPECIFIC MAIN MENU DECLARATIONS ############################################################################
    def module_specific_main_menu_cmds(self):
        # module-specific top-level menu options
        # SPECIFIED IN CHILDREN   # CANT USE 'ABCDEFGHIJKLNOPQRSTUVWXZ'
        return {'0': '\ndump boundary info to file', '1': 'set C'}


    def module_specific_main_menu_operations(self):
        # execution code for post-run cmds          # CANT USE 'ABCDEFGHIJKLNOPQRSTUVWXZ'
        if self.post_run_select == '0':
            print(f'\nWrites ALPHAS to file.')
            bed.boundary_excel_dump(self.SUPPORT_VECTORS, self. SUPPORT_TARGETS, self.SUPPORT_ALPHAS, self.b)
            # THIS IS SEPARATE FROM ALL OTHER FILE DUMPS TO ENABLE EASY LOADING OF NN PARAMETERS
        if self.post_run_select == '1':  # THIS IS HERE TO ALLOW CHANGE WITHOUT HAVING TO GO BACK TO ConfigRun (WHICH PRESERVES self.DEV_ERROR)
            self.C = vui.validate_user_float(f'Enter C > ', min=0)
            self.rglztn_fctr = self.C


    # END MODULE-SPECIFIC MAIN MENU DECLARATIONS ############################################################################


    # OVERWRITES METHODS #######################################################################################3
    # GENERIC FUNCTIONS ####################################################################################################################
    def hyperparameter_display_module(self):
        # print hyperparameter settings to screen
        print(f'\nSVM HYPERPARAMETER SETTINGS:')
        svmc.SVMConfig(self.standard_config, self.sub_config, self.SUPER_WORKING_NUMPY_LIST, self.WORKING_SUPOBJS,
                       self.data_run_orientation, self.margin_type, self.cost_fxn, self.kernel_fxn, self.constant,
                       self.exponent, self.sigma, self.alpha_seed, self.max_passes, self.tol, self.C, self.alpha_selection_alg,
                       self.SMO_a2_selection_method, self.conv_kill, self.pct_change, self.conv_end_method,
                       self.bypass_validation).print_parameters()


    '''def row_column_display_select(self):
        INHERITED '''

    '''def filedump_general_ml_setup_module(self):
        INHERITED'''


    def filedump_package_specific_setup_module(self):
        # module for filedump of setup for specific package, used for train, dev, & test filedump, overwritten in child
        self.wb = ssd.svm_setup_dump(self.wb, self.margin_type, self.rglztn_fctr, self.cost_fxn, self.kernel_fxn, self.constant,
                 self.exponent, self.sigma, self.alpha_seed, self.alpha_selection_alg, self.max_passes, self.tol,
                 self.SMO_a2_selection_method)

    '''def dev_or_test_draw_params(self):
        INHERITED'''

    '''def random_dev_or_test_draw(self, TRAIN_SWNL, size):
        INHERITED'''

    '''def partition_dev_or_test_draw(self, TRAIN_SWNL, number_of_partitions, partition_number):
        INHERITED'''

    '''def category_dev_or_test_draw():
        INHERITED'''

    def generic_ml_core_calc_algorithm(self):
        # return module for package-specific core output algorithm, returns final output vector, overwritten in child
        # not in use
        pass


    def train_data_calc_algorithm(self):
        # module for passing train data thru package-specific core algorithm, returns final output vector

        return svmoc.svm_train_output_calc(self.TRAIN_SWNL[1], self.K, self.ALPHAS, self.b)


    def dev_data_calc_algorithm(self):
        # module for passing dev data thru package-specific core algorithm, returns final output vector
        return svmoc.svm_dev_test_output_calc(self.SUPPORT_VECTORS, self.SUPPORT_TARGETS, self.SUPPORT_ALPHAS, self.b,
                      self.DEV_SWNL[0], self.kernel_fxn, self.constant, self.exponent, self.sigma)


    def test_data_calc_algorithm(self):
        # module for passing test data thru package-specific core algorithm, returns final output vector
        return svmoc.svm_dev_test_output_calc(self.SUPPORT_VECTORS, self.SUPPORT_TARGETS, self.SUPPORT_ALPHAS, self.b,
                      self.TEST_SWNL[0], self.kernel_fxn, self.constant, self.exponent, self.sigma)


    def generic_ml_core_error_algorithm(self):
        # module for package-specific core error algorithm, returns total error
        # not in use
        pass


    def train_data_error_algorithm(self):
        # module for passing train data thru package-specific error algorithm, returns total error

        self.TRAIN_OUTPUT_VECTOR = self.train_data_calc_algorithm()

        return sec.svm_error_calc(self.TRAIN_SWNL[1], self.TRAIN_OUTPUT_VECTOR, self.cost_fxn, self.new_error_start)


    def dev_data_error_algorithm(self):
        # module for passing dev data thru package-specific error algorithm, returns total error
        self.DEV_OUTPUT_VECTOR = svmoc.svm_dev_test_output_calc(self.SUPPORT_VECTORS, self.SUPPORT_TARGETS,
            self.SUPPORT_ALPHAS, self.b, self.DEV_SWNL[0], self.kernel_fxn, self.constant, self.exponent, self.sigma)

        return sec.svm_error_calc(self.DEV_SWNL[1], self.DEV_OUTPUT_VECTOR, self.cost_fxn, self.new_error_start)


    def test_data_error_algorithm(self):
        # module for passing test data thru package-specific error algorithm, returns total error
        self.TEST_OUTPUT_VECTOR = svmoc.svm_dev_test_output_calc(self.SUPPORT_VECTORS, self.SUPPORT_TARGETS,
             self.SUPPORT_ALPHAS, self.b, self.TEST_SWNL[0], self.kernel_fxn, self.constant, self.exponent, self.sigma)

        return sec.svm_error_calc(self.TEST_SWNL[1], self.TEST_OUTPUT_VECTOR, self.cost_fxn, self.new_error_start)


    '''def output_calc(self, calc_algorithm, CALC_OBJECT, object_name):
        # return module with commands to run user-selected matrix for all ML packages
        INHERITED'''


    # END GENERIC FUNCTIONS ####################################################################################################################

    # TRAIN BUILD DEFINITIONS ################################################################################################
    '''def reset_train_data(self):  # reset train data to ORIGINAL DATA
        INHERITED'''
    # END TRAIN BUILD DEFINITIONS ################################################################################################

    # DEV BUILD DEFINITIONS ##################################################################################################
    '''def base_dev_build_module(self):
        INHERITED'''


    def sub_dev_build_module(self):
        # return module with package-specific test matrix build commands, overwritten in child
        # overwritten in child
        pass


    def sub_dev_build_cmds(self):
        # return list with package-specific test matrix build commands
        return {}  # SPECIFIED IN CHILDREN   CANT USE 'RSDFTUVBNA'


    '''def dev_build(self):
        INHERITED'''

    # END DEV BUILD DEFINITIONS ##################################################################################################

    # TEST BUILD DEFINITIONS ##################################################################################################
    '''def base_test_build_module(self):
        INHERITED'''


    def sub_test_build_module(self):
        # return module with package-specific test matrix build commands, overwritten in child
        # overwritten in child
        pass


    def sub_test_build_cmds(self):
        # return list with package-specific test matrix build commands
        return {}  # SPECIFIED IN CHILDREN   CANT USE 'DRSPFTUBONA'


    '''def test_build(self):
        INHERITED'''


    # END TEST BUILD DEFINITIONS ##################################################################################################

    # TRAIN CALC DEFINITIONS ###################################################################################################
    def core_training_code(self):
        # unique run code for particular ML package

        if n.array_equiv(self.DEV_SWNL, []):  # PREVENTS CRASH OF _MLPackage IF DEV NOT AVAILABLE
            self.SUPPORT_VECTORS, self.SUPPORT_ALPHAS, self.SUPPORT_TARGETS, self.SUPPORT_KERNELS, self.b, \
            self.ALPHAS, self.K, self.passes = \
            scrc.SVMCoreRunCode(*self.TRAIN_SWNL[:2], self.WORKING_SUPOBJS, self.K, self.ALPHAS, self.kernel_fxn, self.constant,
            self.exponent, self.sigma, self.margin_type, self.rglztn_fctr, self.alpha_seed, self.alpha_selection_alg,
            self.max_passes, self.tol, self.SMO_a2_selection_method, self.conv_kill, self.pct_change, self.conv_end_method).run()

        else:
            self.SUPPORT_VECTORS, self.SUPPORT_ALPHAS, self.SUPPORT_TARGETS, self.SUPPORT_KERNELS, self.b, \
            self.ALPHAS, self.K, self.passes, self.run_data_as, self.return_kernel_as = \
            scrc.SVMCoreRunCode_MLPackage(self.TRAIN_SWNL, self.DEV_SWNL, self.WORKING_SUPOBJS, self.K, self.ALPHAS,
            self.kernel_fxn, self.constant, self.exponent, self.sigma, self.margin_type, self.rglztn_fctr, self.alpha_seed,
            self.cost_fxn, self.alpha_selection_alg, self.max_passes, self.tol, self.SMO_a2_selection_method,
            self.early_stop_interval, self.conv_kill, self.pct_change, self.conv_end_method, self.run_data_as,
            self.return_kernel_as, self.rebuild_kernel).run()


    def kfold_core_training_code(self):
        # run code with randomization of nn parameters for k-fold

        # 7-4-22 NOT SURE IF RESETTING ALPHAS TO ZERO IS HELPFUL HERE

        self.SUPPORT_VECTORS, self.SUPPORT_ALPHAS, self.SUPPORT_TARGETS, self.SUPPORT_KERNELS, self.b, \
        self.ALPHAS, self.K, self.passes, self.run_data_as, self.return_kernel_as = \
            scrc.SVMCoreRunCode_MLPackage(self.TRAIN_SWNL, self.DEV_SWNL, self.WORKING_SUPOBJS, self.K, self.ALPHAS,
            self.kernel_fxn, self.constant, self.exponent, self.sigma, self.margin_type, self.rglztn_fctr, self.alpha_seed,
            self.cost_fxn, self.alpha_selection_alg, self.max_passes, self.tol, self.SMO_a2_selection_method,
            self.early_stop_interval, self.conv_kill, self.pct_change, self.conv_end_method, self.run_data_as,
            self.return_kernel_as, self.rebuild_kernel).run()


    # END TRAIN CALC DEFINITIONS ###################################################################################################

    # DEV CALC DEFINITIONS ##################################################################################################

    '''def rglztn_partition_iterator(self, number_of_partitions):
        INHERITED'''


    ''''def base_dev_calc_module(self):
        INHERITED'''


    def sub_dev_calc_module(self):
        # return module with package-specific commands to run current dev matrix
        # overwritten in child
        pass


    def sub_dev_calc_cmds(self):
        # return list with package - specific dev calc commands
        return {}  # SPECIFIED IN CHILDREN   CANT USE 'DSA'


    # END DEV CALC DEFINITIONS ##################################################################################################

    # TEST CALC DEFINITIONS ##################################################################################################
    '''def base_test_calc_module(self):
        INHERITED'''


    def sub_test_calc_module(self):
        # return module with package-specific commands to run current test matrix
        # BEAR FIX
        if self.test_calc_select == 'S':
            self.CSUTM_DF = sc.test_cases_calc_standard_configs(self.standard_config, self.TEST_SWNL[0],
                self.TRAIN_SWNL[0], self.TRAIN_SWNL[1], self.margin_type, self.C, self.cost_fxn, self.kernel_fxn,
                self.constant, self.exponent, self.sigma, self.alpha_seed, self.alpha_selection_alg, self.max_passes,
                self.tol, self.SMO_a2_selection_method)


    def sub_test_calc_cmds(self):
        # return list with package - specific test calc commands
        return {'s': 'run special NN test from standard configs'}
                # SPECIFIED IN CHILDREN   CANT USE 'NA'


    # END TEST CALC DEFINITIONS ##################################################################################################

    # TRAIN DATA DISPLAY ##############################################################################################################

    def train_summary_statistics_module(self):
        # returns module for printing summary statistics of train data
        if gs.get_shape('TRAIN_DATA', self.TRAIN_SWNL[0], self.data_run_orientation)[0] != \
                gs.get_shape('KERNEL', self.K, 'ROW')[0]:
            print(f'\n*** TRAIN DATA DOES NOT MATCH KERNEL. MUST RERUN TRAIN DATA. ***\n')
        else:
            # BEAR FIX
            error = self.train_data_error_algorithm()  # DO THIS OUT HERE TO ENSURE self.TRAIN_OUTPUT_VECTOR IS UPDATED B4 sss
            sss.SVMSummaryStatisticsPrint(self.TRAIN_OUTPUT_VECTOR, self.SUPER_WORKING_NUMPY_LIST[1],
                self.TARGET_HEADER, self.ALPHAS, self.SUPPORT_VECTORS,
                self.SUPPORT_TARGETS, self.SUPPORT_ALPHAS, self.b, error, self.passes, self.max_passes).print()


    def print_train_results_module(self):
        # returns module for printing train results for particular ML package

        if gs.get_shape('TRAIN_DATA', self.TRAIN_SWNL[0], self.data_run_orientation)[0] != \
            gs.get_shape('KERNEL', self.K, 'ROW')[0]:
                print(f'\n*** TRAIN DATA DOES NOT MATCH KERNEL. MUST RERUN TRAIN DATA. ***\n')
        else:
            error = self.train_data_error_algorithm()  # DO THIS OUT HERE TO ENSURE self.TRAIN_OUTPUT_VECTOR IS UPDATED B4 sss
            sss.SVMSummaryStatisticsPrint(self.TRAIN_OUTPUT_VECTOR, self.SUPER_WORKING_NUMPY_LIST[1], self.TARGET_HEADER,
                self.ALPHAS, self.SUPPORT_VECTORS, self.SUPPORT_TARGETS, self.SUPPORT_ALPHAS, self.b, error, self.passes,
                self.max_passes).print()


    def train_filedump_module(self):
        # returns module for filedump of train results for particular ML package

        self.filedump_general_ml_setup_module()

        self.wb = ssd.svm_setup_dump(self.wb, self.margin_type, self.rglztn_fctr, self.cost_fxn, self.kernel_fxn,
             self.constant, self.exponent, self.sigma, self.alpha_seed, self.alpha_selection_alg, self.max_passes,
             self.tol, self.SMO_a2_selection_method)

        if gs.get_shape('TRAIN_DATA', self.TRAIN_SWNL[0], self.data_run_orientation)[0] != \
                gs.get_shape('KERNEL', self.K, 'ROW')[0]:
            print(f'\n*** TRAIN DATA DOES NOT MATCH KERNEL. MUST RERUN TRAIN DATA. ***\n')
        else:
            error = self.train_data_error_algorithm()  # PUT THIS HERE TO ENSURE self.TRAIN_OUTPUT_VECTOR IS UPDATED
            self.wb = sss.SVMSummaryStatisticsDump(self.wb, 'TRAIN', self.TRAIN_OUTPUT_VECTOR, self.TRAIN_SWNL[1],
                self.TARGET_HEADER, self.ALPHAS, self.SUPPORT_VECTORS, self.SUPPORT_TARGETS, self.SUPPORT_ALPHAS, self.b,
                error, self.passes, self.max_passes).dump()

    # END TRAIN DATA DISPLAY ##############################################################################################################

    # DEV DATA DISPLAY ##############################A##############################A##############################A##############
    def dev_summary_statistics_module(self):
        # returns module for printing summary statistics of dev data
        error = self.dev_data_error_algorithm()
        sss.SVMSummaryStatisticsPrint(self.DEV_OUTPUT_VECTOR, self.DEV_SWNL[1], self.TARGET_HEADER, self.ALPHAS,
            self.SUPPORT_VECTORS, self.SUPPORT_TARGETS, self.SUPPORT_ALPHAS, self.b, error, self.passes,
            self.max_passes).print()


    '''def print_dev_results_module(self):
        # returns module for printing dev results to screen for all ML packages
        pass
        INHERITED'''


    def dev_filedump_module(self):
        # returns module for filedump of dev results for all ML packages
        # DEV OBJECTS OTHER THAN self.DEV_ERROR MAY NOT EXIST IF DID K-FOLD CV
        self.wb = gdrd.general_dev_results_dump(self.wb, self.DEV_ERROR, self.RGLZTN_FACTORS)

        try:
            error = self.dev_data_error_algorithm()  # DO THIS FIRST TO ENSURE self.DEV_OUTPUT_VECTOR IS UPDATED B4 sss
            self.wb = sss.SVMSummaryStatisticsDump(self.wb, 'DEV STATISTICS', self.DEV_OUTPUT_VECTOR, self.DEV_SWNL[1], self.TARGET_HEADER,
            self.ALPHAS, self.SUPPORT_VECTORS, self.SUPPORT_TARGETS, self.SUPPORT_ALPHAS, self.b, error, self.passes,
            self.max_passes).dump()
        except: pass

    # END DEV DATA DISPLAY ##############################A##############################A##############################A##############

    # CALC DATA DISPLAY ##############################A##############################A##############################A##############
    def test_summary_statistics_module(self):
        # returns module for printing summary statistics of test data
        error = self.test_data_error_algorithm()   # DO THIS OUT HERE TO ENSURE self.TEST_OUTPUT_VECTOR IS UPDATED B4 sss
        sss.SVMSummaryStatisticsPrint(self.TEST_OUTPUT_VECTOR, self.TEST_SWNL[1], self.TARGET_HEADER, self.ALPHAS,
            self.SUPPORT_VECTORS, self.SUPPORT_TARGETS, self.SUPPORT_ALPHAS, self.b, error, self.passes,
            self.max_passes).print()


    '''
    def print_test_results_module(self):
        pass
        INHERITED
        '''


    def test_filedump_module(self):
        # package-specific module for saving test results
        self.wb = gtrd.general_test_results_dump(self.wb, self.CSUTM_DF, self.DISPLAY_COLUMNS, self.display_select, self.display_rows)
        error = self.test_data_error_algorithm()  # PUT THIS HERE TO ENSURE self.TEST_OUTPUT_VECTOR IS UPDATED
        self.wb = sss.SVMSummaryStatisticsDump(self.wb, 'TEST STATISTICS', self.TEST_OUTPUT_VECTOR, self.TEST_SWNL[1], self.TARGET_HEADER,
            self.ALPHAS, self.SUPPORT_VECTORS, self.SUPPORT_TARGETS, self.SUPPORT_ALPHAS, self.b, error, self.passes,
            self.max_passes).dump()

    # END CALC DATA DISPLAY ##############################A##############################A##############################A##############

    '''def base_return_fxn(self):
        INHERITED'''


    def return_fxn(self):
        return *self.base_return_fxn(), self.SUPPORT_VECTORS, self.SUPPORT_TARGETS, self.SUPPORT_ALPHAS, \
               self.SUPPORT_KERNELS, self.b, self.K, self.ALPHAS


    '''def run(self):  # MLRunTemplate loop
        INHERITED'''





if __name__ == '__main__':
    pass










