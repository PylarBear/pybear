import numpy as n, pandas as p
from copy import deepcopy
from debug import IdentifyObjectAndPrint as ioap
from ML_PACKAGE import MLRunTemplate as mlrt
from ML_PACKAGE.SVD import SVDCoreRunCode as scrc, SVDConfig as sic, \
    svd_output_vector_calc as sovc, svd_error_calc as siec
from ML_PACKAGE.SVD.print_results import SVDSummaryStatistics as siss, svd_setup_dump as sisd, \
    svd_train_results_dump as sitrd
from ML_PACKAGE.GENERIC_PRINT import general_ml_setup_dump as gmsd, general_test_results_dump as gterd
from general_list_ops import list_select as ls
import openpyxl as xl
from openpyxl import Workbook


# INHERITED ####################################################################################################################
# MODULE-SPECIFIC MAIN MENU DECLARATIONS ############################################################################
# END MODULE-SPECIFIC MAIN MENU DECLARATIONS ############################################################################

# GENERIC FUNCTIONS ####################################################################################################################
# row_column_display_select()            # user selects rows and columns for display on screen and filedump
# filedump_path()                        # hard-coded directory, user-entered filename
# filedump_general_ml_setup_module()     # module for filedump of general setup for all ML packages, used for train, dev, & tests filedump
# dev_or_test_draw_params()              # module that returns the sample size to pull from train data for dev or tests data sets
# random_dev_or_test_draw()              # randomly select user-specified quantity of examples for dev & tests sets from (remaining) examples in TRAIN_SWNL
# partition_dev_or_test_draw()           # partition (remaining) examples in TRAIN_SWNL for selection into dev or tests
# category_dev_or_test_draw()            # select examples for dev & tests sets using categories in TRAIN_SWNL
# output_calc()                          # return module for running user-selected data into trained ML algorithm and calculating and sorting results for all ML packages, overwritten in child
# END GENERIC FUNCTIONS ####################################################################################################################

# TRAIN BUILD DEFINITIONS ################################################################################################
# reset_train_data()                     # reset train data to ORIGINAL DATA
# END TRAIN BUILD DEFINITIONS ################################################################################################

# DEV BUILD DEFINITIONS ##################################################################################################
# END DEV BUILD DEFINITIONS ##################################################################################################

# TEST BUILD DEFINITIONS ##################################################################################################
# base_test_build_module()               # module with code for building tests objects for all ML packages
# test_build()                           # module for building tests objects, train objects starts as original objects, then tests objects are extracted from train objects
# END TEST BUILD DEFINITIONS ##################################################################################################

# TRAIN CALC DEFINITIONS ###################################################################################################
# END TRAIN CALC DEFINITIONS ###################################################################################################

# DEV CALC DEFINITIONS ##################################################################################################
# rglztn_partition_iterator()            # only used to save space in base_dev_calc_module()
# END DEV CALC DEFINITIONS ##################################################################################################

# TEST CALC DEFINITIONS ##################################################################################################
# base_test_calc_module()                # module for performing tests calculations for all ML packages
# END TEST CALC DEFINITIONS ##################################################################################################

# TRAIN DATA DISPLAY ##############################################################################################################
# END TRAIN DATA DISPLAY ##############################################################################################################

# DEV DATA DISPLAY ###########################################################################################################
# END DEV DATA DISPLAY ############################################################################################A##############

# CALC DATA DISPLAY ############################################################################################A##############
# END CALC DATA DISPLAY ############################################################################################A##############

# base_return_fxn()                      # specify what to return for all ML packages
# run()                                  # MLRunTemplate loop



# OVERWRITTEN ####################################################################################################################
# MODULE-SPECIFIC MAIN MENU DECLARATIONS ############################################################################
#Xmodule_specific_main_menu_cmds()       # returns module-specific top-level menu options, overwritten in child
#Xmodule_specific_main_menu_str()        # returns module-specific string of allowed cmds for top-level menu, overwritten in child
#Xmodule_specific_main_menu_operations() # returns module-specific execution code for post-run cmds, overwritten in child
# END MODULE-SPECIFIC MAIN MENU DECLARATIONS ############################################################################

# GENERIC FUNCTIONS ####################################################################################################################
#Xhyperparameter_display_module()        # print hyperparameter settings to screen
#Xfiledump_package_specific_setup_module()  # module for filedump of setup for specific package, used for train, dev, & tests filedump, overwritten in child
#Xgeneric_ml_core_calc_algorithm()       # return module for package-specific core output algorithm, returns final output vector, overwritten in child
#Xtrain_data_calc_algorithm()            # module for passing train data thru package-specific core algorithm, returns final output vector
#Xdev_data_calc_algorithm()              # module for passing dev data thru package-specific core algorithm, returns final output vector
#Xtest_data_calc_algorithm()             # module for passing dev data thru package-specific core algorithm, returns final output vector
#Xgeneric_ml_core_error_algorithm()      # return module for package-specific core error algorithm, returns total error, overwritten in child
#Xtrain_data_error_algorithm()           # module for passing train data thru package-specific error algorithm, returns total error
#Xdev_data_error_algorithm()             # module for passing dev data thru package-specific error algorithm, returns total error
#Xtest_data_error_algorithm()            # module for passing tests data th
# END GENERIC FUNCTIONS ####################################################################################################################

# TRAIN BUILD DEFINITIONS #####################################################################################################
# END TRAIN BUILD DEFINITIONS ################################################################################################

# DEV BUILD DEFINITIONS ##################################################################################################
#Xbase_dev_build_module()                # module with code for building dev objects for all ML packages
#Xsub_dev_build_module()                 # return module with package-specific dev objects build code, overwritten in child
#Xsub_dev_build_cmds()                   # return list with package-specific dev objects build prompts, overwritten in child
#Xsub_dev_build_str()                    # return string with package-specific allowed dev objects build commands, overwritten in child
#Xdev_build()                            # module for building dev objects, train objects starts as original objects, then dev objects are extracted from train objects
# END DEV BUILD DEFINITIONS ##################################################################################################

# TEST BUILD DEFINITIONS ##################################################################################################
#Xsub_test_build_module()                # return module with package-specific tests objects build code, overwritten in child
#Xsub_test_build_cmds()                  # return list with package-specific tests objects build prompts, overwritten in child
#Xsub_test_build_str()                   # return string with package-specific allowed tests objects build commands, overwritten in child
# END TEST BUILD DEFINITIONS ##################################################################################################

# TRAIN CALC DEFINITIONS ###################################################################################################
#Xcore_training_code()                   # return unique core training algorithm for particular ML package
# END TRAIN CALC DEFINITIONS ###################################################################################################

# DEV CALC DEFINITIONS ##################################################################################################
#Xbase_dev_calc_module()                 # module for performing dev calculations for all ML packages
#Xsub_dev_calc_module()                  # return module with package-specific commands for performing dev calculations
#Xsub_dev_calc_cmds()                    # return list with package-specific dev calc prompts
#Xsub_dev_calc_str()                     # return string with package-specific allowed dev calc commands
# END DEV CALC DEFINITIONS ##################################################################################################

# TEST CALC DEFINITIONS ##################################################################################################
#Xsub_test_calc_module()                 # return module with package-specific commands for performing tests calculations
#Xsub_test_calc_cmds()                   # return list with package-specific tests calc prompts
#Xsub_test_calc_str()                    # return string with package-specific allowed tests calc commands
# END TEST CALC DEFINITIONS ##################################################################################################

# TRAIN DATA DISPLAY ##############################################################################################################
#Xtrain_summary_statistics_module()      # returns module for printing summary statistics of train data for particular ML package
#Xprint_train_results_module()           # returns module for printing train results to screen for particular ML package
#Xtrain_filedump_module()                # returns module for filedump of train results for particular ML package
# END TRAIN DATA DISPLAY ##############################################################################################################

# DEV DATA DISPLAY ###########################################################################################################
#Xdev_summary_statistics_module()        # returns module for printing summary statistics of dev data for all ML packages
#Xprint_dev_results_module()             # returns module for printing dev results to screen for all ML packages
#Xdev_filedump_module()                  # returns module for filedump of dev results for all ML packages
# END DEV DATA DISPLAY ###########################################################################################################

# CALC DATA DISPLAY ###########################################################################################################
#Xtest_summary_statistics_module()       # returns module for printing summary statistics of tests data for particular ML package
#Xprint_test_results_module()            # returns module for printing tests results to screen for particular ML package
#Xtest_filedump_module()                 # returns module for filedump of tests results for particular ML package
# END CALC DATA DISPLAY ###########################################################################################################

#Xreturn_fxn()                           # return self.base_return_fxn()

#  UNIQUE ###############################################################################################################
#  NONE AS OF 2-12-22


class SVDRun(mlrt.MLRunTemplate):

    def __init__(self, standard_config, svd_config, SUPER_RAW_NUMPY_LIST, SUPER_WORKING_NUMPY_LIST, WORKING_VALIDATED_DATATYPES,
                 WORKING_MODIFIED_DATATYPES, WORKING_FILTERING, WORKING_MIN_CUTOFFS, WORKING_USE_OTHER, WORKING_CONTEXT,
                 WORKING_KEEP, WORKING_SCALING, TRAIN_SWNL, DEV_SWNL, TEST_SWNL, split_method, LABEL_RULES, number_of_labels,
                 event_value, negative_value, conv_kill, pct_change, conv_end_method, rglztn_fctr, svd_max_columns, SVD_TRAIN_RESULTS):

        super().__init__(standard_config, svd_config, SUPER_RAW_NUMPY_LIST, SUPER_WORKING_NUMPY_LIST,
                WORKING_VALIDATED_DATATYPES, WORKING_MODIFIED_DATATYPES, WORKING_FILTERING, WORKING_MIN_CUTOFFS,
                WORKING_USE_OTHER, WORKING_CONTEXT, WORKING_KEEP, WORKING_SCALING, TRAIN_SWNL, DEV_SWNL, TEST_SWNL, split_method,
                LABEL_RULES, number_of_labels, event_value, negative_value, conv_kill, pct_change, conv_end_method, rglztn_fctr,
                __name__)

        self.SUPER_WORKING_NUMPY_LIST = SUPER_WORKING_NUMPY_LIST

        # SVD PARAMETERS:
        self.max_columns = svd_max_columns



        # PLACEHOLDERS
        self.X_TRIAL = []
        self.X_TRIAL_HEADER = []
        self.WINNING_COLUMNS = []
        self.COEFFS = []
        self.SCORES = []

        self.tc_method = 'SVD'

        # 4-17-22 REMOVE K-FOLD CV OPTION FROM MAIN MENU
        self.BASE_MAIN_MENU_CMDS = [_ for _ in self.BASE_MAIN_MENU_CMDS if True not in [__ in _ for __ in ['(w)']]]
        self.base_main_menu_str = ''.join([_ for _ in self.base_main_menu_str if True not in [__ in _ for __ in 'W']])

        # 4-17-22 CREATE UNIQUE OBJECT FOR SVD TO ALLOW FOR DISPLAY/DUMP OF TRAIN RESULTS
        self.TRAIN_RESULTS = SVD_TRAIN_RESULTS      # IS EXCHANGABLE BETWEEN SVDConfigRun & SVDRun


    #  INHERITS FOR NOW ####################################################################################################
    # MODULE-SPECIFIC MAIN MENU DECLARATIONS ############################################################################
    def module_specific_main_menu_cmds(self):
        # module-specific top-level menu options
        return []   # CANT USE 'ABCDEFGHIJKLNOPQRSTUVWXZ'
    
    
    def module_specific_main_menu_str(self):
        # module-specific string of allowed cmds for top-level menu
        return ''   # CANT USE 'ABCDEFGHIJKLNOPQRSTUVWXZ'
    
    
    def module_specific_main_menu_operations(self):
        # execution code for post-run cmds
        pass           # CANT USE 'ABCDEFGHIJKLNOPQRSTUVWXZ'

    # END MODULE-SPECIFIC MAIN MENU DECLARATIONS ############################################################################

    # OVERWRITES METHODS #######################################################################################3
    # GENERIC FUNCTIONS ####################################################################################################################
    def hyperparameter_display_module(self):
        # print hyperparameter settings to screen
        print(f'\nMUTUAL INFORMATION HYPERPARAMETER SETTINGS:')
        sic.SVDConfig(self.standard_config, self.sub_config, self.SUPER_WORKING_NUMPY_LIST[0], self.SUPER_WORKING_NUMPY_LIST[1][0],
            self.rglztn_fctr, self.conv_kill, self.pct_change, self.conv_end_method, self.max_columns).print_parameters()


    '''def row_column_display_select(self):
        INHERITED '''

    '''def filedump_general_ml_setup_module(self):
        INHERITED'''


    def filedump_package_specific_setup_module(self):
        # module for filedump of setup for specific package, used for train, dev, & tests filedump,
        self.wb = sisd.svd_setup_dump(self.wb, self.max_columns)


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
        return sovc.svd_output_vector_calc(self.TRAIN_SWNL[0], self.WINNING_COLUMNS, self.COEFFS, orientation='column')


    def dev_data_calc_algorithm(self):
    # module for passing dev data thru package-specific core algorithm, returns final output vector
    #     return sovc.svd_output_vector_calc(self.DEV_SWNL[0], self.WINNING_COLUMNS, self.COEFFS, orientation='column')
        print(f'\n*** DEV CALC NOT AVAILABLE FOR SVD ***\n')
        return []


    def test_data_calc_algorithm(self):
    # module for passing tests data thru package-specific core algorithm, returns final output vector
        return sovc.svd_output_vector_calc(self.TEST_SWNL[0], self.WINNING_COLUMNS, self.COEFFS, orientation='column')


    def generic_ml_core_error_algorithm(self):
        # module for package-specific core error algorithm, returns total error
        # not in use
        pass


    def train_data_error_algorithm(self):
        # module for passing train data thru package-specific error algorithm, returns total error
        self.TRAIN_OUTPUT_VECTOR = sovc.svd_output_vector_calc(
                        self.TRAIN_SWNL[0], self.WINNING_COLUMNS, self.COEFFS, orientation='column')

        return siec.svd_error_calc(self.TRAIN_OUTPUT_VECTOR, self.TRAIN_SWNL[2])


    def dev_data_error_algorithm(self):
        # module for passing dev data thru package-specific error algorithm, returns total error
        self.DEV_OUTPUT_VECTOR = sovc.svd_output_vector_calc(self.DEV_SWNL[0], self.WINNING_COLUMNS, self.COEFFS,
                                                             orientation='column')

        return siec.svd_error_calc(self.DEV_OUTPUT_VECTOR, self.DEV_SWNL[2])


    def test_data_error_algorithm(self):
        # module for passing tests data thru package-specific error algorithm, returns total error
        self.TEST_OUTPUT_VECTOR = sovc.svd_output_vector_calc(self.TEST_SWNL[0], self.WINNING_COLUMNS, self.COEFFS,
                                                              orientation='column')

        return siec.svd_error_calc(self.TEST_OUTPUT_VECTOR, self.TEST_SWNL[2])


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


    def base_dev_build_module(self):
        print(f'\n *** DEV IS NOT AVAILABLE FOR SVD *** \n')


    def sub_dev_build_module(self):
        # return module with package-specific tests matrix build commands, overwritten in child
        # overwritten in child
        pass


    def sub_dev_build_cmds(self):
        # return list with package-specific tests matrix build commands
        return []  # SPECIFIED IN CHILDREN   CANT USE 'RSDFTUVBNA'


    def sub_dev_build_str(self):
        # return string with package-specific allowed tests matrix build commands
        return ''  # SPECIFIED IN CHILDREN   CANT USE 'RSDFTUVBNA'


    def dev_build(self):
        print(f'\n*** DEV OBJECTS ARE NOT AVAILABLE FOR SVD ***\n')


    # END DEV BUILD DEFINITIONS ##################################################################################################

    # TEST BUILD DEFINITIONS ##################################################################################################
    '''def base_test_build_module(self):
        INHERITED'''


    def sub_test_build_module(self):
        # return module with package-specific tests matrix build commands, overwritten in child
        # overwritten in child
        pass


    def sub_test_build_cmds(self):
        # return list with package-specific tests matrix build commands
        return []  # SPECIFIED IN CHILDREN   CANT USE 'DRSPFTUBONA'


    def sub_test_build_str(self):
        # return string with package-specific allowed tests matrix build commands
        return ''  # SPECIFIED IN CHILDREN   CANT USE 'DRSPFTUBONA'


    '''def test_build(self):
        INHERITED'''


    # END TEST BUILD DEFINITIONS ##################################################################################################

    # TRAIN CALC DEFINITIONS ###################################################################################################
    def core_training_code(self):
        # unique run code for particular ML package
        self.WINNING_COLUMNS, self.COEFFS, self.TRAIN_RESULTS = scrc.SVDCoreRunCode(self.TRAIN_SWNL[0], self.TRAIN_SWNL[1],
            self.TRAIN_SWNL[2], self.TRAIN_SWNL[3], self.max_columns, orientation='column').run()

    # END TRAIN CALC DEFINITIONS ###################################################################################################

    # DEV CALC DEFINITIONS ##################################################################################################

    '''def rglztn_partition_iterator(self, number_of_partitions):
        INHERITED'''


    def base_dev_calc_module(self):
        print(f'\n *** DEV CALC NOT AVAILABLE FOR SVD *** \n')


    def sub_dev_calc_module(self):
        # return module with package-specific commands to run current dev matrix
        # overwritten in child
        pass


    def sub_dev_calc_cmds(self):
        # return list with package - specific dev calc commands
        return []  # SPECIFIED IN CHILDREN   CANT USE 'DSA'


    def sub_dev_calc_str(self):
        # return string with package-specific allowed dev calc commands
        return ''  # SPECIFIED IN CHILDREN   CANT USE 'DSA'

    # END DEV CALC DEFINITIONS ##################################################################################################

    # TEST CALC DEFINITIONS ##################################################################################################
    '''def base_test_calc_module(self):
        INHERITED'''


    def sub_test_calc_module(self):
        # return module with package-specific commands to run current tests matrix
        if self.test_calc_select == 'S':
            # 3-22-22 BEAR FIX
            self.CSUTM_DF = sc.test_cases_calc_standard_configs(self.standard_config, self.TEST_SWNL[0],
                            self.TRAIN_SWNL[0], self.ARRAY_OF_NODES, self.SELECT_LINK_FXN, self.TRAIN_SWNL[1],
                            self.activation_constant)


    def sub_test_calc_cmds(self):
        # return list with package - specific tests calc commands
        return [
            'run special SVD tests from standard configs(s)'
        ]                       # SPECIFIED IN CHILDREN   CANT USE 'NA'


    def sub_test_calc_str(self):
        # return string with package-specific allowed tests calc commands
        return 'S'  # SPECIFIED IN CHILDREN   CANT USE 'NA''


    # END TEST CALC DEFINITIONS ##################################################################################################

    # TRAIN DATA DISPLAY ##############################################################################################################
    def train_summary_statistics_module(self):
        # returns module for printing summary statistics of train data
        siss.SVDSummaryStatisticsPrint(self.TRAIN_OUTPUT_VECTOR, self.TRAIN_SWNL[2], 'None', self.train_data_error_algorithm()).print()


    def print_train_results_module(self):
        # returns module for printing train results for particular ML package
        try:
            COPY_TRAIN_RESULTS = n.array(deepcopy(self.TRAIN_RESULTS)).transpose()
            TRAIN_RESULTS_DF = p.DataFrame(data=COPY_TRAIN_RESULTS[1:], columns=COPY_TRAIN_RESULTS[0])
            p.set_option('colheader_justify', 'center')
            p.set_option('display.max_columns', 8)
            p.set_option('display.width', 0)
            print(TRAIN_RESULTS_DF)
            print()
        except:
            print(f'\n*** TRAIN RESULTS HAVE NOT BEEN CALCULATED YET, OR ERROR PRINTING RESULTS OBJECT')


    def train_filedump_module(self):
        # returns module for filedump of train results for particular ML package
        if self.TRAIN_RESULTS != []:
            try:
                self.filedump_general_ml_setup_module()
                self.wb = sisd.svd_setup_dump(self.wb, self.max_columns)

                self.wb = sitrd.svd_train_results_dump(self.wb, self.TRAIN_RESULTS)
                error = self.train_data_error_algorithm()  # DO THIS OUT HERE SO THAT self.TRAIN_OUTPUT_VECTOR IS CALCULATED B4 siss
                self.wb = siss.SVDSummaryStatisticsDump(self.wb, self.TRAIN_OUTPUT_VECTOR, self.TRAIN_SWNL[2], 'None',
                                                        error, 'TRAIN STATISTICS').dump()
            except:
                print(f'\n *** EXCEPTION TRYING TO DUMP TRAIN RESULTS TO FILE IN SVDRun.train_filedump_module()')
        else:
            print(f'\n*** TRAIN RESULTS HAVE NOT BEEN GENERATED YET ***\n')

    # END TRAIN DATA DISPLAY ##############################################################################################################

    # DEV DATA DISPLAY ##############################A##############################A##############################A##############
    def dev_summary_statistics_module(self):
        # returns module for printing summary statistics of dev data
        print(f'\n *** DEV SUMMARY STATISTICS NOT AVAILABLE FOR SVD *** \n')
        # siss.SVDSummaryStatisticsPrint(self.DEV_OUTPUT_VECTOR, self.DEV_SWNL[2], 'None', self.dev_data_error_algorithm()).print()


    '''def print_dev_results_module(self):
        # returns module for printing dev results to screen for all ML packages
        pass
        INHERITED'''


    def dev_filedump_module(self):
        # returns module for filedump of dev results for all ML packages
        print(f'\n*** DEV FILEDUMP NOT AVAILABLE FOR SVD ***\n')
        # BEAR FIX SVDSummaryStatisticsDump
        # self.wb = gdrd.general_dev_results_dump(self.wb, self.DEV_ERROR, self.RGLZTN_FACTORS)
        # self.wb = siss.SVDSummaryStatisticsDump(self.wb, self.DEV_OUTPUT_VECTOR, self.DEV_SWNL[2], 'None',
        #                                               self.dev_data_error_algorithm(), 'DEV STATISTICS).dump()


    # END DEV DATA DISPLAY ##############################A##############################A##############################A##############
    def test_summary_statistics_module(self):
        # returns module for printing summary statistics of tests data
        error = self.test_data_error_algorithm()  # DO THIS OUT HERE TO ENSURE self.TEST_OUTPUT_VECTOR IS UPDATED B4 siss
        siss.SVDSummaryStatisticsPrint(self.TEST_OUTPUT_VECTOR, self.TEST_SWNL[2], 'None', error).print()


    '''
    def print_test_results_module(self):
        pass
        INHERITED
        '''


    def test_filedump_module(self):
        # package-specific module for saving tests results
        # self.wb = sigtrd.svd_generic_test_results_dump(self.wb, self.CSUTM_DF, self.DISPLAY_COLUMNS, self.display_criteria, self.display_rows)
        self.wb = gterd.general_test_results_dump(self.wb, self.CSUTM_DF, self.DISPLAY_COLUMNS, self.display_select, self.display_rows)

        error = self.test_data_error_algorithm()  # DO THIS OUT HERE TO ENSURE self.TEST_OUTPUT_VECTOR IS UPDATED B4 siss

        self.wb = siss.SVDSummaryStatisticsDump(self.wb, self.TEST_OUTPUT_VECTOR, self.TEST_SWNL[2], 'None',
                                                error, 'TEST STATISTICS').dump()


    # END CALC DATA DISPLAY #############################################################A##############################A##############

    '''def base_return_fxn(self):
        INHERITED'''


    def return_fxn(self):
        # TAKE WINNING_COLUMNS OUT TO ConfigRun TO ENABLE DATA TO BE PERMANTENTLY CHOPPED DOWN TO WINNING_COLUMNS ONLY
        return *self.base_return_fxn(), self.WINNING_COLUMNS, self.TRAIN_RESULTS


    '''def run(self):  # MLRunTemplate loop
        INHERITED'''





if __name__ == '__main__':
    pass




























