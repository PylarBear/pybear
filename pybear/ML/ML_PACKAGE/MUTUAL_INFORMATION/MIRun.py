import sys, inspect
import numpy as np, pandas as pd
from debug import get_module_name as gmn
from data_validation import arg_kwarg_validater as akv
from general_data_ops import get_shape as gs
from ML_PACKAGE import MLRunTemplate as mlrt
from ML_PACKAGE.MUTUAL_INFORMATION import MICoreRunCode as micrc, MIConfig as mic, mi_output_vector_calc as miovc, \
    mi_error_calc as miec
from ML_PACKAGE.MUTUAL_INFORMATION.print_results import MISummaryStatistics as miss, mi_setup_dump as misd, \
    mi_train_results_dump as mitrd
from ML_PACKAGE.GENERIC_PRINT import general_test_results_dump as gterd
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
# END DEV BUILD DEFINITIONS ##################################################################################################

# TEST BUILD DEFINITIONS ##################################################################################################
# base_test_build_module()               # module with code for building test objects for all ML packages
# test_build()                           # module for building test objects, train objects starts as original objects, then test objects are extracted from train objects
# END TEST BUILD DEFINITIONS ##################################################################################################

# TRAIN CALC DEFINITIONS ###################################################################################################
# END TRAIN CALC DEFINITIONS ###################################################################################################

# DEV CALC DEFINITIONS ##################################################################################################
# rglztn_partition_iterator()            # only used to save space in base_dev_calc_module()
# END DEV CALC DEFINITIONS ##################################################################################################

# TEST CALC DEFINITIONS ##################################################################################################
# base_test_calc_module()                # module for performing test calculations for all ML packages
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
#Xmodule_specific_main_menu_operations() # returns module-specific execution code for post-run cmds, overwritten in child
# END MODULE-SPECIFIC MAIN MENU DECLARATIONS ############################################################################

# GENERIC FUNCTIONS ####################################################################################################################
#Xhyperparameter_display_module()        # print hyperparameter settings to screen
#Xfiledump_package_specific_setup_module()  # module for filedump of setup for specific package, used for train, dev, & test filedump, overwritten in child
#Xgeneric_ml_core_calc_algorithm()       # return module for package-specific core output algorithm, returns final output vector, overwritten in child
#Xtrain_data_calc_algorithm()            # module for passing train data thru package-specific core algorithm, returns final output vector
#Xdev_data_calc_algorithm()              # module for passing dev data thru package-specific core algorithm, returns final output vector
#Xtest_data_calc_algorithm()             # module for passing dev data thru package-specific core algorithm, returns final output vector
#Xgeneric_ml_core_error_algorithm()      # return module for package-specific core error algorithm, returns total error, overwritten in child
#Xtrain_data_error_algorithm()           # module for passing train data thru package-specific error algorithm, returns total error
#Xdev_data_error_algorithm()             # module for passing dev data thru package-specific error algorithm, returns total error
#Xtest_data_error_algorithm()            # module for passing test data th
# END GENERIC FUNCTIONS ####################################################################################################################

# TRAIN BUILD DEFINITIONS #####################################################################################################
# END TRAIN BUILD DEFINITIONS ################################################################################################

# DEV BUILD DEFINITIONS ##################################################################################################
#Xbase_dev_build_module()                # module with code for building dev objects for all ML packages
#Xsub_dev_build_module()                 # return module with package-specific dev objects build code, overwritten in child
#Xsub_dev_build_cmds()                   # return list with package-specific dev objects build prompts, overwritten in child
#Xdev_build()                            # module for building dev objects, train objects starts as original objects, then dev objects are extracted from train objects
# END DEV BUILD DEFINITIONS ##################################################################################################

# TEST BUILD DEFINITIONS ##################################################################################################
#Xsub_test_build_module()                # return module with package-specific test objects build code, overwritten in child
#Xsub_test_build_cmds()                  # return list with package-specific test objects build prompts, overwritten in child
# END TEST BUILD DEFINITIONS ##################################################################################################

# TRAIN CALC DEFINITIONS ###################################################################################################
#Xcore_training_code()                   # return unique core training algorithm for particular ML package
# END TRAIN CALC DEFINITIONS ###################################################################################################

# DEV CALC DEFINITIONS ##################################################################################################
#Xbase_dev_calc_module()                 # module for performing dev calculations for all ML packages
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
#Xdev_summary_statistics_module()        # returns module for printing summary statistics of dev data for all ML packages
#Xprint_dev_results_module()             # returns module for printing dev results to screen for all ML packages
#Xdev_filedump_module()                  # returns module for filedump of dev results for all ML packages
# END DEV DATA DISPLAY ###########################################################################################################

# CALC DATA DISPLAY ###########################################################################################################
#Xtest_summary_statistics_module()       # returns module for printing summary statistics of test data for particular ML package
#Xprint_test_results_module()            # returns module for printing test results to screen for particular ML package
#Xtest_filedump_module()                 # returns module for filedump of test results for particular ML package
# END CALC DATA DISPLAY ###########################################################################################################

#Xreturn_fxn()                           # return self.base_return_fxn()

#  UNIQUE ###############################################################################################################
#  NONE AS OF 2-12-22


class MIRun(mlrt.MLRunTemplate):


    def __init__(self, standard_config, mi_config, SUPER_RAW_NUMPY_LIST, RAW_SUPOBJS, SUPER_WORKING_NUMPY_LIST,
                 WORKING_SUPOBJS, data_run_orientation, target_run_orientation, refvecs_run_orientation, WORKING_CONTEXT,
                 WORKING_KEEP, TRAIN_SWNL, DEV_SWNL, TEST_SWNL, split_method, LABEL_RULES, number_of_labels, event_value,
                 negative_value, conv_kill, pct_change, conv_end_method, rglztn_type, rglztn_fctr, bypass_validation,
                 TARGET_TRANSPOSE, TARGET_AS_LIST, mi_batch_method, mi_batch_size, mi_int_or_bin_only, mi_max_columns,
                 mi_bypass_agg, intcpt_col_idx, TRAIN_RESULTS, Y_OCCUR_HOLDER, Y_SUM_HOLDER, Y_FREQ_HOLDER):

        super().__init__(standard_config, mi_config, SUPER_RAW_NUMPY_LIST, RAW_SUPOBJS, SUPER_WORKING_NUMPY_LIST,
                 WORKING_SUPOBJS, data_run_orientation, target_run_orientation, refvecs_run_orientation, WORKING_CONTEXT,
                 WORKING_KEEP, TRAIN_SWNL, DEV_SWNL, TEST_SWNL, split_method, LABEL_RULES, number_of_labels, event_value,
                 negative_value, conv_kill, pct_change, conv_end_method, rglztn_type, rglztn_fctr, bypass_validation,
                 __name__)


        self.this_module = gmn.get_module_name(str(sys.modules[__name__]))
        fxn = '__init__'

        self.bypass_validation = akv.arg_kwarg_validater(bypass_validation, 'bypass_validation', [True, False, None],
                                                         self.this_module, fxn)


        # MI PARAMETERS:
        self.batch_method = mi_batch_method
        self.batch_size = mi_batch_size
        self.int_or_bin_only = mi_int_or_bin_only
        self.max_columns = mi_max_columns
        self.bypass_agg = mi_bypass_agg
        self.intcpt_col_idx = intcpt_col_idx

        self.base_disallowed = 'hw'


        # PLACEHOLDERS
        self.X_TRIAL = []
        self.X_TRIAL_HEADER = []
        self.WINNING_COLUMNS = []
        self.COEFFS = []
        self.SCORES = []

        self.tc_method = 'MI'

        # 4-17-22 CREATE UNIQUE OBJECT FOR MI TO ALLOW FOR DISPLAY/DUMP OF TRAIN RESULTS
        self.TRAIN_RESULTS = TRAIN_RESULTS      # IS EXCHANGABLE BETWEEN MIConfigRun & MIRun

        self.TARGET_TRANSPOSE = TARGET_TRANSPOSE
        self.TARGET_AS_LIST = TARGET_AS_LIST
        self.Y_OCCUR_HOLDER = Y_OCCUR_HOLDER
        self.Y_SUM_HOLDER = Y_SUM_HOLDER
        self.Y_FREQ_HOLDER = Y_FREQ_HOLDER


    #  INHERITS FOR NOW ####################################################################################################
    # MODULE-SPECIFIC MAIN MENU DECLARATIONS ############################################################################
    def module_specific_main_menu_cmds(self):
        # module-specific top-level menu options
        return {}   # CANT USE 'ABCDEFGHIJKLNOPQRSTUVWXZ'
    
    
    def module_specific_main_menu_operations(self):
        # execution code for post-run cmds
        pass           # CANT USE 'ABCDEFGHIJKLNOPQRSTUVWXZ'

    # END MODULE-SPECIFIC MAIN MENU DECLARATIONS ############################################################################

    # OVERWRITES METHODS #######################################################################################3
    # GENERIC FUNCTIONS ####################################################################################################################
    def hyperparameter_display_module(self):
        # print hyperparameter settings to screen
        print(f'\nMUTUAL INFORMATION HYPERPARAMETER SETTINGS:')

        mic.MIConfig(self.standard_config, self.sub_config, self.SUPER_WORKING_NUMPY_LIST, self.WORKING_SUPOBJS,
                     self.data_run_orientation, self.batch_method, self.batch_size, self.int_or_bin_only, self.max_columns,
                     self.bypass_agg, self.intcpt_col_idx, self.bypass_validation).print_parameters()


    '''def row_column_display_select(self):
        INHERITED '''

    '''def filedump_general_ml_setup_module(self):
        INHERITED'''


    def filedump_package_specific_setup_module(self):
        # module for filedump of setup for specific package, used for train, dev, & test filedump,
        self.wb = misd.mi_setup_dump(self.wb, self.batch_method, self.batch_size, self.int_or_bin_only,
                                     self.max_columns, self.intcpt_col_idx)



    '''def dev_or_test_draw_params(self):
        INHERITED'''

    '''def random_dev_or_test_draw(self, TRAIN_SWNL, size):
        INHERITED'''

    '''def partition_dev_or_test_draw(self, TRAIN_SWNL, number_of_partitions, partition_number):
        INHERITED'''

    '''def category_dev_or_test_draw():
        INHERITED'''


    def generic_ml_core_calc_algorithm(self):
        '''return module for package-specific core output algorithm, returns final output vector, overwritten in child'''
        # not in use
        pass


    def train_data_calc_algorithm(self):
        '''module for passing train data thru package-specific core algorithm, returns final output vector'''
        return miovc.mi_output_vector_calc(self.TRAIN_SWNL[0], self.data_run_orientation, self.WINNING_COLUMNS, self.COEFFS)


    def dev_data_calc_algorithm(self):
        '''module for passing dev data thru package-specific core algorithm, returns final output vector'''
        # return miovc.mi_output_vector_calc(self.DEV_SWNL[0], self.data_run_orientation, self.WINNING_COLUMNS, self.COEFFS)
        print(f'\n*** DEV CALC NOT AVAILABLE FOR MI ***\n')
        return []


    def test_data_calc_algorithm(self):
        '''module for passing test data thru package-specific core algorithm, returns final output vector'''
        return miovc.mi_output_vector_calc(self.TEST_SWNL[0], self.data_run_orientation, self.WINNING_COLUMNS, self.COEFFS)


    def generic_ml_core_error_algorithm(self):
        # module for package-specific core error algorithm, returns total error
        # not in use
        pass


    def train_data_error_algorithm(self):
        '''module for passing train data thru package-specific error algorithm, returns total error'''
        self.TRAIN_OUTPUT_VECTOR = miovc.mi_output_vector_calc(self.TRAIN_SWNL[0], self.data_run_orientation,
                                                               self.WINNING_COLUMNS, self.COEFFS)

        return miec.mi_error_calc(self.TRAIN_OUTPUT_VECTOR, self.TRAIN_SWNL[1])


    def dev_data_error_algorithm(self):
        '''module for passing dev data thru package-specific error algorithm, returns total error'''
        self.DEV_OUTPUT_VECTOR = miovc.mi_output_vector_calc(self.DEV_SWNL[0], self.data_run_orientation,
                                                             self.WINNING_COLUMNS, self.COEFFS)

        return miec.mi_error_calc(self.DEV_OUTPUT_VECTOR, self.DEV_SWNL[1])


    def test_data_error_algorithm(self):
        '''module for passing test data thru package-specific error algorithm, returns total error'''
        self.TEST_OUTPUT_VECTOR = miovc.mi_output_vector_calc(self.TEST_SWNL[0], self.data_run_orientation,
                                                              self.WINNING_COLUMNS, self.COEFFS)

        return miec.mi_error_calc(self.TEST_OUTPUT_VECTOR, self.TEST_SWNL[2])


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
        print(f'\n *** DEV IS NOT AVAILABLE FOR MI *** \n')


    def sub_dev_build_module(self):
        # return module with package-specific test matrix build commands, overwritten in child
        # overwritten in child
        pass


    def sub_dev_build_cmds(self):
        # return list with package-specific test matrix build commands
        return {}  # SPECIFIED IN CHILDREN   CANT USE 'RSDFTUVBNA'


    def dev_build(self):
        print(f'\n*** DEV OBJECTS ARE NOT AVAILABLE FOR MI ***\n')


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
        '''unique run code for particular ML package'''

        while True:  # THIS IS HERE JUST TO BYPASS ALL CODE IF MULTI-COLUMN TARGET
            # CANT DO MUTUAL INFORMATION ON MULTI-LABEL TARGET
            if gs.get_shape('self.TRAIN_SWNL', self.TRAIN_SWNL[1], self.target_run_orientation)[1] > 1:
                print(f'\n*** TARGET ENTERING MICoreRunCode HAS MULTIPLE VECTORS, CANNOT DO MUTUAL INFORMATION ***\n')
                break

            self.WINNING_COLUMNS, self.COEFFS, self.TRAIN_RESULTS = micrc.MICoreRunCode(self.TRAIN_SWNL[0],
                self.WORKING_SUPOBJS[0][msod.QUICK_POSN_DICT()["HEADER"]], self.TRAIN_SWNL[1], self.TARGET_TRANSPOSE,
                self.TARGET_AS_LIST, self.data_run_orientation, self.target_run_orientation, self.batch_method,
                self.batch_size, self.max_columns, self.bypass_agg, self.intcpt_col_idx, self.Y_OCCUR_HOLDER,
                self.Y_SUM_HOLDER, self.Y_FREQ_HOLDER, self.bypass_validation).run()

            break

    # END TRAIN CALC DEFINITIONS ###################################################################################################

    # DEV CALC DEFINITIONS ##################################################################################################

    '''def rglztn_partition_iterator(self, number_of_partitions):
        INHERITED'''


    def base_dev_calc_module(self):
        print(f'\n *** DEV CALC NOT AVAILABLE FOR MI *** \n')


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
        if self.test_calc_select == 'S':
            # 3-22-22 BEAR FIX
            self.CSUTM_DF = sc.test_cases_calc_standard_configs(self.standard_config, self.TEST_SWNL[0],
                            self.TRAIN_SWNL[0], self.ARRAY_OF_NODES, self.SELECT_LINK_FXN, self.TRAIN_SWNL[1],
                            self.activation_constant)


    def sub_test_calc_cmds(self):
        # return list with package - specific test calc commands
        return {
            's': 'run special MI test from standard configs'
        }                       # SPECIFIED IN CHILDREN   CANT USE 'NA'


    # END TEST CALC DEFINITIONS ##################################################################################################

    # TRAIN DATA DISPLAY ##############################################################################################################
    def train_summary_statistics_module(self):
        # returns module for printing summary statistics of train data
        miss.MISummaryStatisticsPrint(self.TRAIN_OUTPUT_VECTOR, self.TRAIN_SWNL[1], 'None', self.train_data_error_algorithm()).print()


    def print_train_results_module(self):
        # returns module for printing train results for particular ML package
        try:
            pd.set_option('colheader_justify', 'center')
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', 140)
            print(self.TRAIN_RESULTS)
            print()
        except:
            print(f'\n*** TRAIN RESULTS HAVE NOT BEEN CALCULATED YET, OR ERROR PRINTING RESULTS OBJECT')


    def train_filedump_module(self):
        # returns module for filedump of train results for particular ML package

        fxn = inspect.stack()[0][3]

        if not self.TRAIN_RESULTS.equals(pd.DataFrame({})):
            try: self.filedump_general_ml_setup_module()
            except: print(f'\n *** {self.this_module}.{fxn}() >>> EXCEPTION IN filedump_general_ml_setup_module() ***')

            try: self.wb = misd.mi_setup_dump(self.wb, self.batch_method, self.batch_size, self.int_or_bin_only,
                                             self.max_columns, self.intcpt_col_idx)
            except: print(f'\n *** {self.this_module}.{fxn}() >>> EXCEPTION IN mi_setup_dump() ***')

            self.wb = mitrd.mi_train_results_dump(self.wb, self.TRAIN_RESULTS)

            # try: self.wb = mitrd.mi_train_results_dump(self.wb, self.TRAIN_RESULTS)
            # except: print(f'\n *** {self.this_module}.{fxn}() >>> EXCEPTION IN mi_train_results_dump() ***')

            try: error = self.train_data_error_algorithm()  # DO THIS OUT HERE SO THAT self.TRAIN_OUTPUT_VECTOR IS CALCULATED B4 miss
            except:
                error = 'Nan'
                print(f'\n *** {self.this_module}.{fxn}() >>> EXCEPTION IN train_data_error_algorithm() ***')

            try: self.wb = miss.MISummaryStatisticsDump(self.wb, self.TRAIN_OUTPUT_VECTOR, self.TRAIN_SWNL[1], 'None',
                                                        error, 'TRAIN STATISTICS').dump()
            except: print(f'\n *** {self.this_module}.{fxn}() >>> EXCEPTION IN MISummaryStatisticsDump() ***')

        else:
            print(f'\n*** TRAIN RESULTS HAVE NOT BEEN GENERATED YET ***\n')

    # END TRAIN DATA DISPLAY ##############################################################################################################

    # DEV DATA DISPLAY ##############################A##############################A##############################A##############
    def dev_summary_statistics_module(self):
        # returns module for printing summary statistics of dev data
        print(f'\n *** DEV SUMMARY STATISTICS NOT AVAILABLE FOR MI *** \n')
        # miss.MISummaryStatisticsPrint(self.DEV_OUTPUT_VECTOR, self.DEV_SWNL[1], 'None', self.dev_data_error_algorithm()).print()


    '''def print_dev_results_module(self):
        # returns module for printing dev results to screen for all ML packages
        pass
        INHERITED'''


    def dev_filedump_module(self):
        # returns module for filedump of dev results for all ML packages
        print(f'\n*** DEV FILEDUMP NOT AVAILABLE FOR MI ***\n')
        # self.wb = gdrd.general_dev_results_dump(self.wb, self.DEV_ERROR, self.RGLZTN_FACTORS)
        # self.wb = miss.MISummaryStatisticsDump(self.wb, self.DEV_OUTPUT_VECTOR, self.DEV_SWNL[1], 'None',
        #                                               self.dev_data_error_algorithm(), 'DEV STATISTICS).dump()


    # END DEV DATA DISPLAY ##############################A##############################A##############################A##############
    def test_summary_statistics_module(self):
        # returns module for printing summary statistics of test data
        error = self.test_data_error_algorithm()  # DO THIS OUT HERE TO ENSURE self.TEST_OUTPUT_VECTOR IS UPDATED B4 miss
        miss.MISummaryStatisticsPrint(self.TEST_OUTPUT_VECTOR, self.TEST_SWNL[2], 'None', error).print()


    '''
    def print_test_results_module(self):
        pass
        INHERITED
        '''


    def test_filedump_module(self):
        # package-specific module for saving test results
        # self.wb = migtrd.mi_generic_test_results_dump(self.wb, self.CSUTM_DF, self.DISPLAY_COLUMNS, self.display_criteria, self.display_rows)
        self.wb = gterd.general_test_results_dump(self.wb, self.CSUTM_DF, self.DISPLAY_COLUMNS, self.display_select, self.display_rows)

        error = self.test_data_error_algorithm()  # DO THIS OUT HERE TO ENSURE self.TEST_OUTPUT_VECTOR IS UPDATED B4 miss

        self.wb = miss.MISummaryStatisticsDump(self.wb, self.TEST_OUTPUT_VECTOR, self.TEST_SWNL[2], 'None',
                                                error, 'TEST STATISTICS').dump()


    # END CALC DATA DISPLAY #############################################################A##############################A##############

    '''def base_return_fxn(self):
        INHERITED'''


    def return_fxn(self):
        # TAKE WINNING_COLUMNS OUT TO ConfigRun TO ENABLE DATA TO BE PERMANENTLY CHOPPED DOWN TO WINNING_COLUMNS ONLY
        return *self.base_return_fxn(), self.WINNING_COLUMNS, self.TRAIN_RESULTS


    '''def run(self):  # MLRunTemplate loop
        INHERITED'''































if __name__ == '__main__':
    from MLObjects.TestObjectCreators.SXNL import CreateSXNL as csxnl
    from MLObjects.ObjectOrienter import MLObjectOrienter as mloo

    bypass_validation = False

    DATA = pd.read_csv(r'C:\Users\Bill\Documents\WORK STUFF\RESUME\1 - OTHER\SRP\beer_reviews.csv',
                      nrows=10000,
                      header=0).dropna(axis=0)

    DATA = DATA[DATA.keys()[[3, 4, 5, 8, 9, 11]]]

    TARGET = DATA['review_overall']
    TARGET_HEADER = [['review_overall']]
    TARGET = TARGET.to_numpy().reshape((1,-1))

    DATA = DATA.drop(columns=['review_overall'])

    RAW_DATA = DATA.copy()
    RAW_DATA_HEADER = np.fromiter(RAW_DATA.keys(), dtype='<U50').reshape((1,-1))
    RAW_DATA = RAW_DATA.to_numpy()

    # GIVEN TO MIRun, NOT CreateSXNL!
    data_given_format = 'ARRAY'
    data_given_orient = 'COLUMN'
    target_given_format = 'ARRAY'
    target_given_orient = 'COLUMN'
    refvecs_given_format = 'ARRAY'
    refvecs_given_orient = 'COLUMN'

    SXNLClass = csxnl.CreateSXNL(rows=None,
                                 bypass_validation=False,
                                 data_return_format=data_given_format,
                                 data_return_orientation=data_given_orient,
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
                                 target_return_format=target_given_format,
                                 target_return_orientation=target_given_orient,
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
                                 refvecs_return_format=refvecs_given_format,
                                 refvecs_return_orientation=target_given_orient,
                                 REFVECS_OBJECT=np.fromiter(range(len(RAW_DATA)), dtype=int).reshape((1,-1)),
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

    del SXNLClass

    TargetOrienter = mloo.MLObjectOrienter(
                                            target_is_multiclass=False,
                                            TARGET=SWNL[1],
                                            target_given_orientation=target_given_orient,
                                            target_return_orientation=target_given_orient,
                                            target_return_format='AS_GIVEN',

                                            TARGET_TRANSPOSE=None,
                                            target_transpose_given_orientation=None,
                                            target_transpose_return_orientation=target_given_orient,
                                            target_transpose_return_format='AS_GIVEN',

                                            TARGET_AS_LIST=None,
                                            target_as_list_given_orientation=None,
                                            target_as_list_return_orientation=target_given_orient,

                                            RETURN_OBJECTS=['TARGET', 'TARGET_TRANSPOSE', 'TARGET_AS_LIST'],

                                            bypass_validation=bypass_validation,
                                            calling_module='MIRun',
                                            calling_fxn='test_MIRun'
        )


    data_run_orientation = SXNLClass.data_current_orientation
    target_run_orientation = SXNLClass.target_current_orientation
    refvecs_run_orientation = SXNLClass.refvecs_current_orientation

    SWNL[1] = TargetOrienter.TARGET
    TARGET_TRANSPOSE = TargetOrienter.TARGET_TRANSPOSE
    TARGET_AS_LIST = TargetOrienter.TARGET_AS_LIST


    WORKING_CONTEXT = []
    WORKING_KEEP = SRNL[1][0]

    TRAIN_SWNL = []
    DEV_SWNL = []
    TEST_SWNL = []

    standard_config = 'None'
    mi_config = 'None'

    split_method = 'None'
    LABEL_RULES = []
    number_of_labels = 1
    event_value = ''
    negative_value = ''
    mi_batch_method = 'B'
    mi_batch_size = ...
    mi_int_or_bin_only = 'Y'
    mi_max_columns = 20
    mi_bypass_agg = False
    TRAIN_RESULTS = pd.DataFrame({})
    intcpt_col_idx = None
    Y_OCCUR_HOLDER = None
    Y_SUM_HOLDER = None
    Y_FREQ_HOLDER = None


    TRAIN_SWNL, DEV_SWNL, TEST_SWNL, WINNING_COLUMNS, TRAIN_RESULTS = \
        MIRun(standard_config, mi_config, SRNL, RAW_SUPOBJS, SWNL, WORKING_SUPOBJS, data_run_orientation,
            target_run_orientation, refvecs_run_orientation, WORKING_CONTEXT, WORKING_KEEP, TRAIN_SWNL, DEV_SWNL,
            TEST_SWNL, split_method, LABEL_RULES, number_of_labels, event_value, negative_value, 'dum_conv_kill',
            'dum_pct_change', 'dum_conv_end_method', 'dum_rglztn_type', 'dum_rglztn_fctr', bypass_validation,
            TARGET_TRANSPOSE, TARGET_AS_LIST, mi_batch_method, mi_batch_size, mi_int_or_bin_only, mi_max_columns,
            mi_bypass_agg, intcpt_col_idx, TRAIN_RESULTS, Y_OCCUR_HOLDER, Y_SUM_HOLDER, Y_FREQ_HOLDER).run()





































