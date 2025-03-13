import sys, inspect
import pandas as pd, numpy as np
from copy import deepcopy
from general_data_ops import get_shape as gs
from data_validation import validate_user_input as vui
from ML_PACKAGE import MLRunTemplate as mlrt
from ML_PACKAGE.GREEDY_MULTIPLE_LINEAR_REGRESSION import GMLRCoreRunCode as gcrc, GMLRConfig as gc
from ML_PACKAGE.MLREGRESSION import mlr_output_vector_calc as movc, mlr_error_calc as mec
from ML_PACKAGE.GREEDY_MULTIPLE_LINEAR_REGRESSION.print_results import GMLRSummaryStatistics as gss, gmlr_setup_dump as gsd, \
    gmlr_train_results_dump as gtrd
from ML_PACKAGE.GENERIC_PRINT import general_test_results_dump as gterd, general_dev_results_dump as gdrd
from MLObjects import MLObject as mlo
from MLObjects.SupportObjects import master_support_object_dict as msod


# INHERITED ####################################################################################################################
# MODULE-SPECIFIC MAIN MENU DECLARATIONS ############################################################################
# END MODULE-SPECIFIC MAIN MENU DECLARATIONS ############################################################################

# GENERIC FUNCTIONS ####################################################################################################################
# row_column_display_select()            # user selects rows and columns for display on screen and filedump
# filedump_path()                        # hard-coded directory, user-entered filename
# filedump_general_ml_setup_module()     # module for filedump of general setup for all ML packages, used for train, dev, & tests filedump
# dev_or_test_draw_params()              # module that returns the sample size to pull from train data for dev or tests data sets
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
#Xmodule_specific_main_menu_operations() # returns module-specific execution code for post-run cmds, overwritten in child
# END MODULE-SPECIFIC MAIN MENU DECLARATIONS ############################################################################

# GENERIC FUNCTIONS ####################################################################################################################
#Xhyperparameter_display_module()        # print hyperparameter settings to screen
#Xfiledump_package_specific_setup_module()  # module for filedump of setup for specific package, used for train, dev, & tests filedump, overwritten in child
# random_dev_or_test_draw()              # randomly select user-specified quantity of examples for dev & tests sets from (remaining) examples in TRAIN_SWNL
# partition_dev_or_test_draw()           # partition (remaining) examples in TRAIN_SWNL for selection into dev or tests
# category_dev_or_test_draw()            # select examples for dev & tests sets using categories in TRAIN_SWNL
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
#Xdev_build()                            # module for building dev objects, train objects starts as original objects, then dev objects are extracted from train objects
# END DEV BUILD DEFINITIONS ##################################################################################################

# TEST BUILD DEFINITIONS ##################################################################################################
#Xsub_test_build_module()                # return module with package-specific tests objects build code, overwritten in child
#Xsub_test_build_cmds()                  # return list with package-specific tests objects build prompts, overwritten in child
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
#Xsub_test_calc_module()                 # return module with package-specific commands for performing tests calculations
#Xsub_test_calc_cmds()                   # return list with package-specific tests calc prompts
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
#  NONE AS OF 6/16/23









class GMLRRun(mlrt.MLRunTemplate):

    def __init__(self, standard_config, gmlr_config, SUPER_RAW_NUMPY_LIST, RAW_SUPOBJS, SUPER_WORKING_NUMPY_LIST,
                 WORKING_SUPOBJS, data_run_orientation, target_run_orientation, refvecs_run_orientation, WORKING_CONTEXT,
                 WORKING_KEEP, TRAIN_SWNL, DEV_SWNL, TEST_SWNL, split_method, LABEL_RULES, number_of_labels, event_value,
                 negative_value, gmlr_conv_kill, gmlr_pct_change, gmlr_conv_end_method, gmlr_rglztn_type, gmlr_rglztn_fctr,
                 bypass_validation, TARGET_TRANSPOSE, TARGET_AS_LIST, gmlr_batch_method, gmlr_batch_size, gmlr_type,
                 gmlr_score_method, gmlr_float_only, gmlr_max_columns, gmlr_bypass_agg, intcpt_col_idx, TRAIN_RESULTS):

        super().__init__(standard_config, gmlr_config, SUPER_RAW_NUMPY_LIST, RAW_SUPOBJS, SUPER_WORKING_NUMPY_LIST,
                WORKING_SUPOBJS, data_run_orientation, target_run_orientation, refvecs_run_orientation, WORKING_CONTEXT,
                WORKING_KEEP, TRAIN_SWNL, DEV_SWNL, TEST_SWNL, split_method, LABEL_RULES, number_of_labels, event_value,
                negative_value, gmlr_conv_kill, gmlr_pct_change, gmlr_conv_end_method, gmlr_rglztn_type, gmlr_rglztn_fctr,
                bypass_validation, __name__)

        self.SUPER_WORKING_NUMPY_LIST = SUPER_WORKING_NUMPY_LIST
        self.TARGET_TRANSPOSE = TARGET_TRANSPOSE
        self.TARGET_AS_LIST = TARGET_AS_LIST

        # GMLR PARAMETERS:
        self.batch_method = gmlr_batch_method
        self.batch_size = gmlr_batch_size
        self.gmlr_type = gmlr_type
        self.score_method = gmlr_score_method
        self.intcpt_col_idx = intcpt_col_idx
        self.float_only = gmlr_float_only
        self.max_columns = gmlr_max_columns
        self.bypass_agg = gmlr_bypass_agg

        # PLACEHOLDERS
        self.X_TRIAL = []
        self.X_TRIAL_HEADER = []
        self.WINNING_COLUMNS = []
        self.COEFFS = []
        self.CUMUL_SCORE = []

        self.tc_method = 'GMLR'

        # 2-23-22 CREATE UNIQUE OBJECT FOR GMLR TO ALLOW FOR DISPLAY/DUMP OF TRAIN RESULTS
        self.TRAIN_RESULTS = TRAIN_RESULTS      # IS EXCHANGABLE BETWEEN GMLRConfigRun & GMLRRun


    #  INHERITS FOR NOW ####################################################################################################
    # MODULE-SPECIFIC MAIN MENU DECLARATIONS ############################################################################
    def module_specific_main_menu_cmds(self):
        # module-specific top-level menu options
        # CANT USE 'ABCDEFGHIJKLNOPQRSTUVWXZ'
        return {'0': 'set regularization factor'}
    
    
    def module_specific_main_menu_operations(self):
        # execution code for post-run cmds      # CANT USE 'ABCDEFGHIJKLNOPQRSTUVWXZ'
        if self.post_run_select == '0':
            self.rglztn_fctr = vui.validate_user_float(f'Enter regularization factor (currently {self.rglztn_fctr}) > ', min=0)

    # END MODULE-SPECIFIC MAIN MENU DECLARATIONS ############################################################################

    # OVERWRITES METHODS #######################################################################################3
    # GENERIC FUNCTIONS ####################################################################################################################
    def hyperparameter_display_module(self):
        # print hyperparameter settings to screen
        print(f'\nGREEDY MULTIPLE LINEAR REGRESSION HYPERPARAMETER SETTINGS:')
        gc.GMLRConfig(self.standard_config, self.sub_config, self.SUPER_WORKING_NUMPY_LIST, self.WORKING_SUPOBJS,
            self.data_run_orientation, self.conv_kill, self.pct_change, self.conv_end_method, self.rglztn_type,
            self.rglztn_fctr, self.batch_method, self.batch_size, self.gmlr_type, self.score_method, self.float_only,
            self.max_columns, self.intcpt_col_idx, self.bypass_agg, self.bypass_validation).print_parameters()


    '''def row_column_display_select(self):
        INHERITED '''

    '''def filedump_general_ml_setup_module(self):
        INHERITED'''


    def filedump_package_specific_setup_module(self):
        # module for filedump of setup for specific package, used for train, dev, & tests filedump,
        self.wb = gsd.gmlr_setup_dump(self.wb, self.conv_kill, self.pct_change, self.conv_end_method, self.rglztn_type,
              self.rglztn_fctr, self.batch_method, self.batch_size, {'L': 'Lazy', 'F': 'Forward', 'B': 'Backward'}[self.gmlr_type],
              {'F': 'F-score', 'Q': 'RSQ', 'A': 'Adj RSQ', 'R':'r'}[self.score_method], self.float_only, self.max_columns,
              self.intcpt_col_idx)


    '''def dev_or_test_draw_params(self):
        INHERITED'''


    def random_dev_or_test_draw(self, size=None):

        fxn = inspect.stack()[0][3]

        # CREATE A BACKUP OF self.TRAIN_SWNL[0] (DATA) BECAUSE super().random_dev_or_test_draw(size) MODIFIES self.TRAIN_SWNL
        # IN-PLACE, SO THAT IF XTX_INV FAILS, THE BACKUP CAN BE RESTORE self.TRAIN_SWNL TO ORIGINAL UNCHOPPED STATE
        TRAIN_SWNL_BACKUP = [deepcopy(_) if isinstance(_, dict) else _.copy() for _ in self.TRAIN_SWNL]

        ctr = 0
        while True:
            ctr += 1

            print(f'\nTrying to get invertible train split. Try #{ctr}...')

            RETURN_SWNL = super().random_dev_or_test_draw(size)

            ObjectClass = mlo.MLObject(
                                        self.TRAIN_SWNL[0],
                                        self.data_run_orientation,
                                        name='TRAIN DATA',
                                        return_orientation='AS_GIVEN',
                                        return_format='AS_GIVEN',
                                        bypass_validation=self.bypass_validation,
                                        calling_module=self.this_module,
                                        calling_fxn=fxn
            )

            _XTX = ObjectClass.return_XTX(return_format='ARRAY')

            try:
                np.linalg.inv(_XTX + self.rglztn_fctr * np.identity(len(_XTX)))
                print(f'Success.')
                break  # IF MAKE IT THRU, TRAIN DATA IS INVERTIBLE AND self.TRAIN_SWNL[0] AND RETURN_SWNL CAN STAY AS IS
            except np.linalg.LinAlgError:
                print(f'Fail.')
                if ctr % 20 != 0:
                    self.TRAIN_SWNL = [deepcopy(_) if isinstance(_, dict) else _.copy() for _ in TRAIN_SWNL_BACKUP]
                    continue
                else:
                    print(f'\n*** random_dev_or_test_draw() HAS FAILED {ctr} TIMES TRYING TO SPLIT TRAIN INTO AN INVERTIBLE STATE *** ')
                    print(f'*** SHAPE OF TRAIN DATA IS {gs.get_shape("TRAIN DATA", self.TRAIN_SWNL[0], self.data_run_orientation)} *** ')
                    print(f'*** CURRENT RIDGE IS {self.rglztn_fctr} ***')
                    __ = vui.validate_user_str(f'\nChange ridge(c), Keep trying(k), proceed anyway(p), abort(a), quit(q)? > ', 'CKPAQ')

                    if __ in ['C', 'K']:
                        self.TRAIN_SWNL = [deepcopy(_) if isinstance(_, dict) else _.copy() for _ in TRAIN_SWNL_BACKUP]

                    if __ == 'C':
                        while True:
                            _rglzn_holder = vui.validate_user_float(f'\nEnter new regularization factor > ', min=0)
                            if vui.validate_user_str(f'User entered {_rglzn_holder}, accept? (y/n) > ', 'YN') == 'Y':
                                self.rglztn_fctr = _rglzn_holder; del __, _rglzn_holder; break
                        continue
                    elif __ == 'K': ctr = 0; del __; continue
                    elif __ == 'P': break
                    elif __ == 'A': del __, TRAIN_SWNL_BACKUP, RETURN_SWNL, ObjectClass, _XTX; raise TimeoutError
                    elif __ == 'Q': del __; print(f'*** USER TERMINATED ***'); quit()
            except:
                print(f'EXCEPTION FOR ERROR OTHER THAN numpy.linalg.LinAlgError')
                raise

        del TRAIN_SWNL_BACKUP, ObjectClass, _XTX

        return RETURN_SWNL


    def partition_dev_or_test_draw(self, number_of_partitions, partition_number):
        # IF TRAIN DOESNT INVERT AFTER THIS, CANT DO ANYTHING ABOUT IT, CANT DO THIS DIFFERENTLY SO HAVE TO FIND ANOTHER WAY,
        # AT LEAST LET USER KNOW IT DIDNT INVERT

        fxn = inspect.stack()[0][3]

        # self.TRAIN_SWNL IS IMPLICITY MODIFIED
        RETURN_SWNL = super().partition_dev_or_test_draw(number_of_partitions, partition_number)

        ObjectClass = mlo.MLObject(self.TRAIN_SWNL[0],
                                   self.data_run_orientation,
                                   name='TRAIN DATA', return_orientation='AS_GIVEN', return_format='AS_GIVEN',
                                   bypass_validation=self.bypass_validation, calling_module=self.this_module, calling_fxn=fxn
                                   )

        _XTX = ObjectClass.return_XTX(return_format='ARRAY')

        del ObjectClass

        while True:
            print(f'\nTesting for invertible train split...')
            try:
                np.linalg.inv(_XTX + self.rglztn_fctr * np.identity(len(_XTX)))
                print(f'Success.')
                break

            except np.linalg.LinAlgError:
                print(f'\n*** XTX FOR TRAIN DATA SPLIT FAILED TO INVERT WITH RIDGE SET TO {self.rglztn_fctr} ***')
                print(f'*** SHAPE OF TRAIN DATA IS {gs.get_shape("DATA", self.TRAIN_SWNL[0], self.data_run_orientation)} ROWS/COLUMNS ***')

                __ =  vui.validate_user_str(f'\nChange lambda(c), proceed anyway(p), abort(a), quit(q)? > ', 'CAQ')

                if __ == 'C':
                    while True:
                        _rgzn_holder = vui.validate_user_float(f'\nEnter new lambda (currently {self.rglztn_fctr}) > ', min=0)
                        if vui.validate_user_str(f'User entered {_rgzn_holder}, accept? (y/n) > ', 'YN') == 'Y':
                            self.rglztn_fctr = _rgzn_holder; del _rgzn_holder; break
                    continue
                elif __ == 'P': break
                elif __ == 'A': del RETURN_SWNL, _XTX; raise TimeoutError
                elif __ == 'Q': print(f'*** USER TERMINATED ***'); quit()

            except:
                print(f'EXCEPTION FOR ERROR OTHER THAN numpy.linalg.LinAlgError')
                raise

        del _XTX

        return RETURN_SWNL


    def category_dev_or_test_draw(self):
        # IF TRAIN DOESNT INVERT AFTER THIS, CAN TRY TO DO THIS ON DIFFERENT CATEGORIES OTHERWISE HAVE TO FIND ANOTHER WAY
        # AT LEAST LET USER KNOW IT DIDNT INVERT
        fxn = inspect.stack()[0][3]

        RETURN_SWNL = super().category_dev_or_test_draw()

        print(f'\nTesting for invertible train split.')

        ObjectClass = mlo.MLObject(self.TRAIN_SWNL[0], self.data_run_orientation, name='TRAIN DATA',
                                   return_orientation='AS_GIVEN', return_format='AS_GIVEN',
                                   bypass_validation=self.bypass_validation,
                                   calling_module=self.this_module, calling_fxn=fxn
                                   )

        _XTX = ObjectClass.return_XTX

        del ObjectClass

        while True:
            print(f'\nTesting for invertible train split...')
            try:
                np.linalg.inv(_XTX + self.rglztn_fctr * np.identity(len(_XTX)))
                print(f'Success.')
                break

            except np.linalg.LinAlgError:
                print(f'\n*** XTX FOR TRAIN DATA SPLIT FAILED TO INVERT WITH RIDGE SET TO {self.rglztn_fctr} ***')
                print(f'*** SHAPE OF TRAIN DATA IS {gs.get_shape("DATA", self.TRAIN_SWNL[0], self.data_run_orientation)} ROWS/COLUMNS ***')

                __ = vui.validate_user_str(f'\nChange lambda(c), proceed anyway(p), abort(a), quit(q)? > ', 'CAQ')

                if __ == 'C':
                    while True:
                        _rgzn_holder = vui.validate_user_float(f'\nEnter new lambda (currently {self.rglztn_fctr}) > ', min=0)
                        if vui.validate_user_str(f'User entered {_rgzn_holder}, accept? (y/n) > ', 'YN') == 'Y':
                            self.rglztn_fctr = _rgzn_holder; del _rgzn_holder; break
                    continue
                elif __ == 'P': break
                elif __ == 'A': del RETURN_SWNL, _XTX; raise TimeoutError
                elif __ == 'Q': print(f'*** USER TERMINATED ***'); quit()

            except:
                print(f'EXCEPTION FOR ERROR OTHER THAN numpy.linalg.LinAlgError')
                raise

        del _XTX

        return RETURN_SWNL


    def generic_ml_core_calc_algorithm(self):
        # return module for package-specific core output algorithm, returns final output vector, overwritten in child
        # not in use
        pass


    def train_data_calc_algorithm(self):
        # module for passing train data thru package-specific core algorithm, returns final output vector

        # MUST SIZE TRAIN DATA TO CORRECT COLUMNS VIA self.WINNING_COLUMNS
        TrainDataClass = mlo.MLObject(
                                      self.TRAIN_SWNL[0],
                                      self.data_run_orientation,
                                      name='TRAIN_DATA',
                                      return_orientation='AS_GIVEN',
                                      return_format='AS_GIVEN',
                                      bypass_validation=self.bypass_validation,
                                      calling_module=self.this_module,
                                      calling_fxn=inspect.stack()[0][3]
        )

        return movc.mlr_output_vector_calc(TrainDataClass.return_columns(self.WINNING_COLUMNS, return_orientation='AS_GIVEN',
                                                                         return_format='AS_GIVEN'),
                                           self.data_run_orientation,
                                           self.COEFFS)


    def dev_data_calc_algorithm(self):
        # module for passing dev data thru package-specific core algorithm, returns final output vector

        # MUST SIZE DEV DATA TO CORRECT COLUMNS VIA self.WINNING_COLUMNS
        DevDataClass = mlo.MLObject(
                                    self.DEV_SWNL[0],
                                    self.data_run_orientation,
                                    name='DEV_DATA',
                                    return_orientation='AS_GIVEN',
                                    return_format='AS_GIVEN',
                                    bypass_validation=self.bypass_validation,
                                    calling_module=self.this_module,
                                    calling_fxn=inspect.stack()[0][3]
        )

        return movc.mlr_output_vector_calc(DevDataClass.return_columns(self.WINNING_COLUMNS, return_orientation='AS_GIVEN',
                                                                        return_format='AS_GIVEN'),
                                            self.data_run_orientation,
                                            self.COEFFS)


    def test_data_calc_algorithm(self):
        # module for passing tests data thru package-specific core algorithm, returns final output vector

        # MUST SIZE TEST DATA TO CORRECT COLUMNS VIA self.WINNING_COLUMNS
        TestDataClass = mlo.MLObject(
                                     self.TEST_SWNL[0],
                                     self.data_run_orientation,
                                     name='TEST_DATA',
                                     return_orientation='AS_GIVEN',
                                     return_format='AS_GIVEN',
                                     bypass_validation=self.bypass_validation,
                                     calling_module=self.this_module,
                                     calling_fxn=inspect.stack()[0][3]
        )

        return movc.mlr_output_vector_calc(TestDataClass.return_columns(self.WINNING_COLUMNS, return_orientation='AS_GIVEN',
                                                                        return_format='AS_GIVEN'),
                                            self.data_run_orientation,
                                            self.COEFFS)


    def generic_ml_core_error_algorithm(self):
        # module for package-specific core error algorithm, returns total error
        # not in use
        pass


    def train_data_error_algorithm(self):
        # module for passing train data thru package-specific error algorithm, returns total error

        # MUST SIZE TRAIN DATA TO CORRECT COLUMNS VIA self.WINNING_COLUMNS
        TrainDataClass = mlo.MLObject(
                                      self.TRAIN_SWNL[0],
                                      self.data_run_orientation,
                                      name='TRAIN_DATA',
                                      return_orientation='AS_GIVEN',
                                      return_format='AS_GIVEN',
                                      bypass_validation=self.bypass_validation,
                                      calling_module=self.this_module,
                                      calling_fxn=inspect.stack()[0][3]
        )

        self.TRAIN_OUTPUT_VECTOR =  movc.mlr_output_vector_calc(TrainDataClass.return_columns(self.WINNING_COLUMNS,
                                                                return_orientation='AS_GIVEN', return_format='AS_GIVEN'),
                                                                self.data_run_orientation,
                                                                self.COEFFS)

        return mec.mlr_error_calc(self.TRAIN_OUTPUT_VECTOR, self.TRAIN_SWNL[1])


    def dev_data_error_algorithm(self):
        # module for passing dev data thru package-specific error algorithm, returns total error

        # MUST SIZE DEV DATA TO CORRECT COLUMNS VIA self.WINNING_COLUMNS
        DevDataClass = mlo.MLObject(
                                    self.DEV_SWNL[0],
                                    self.data_run_orientation,
                                    name='DEV_DATA',
                                    return_orientation='AS_GIVEN',
                                    return_format='AS_GIVEN',
                                    bypass_validation=self.bypass_validation,
                                    calling_module=self.this_module,
                                    calling_fxn=inspect.stack()[0][3]
        )


        self.DEV_OUTPUT_VECTOR =  movc.mlr_output_vector_calc(DevDataClass.return_columns(self.WINNING_COLUMNS,
                                                              return_orientation='AS_GIVEN', return_format='AS_GIVEN'),
                                                              self.data_run_orientation,
                                                              self.COEFFS)

        return mec.mlr_error_calc(self.DEV_OUTPUT_VECTOR, self.DEV_SWNL[1])


    def test_data_error_algorithm(self):
        # module for passing tests data thru package-specific error algorithm, returns total error

        # MUST SIZE DEV DATA TO CORRECT COLUMNS VIA self.WINNING_COLUMNS

        TestDataClass = mlo.MLObject(
                                    self.TEST_SWNL[0],
                                    self.data_run_orientation,
                                    name='TEST_DATA',
                                    return_orientation='AS_GIVEN',
                                    return_format='AS_GIVEN',
                                    bypass_validation=self.bypass_validation,
                                    calling_module=self.this_module,
                                    calling_fxn=inspect.stack()[0][3]
        )

        self.TEST_OUTPUT_VECTOR = movc.mlr_output_vector_calc(TestDataClass.return_columns(self.WINNING_COLUMNS,
                                                                                         return_orientation='AS_GIVEN',
                                                                                         return_format='AS_GIVEN'),
                                                             self.data_run_orientation,
                                                             self.COEFFS)

        return mec.mlr_error_calc(self.TEST_OUTPUT_VECTOR, self.TEST_SWNL[1])


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
        # return module with package-specific tests matrix build commands, overwritten in child
        # overwritten in child
        pass


    def sub_dev_build_cmds(self):
        # return list with package-specific tests matrix build commands
        return {}  # SPECIFIED IN CHILDREN   CANT USE 'RSDFTUVBNA'


    '''def dev_build(self):
        INHERITED'''


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
        return {}  # SPECIFIED IN CHILDREN   CANT USE 'DRSPFTUBONA'


    '''def test_build(self):
        INHERITED'''


    # END TEST BUILD DEFINITIONS ##################################################################################################

    # TRAIN CALC DEFINITIONS ###################################################################################################
    def core_training_code(self):
        # unique run code for particular ML package

        while True:  # THIS IS HERE JUST TO BYPASS ALL CODE IF MULTI-COLUMN TARGET
            if gs.get_shape('TARGET', self.TRAIN_SWNL[1], self.target_run_orientation)[1] > 1:
                print(f'\n*** TARGET ENTERING GMLRCoreRunCode HAS MULTIPLE VECTORS, CANNOT DO MULTIPLE LINEAR REGRESSION. ***\n')
                break

            if len(self.DEV_SWNL) == 0 and len(self.TEST_SWNL) == 0:
                self.WINNING_COLUMNS, self.COEFFS, self.TRAIN_RESULTS = \
                    gcrc.GMLRCoreRunCode(self.TRAIN_SWNL[0], self.WORKING_SUPOBJS[0][msod.QUICK_POSN_DICT()['HEADER']].reshape((1,-1)),
                        self.TRAIN_SWNL[1], self.TARGET_TRANSPOSE, self.TARGET_AS_LIST, self.data_run_orientation,
                        self.target_run_orientation, self.conv_kill, self.pct_change, self.conv_end_method, self.rglztn_type,
                        self.rglztn_fctr, self.batch_method, self.batch_size, self.gmlr_type, self.score_method, self.max_columns,
                        self.intcpt_col_idx, self.bypass_agg, self.bypass_validation).run()
            else:
                # 6/15/23 self.TARGET_TRANSPOSE and self.TARGET_AS_LIST ARE CARRIED THRU FROM
                # ObjectOrienter IN ConfigRun. THEY ARE SHAPED FOR FULL DATA, AND WILL ONLY WORK WHEN TRAIN_SWNL HOLDS
                # THE FULL OBJECTS.  WILL EXCEPT IF TRAIN HAS BEEN CHOPPED TO BUILD DEV AND/OR TEST. GIVEN ALL THE SOURCES OF
                # CHANGE THAT IMPACT THE SIZE OF TRAIN_SWNL, IT IS NOT FEASIBLE TO BUILD TRANSPOSES ET AL FOR EVERY
                # CONTINGENCY, AND EVEN THEN, THE CONTINGENCIES HAPPEN SO CLOSE TO ENTRY TO CoreRunCode IT IS PROBABLY
                # BETTER TO JUST LET IT BE HANDLED IN THERE. SO IF EXCEPTS ABOVE, MOST LIKELY IS FOR MIS-SHAPES, JUST PASS
                # None FOR ALL THE HELPERS.
                self.WINNING_COLUMNS, self.COEFFS, self.TRAIN_RESULTS = \
                    gcrc.GMLRCoreRunCode(self.TRAIN_SWNL[0], self.WORKING_SUPOBJS[0][msod.QUICK_POSN_DICT()['HEADER']].reshape((1,-1)),
                        self.TRAIN_SWNL[1], None, None, self.data_run_orientation, self.target_run_orientation, self.conv_kill,
                        self.pct_change, self.conv_end_method, self.rglztn_type, self.rglztn_fctr, self.batch_method,
                        self.batch_size, self.gmlr_type, self.score_method, self.max_columns, self.intcpt_col_idx,
                        self.bypass_agg, self.bypass_validation).run()

            break


    def kfold_core_training_code(self):
        # unique run code for kfold CV for particular ML package

        # KERNEL MUST BE RECALCULATED EVERYTIME SINCE DIFFERENT rglztn_fctr COULD SELECT DIFFERENT COLUMNS (HENCE THATS
        # Y CANT PASS XTX TO GMLRCoreRunCode)

        while True:  # THIS IS HERE JUST TO BYPASS ALL CODE IF MULTI-COLUMN TARGET
            if gs.get_shape('TARGET', self.TRAIN_SWNL[1], self.target_run_orientation)[1] > 1:
                print(f'\n*** TARGET ENTERING GMLRCoreRunCode HAS MULTIPLE VECTORS, CANNOT DO MULTIPLE LINEAR REGRESSION. ***\n')
                break

            self.WINNING_COLUMNS, self.COEFFS, self.TRAIN_RESULTS = \
                gcrc.GMLRCoreRunCode(self.TRAIN_SWNL[0], self.WORKING_SUPOBJS[0][msod.QUICK_POSN_DICT()['HEADER']].reshape((1,-1)),
                    self.TRAIN_SWNL[1], None, None, self.data_run_orientation, self.target_run_orientation, self.conv_kill,
                    self.pct_change, self.conv_end_method, self.rglztn_type, self.rglztn_fctr, self.batch_method, self.batch_size,
                    self.gmlr_type, self.score_method, self.max_columns, self.intcpt_col_idx, self.bypass_agg,
                    self.bypass_validation).run()

            break
    # END TRAIN CALC DEFINITIONS ###################################################################################################

    # DEV CALC DEFINITIONS ##################################################################################################

    '''def rglztn_partition_iterator(self, number_of_partitions):
        INHERITED'''


    '''def base_dev_calc_module(self):
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
        # return module with package-specific commands to run current tests matrix
        if self.test_calc_select == 'S':
            # 3-22-22 BEAR FIX
            self.CSUTM_DF = sc.test_cases_calc_standard_configs(self.standard_config, self.TEST_SWNL[0], self.TRAIN_SWNL[0],
                                self.ARRAY_OF_NODES, self.SELECT_LINK_FXN, self.TRAIN_SWNL[1], self.activation_constant)


    def sub_test_calc_cmds(self):
        # return list with package - specific tests calc commands
        return {
            's': 'run special GMLR tests from standard configs'
        }                       # SPECIFIED IN CHILDREN   CANT USE 'NA'

    # END TEST CALC DEFINITIONS ##################################################################################################


    # TRAIN DATA DISPLAY ##############################################################################################################
    def train_summary_statistics_module(self):
        # returns module for printing summary statistics of train data
        gss.GMLRSummaryStatisticsPrint(self.TRAIN_OUTPUT_VECTOR, self.TRAIN_SWNL[1], 'None', self.train_data_error_algorithm()).print()


    def print_train_results_module(self):
        # returns module for printing train results for particular ML package
        try:
            pd.set_option('colheader_justify', 'center')
            pd.set_option('display.max_columns', 8)
            pd.set_option('display.width', 140)
            print(self.TRAIN_RESULTS)
            print()
        except:
            print(f'\n*** TRAIN RESULTS HAVE NOT BEEN CALCULATED YET, OR ERROR PRINTING RESULTS OBJECT ***\n')


    def train_filedump_module(self):
        # returns module for filedump of train results for particular ML package

        fxn = inspect.stack()[0][3]

        if not self.TRAIN_RESULTS.equals(pd.DataFrame({})):
            try: self.filedump_general_ml_setup_module()
            except: print(f'\n *** {self.this_module}.{fxn}() >>> EXCEPTION IN filedump_general_ml_setup_module() ***')
            try:
                self.wb = gsd.gmlr_setup_dump(self.wb, self.conv_kill, self.pct_change, self.conv_end_method, self.rglztn_type,
                    self.rglztn_fctr, self.batch_method, self.batch_size, {'L': 'Lazy', 'F': 'Forward', 'B': 'Backward'}[self.gmlr_type],
                    {'F': 'F-score', 'Q': 'RSQ', 'A': 'ADJ RSQ', 'R':'r'}[self.score_method], self.float_only, self.max_columns,
                    self.intcpt_col_idx)
            except: print(f'\n *** {self.this_module}.{fxn}() >>> EXCEPTION IN gmlr_setup_dump() ***')
            try: self.wb = gtrd.gmlr_train_results_dump(self.wb, self.TRAIN_RESULTS)
            except: print(f'\n *** {self.this_module}.{fxn}() >>> EXCEPTION IN gmlr_train_results_dump() ***')

            try: error = self.train_data_error_algorithm()  # DO THIS OUT HERE SO THAT self.TRAIN_OUTPUT_VECTOR IS CALCULATED B4 gss
            except:
                error = 'NaN'
                print(f'\n *** {self.this_module}.{fxn}() >>> EXCEPTION IN train_data_error_algorithm() ***')

            try:  self.wb = gss.GMLRSummaryStatisticsDump(self.wb, self.TRAIN_OUTPUT_VECTOR, self.TRAIN_SWNL[1], 'None',
                                                        error, 'TRAIN STATISTICS').dump()
            except: print(f'\n *** {self.this_module}.{fxn}() >>> EXCEPTION IN GMLRSummaryStatisticsDump() ***')

        else:
            print(f'\n*** TRAIN RESULTS HAVE NOT BEEN GENERATED YET ***\n')

    # END TRAIN DATA DISPLAY ##############################################################################################################

    # DEV DATA DISPLAY ##############################A##############################A##############################A##############
    def dev_summary_statistics_module(self):
        # returns module for printing summary statistics of dev data
        # BEAR
        try: gss.GMLRSummaryStatisticsPrint(self.DEV_OUTPUT_VECTOR, self.DEV_SWNL[1], 'None', self.dev_data_error_algorithm()).print()
        except: print(f'\n*** ERROR TRYING TO RUN GMLRSummaryStatisticsPrint ***')


    '''def print_dev_results_module(self):
        # returns module for printing dev results to screen for all ML packages
        pass
        INHERITED'''


    def dev_filedump_module(self):
        # returns module for filedump of dev results for all ML packages
        # BEAR
        try: self.wb = gdrd.general_dev_results_dump(self.wb, self.DEV_ERROR, self.RGLZTN_FACTORS)
        except: print(f'\n*** ERROR TRYING TO RUN general_dev_results_dump ***')
        # BEAR FIX GMLRSummaryStatisticsDump
        try: self.wb = gss.GMLRSummaryStatisticsDump(self.wb, self.DEV_OUTPUT_VECTOR, self.DEV_SWNL[1], 'None',
                                                      self.dev_data_error_algorithm(), 'DEV STATISTICS').dump()
        except: print(f'\n*** ERROR TRYING TO RUN GMLRSummaryStatisticsDump ***')

    # END DEV DATA DISPLAY ##############################A##############################A##############################A##############
    def test_summary_statistics_module(self):
        # returns module for printing summary statistics of tests data
        error = self.test_data_error_algorithm()  # DO THIS OUT HERE TO ENSURE self.TEST_OUTPUT_VECTOR IS UPDATED B4 gss
        gss.GMLRSummaryStatisticsPrint(self.TEST_OUTPUT_VECTOR, self.TEST_SWNL[1], 'None', error).print()


    '''
    def print_test_results_module(self):
        pass
        INHERITED
        '''


    def test_filedump_module(self):
        # package-specific module for saving tests results
        # self.wb = ggtrd.gmlr_generic_test_results_dump(self.wb, self.CSUTM_DF, self.DISPLAY_COLUMNS, self.display_criteria, self.display_rows)
        self.wb = gterd.general_test_results_dump(self.wb, self.CSUTM_DF, self.DISPLAY_COLUMNS, self.display_select, self.display_rows)

        error = self.test_data_error_algorithm()  # DO THIS OUT HERE TO ENSURE self.TEST_OUTPUT_VECTOR IS UPDATED B4 gss

        self.wb = gss.GMLRSummaryStatisticsDump(self.wb, self.TEST_OUTPUT_VECTOR, self.TEST_SWNL[1], 'None',
                                                error, 'TEST STATISTICS').dump()


    # END CALC DATA DISPLAY #############################################################A##############################A##############

    '''def base_return_fxn(self):
        INHERITED'''


    def return_fxn(self):
        # TAKE WINNING_COLUMNS OUT TO ConfigRun TO ENABLE DATA TO BE PERMANTENTLY CHOPPED DOWN TO WINNING_COLUMNS ONLY
        return *self.base_return_fxn(), self.WINNING_COLUMNS, self.COEFFS, self.TRAIN_RESULTS


    '''def run(self):  # MLRunTemplate loop
        INHERITED'''























if __name__ == '__main__':
    pass




























