import inspect, sys, time
import numpy as np, pandas as pd
from copy import deepcopy
from debug import get_module_name as gmn
from data_validation import validate_user_input as vui
from general_data_ops import get_shape as gs
from ML_PACKAGE import MLRunTemplate as mlrt
from ML_PACKAGE.MLREGRESSION import MLRegressionConfig as mlrc, MLRegressionCoreRunCode as mlrcrc, \
    mlr_output_vector_calc as movc, mlr_error_calc as mec
from ML_PACKAGE.MLREGRESSION.print_results import MLRegressionSummaryStatistics as mlrss, mlregression_setup_dump as mlrsd, \
    mlregression_train_results_dump as mlrtrd
from ML_PACKAGE.GENERIC_PRINT import general_dev_results_dump as gdrd, general_test_results_dump as gterd
from MLObjects import MLObject as mlo
from MLObjects.ObjectOrienter import MLObjectOrienter as mloo
from MLObjects.SupportObjects import master_support_object_dict as msod


# INHERITED ####################################################################################################################
# MODULE-SPECIFIC MAIN MENU DECLARATIONS ############################################################################
# END MODULE-SPECIFIC MAIN MENU DECLARATIONS ############################################################################

# GENERIC FUNCTIONS ####################################################################################################################
# row_column_display_select()            # user selects rows and columns for display on screen and filedump
# filedump_path()                        # hard-coded directory, user-entered filename
# filedump_general_ml_setup_module()     # module for filedump of general setup for all ML packages, used for train, dev, & test filedump
# dev_or_test_draw_params()              # module that returns the sample size to pull from train data for dev or test data sets
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
# random_dev_or_test_draw()              # randomly select user-specified quantity of examples for dev & test sets from (remaining) examples in TRAIN_SWNL
# partition_dev_or_test_draw()           # partition (remaining) examples in TRAIN_SWNL for selection into dev or test
# category_dev_or_test_draw()            # select examples for dev & test sets using categories in TRAIN_SWNL
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
#Xkfold_core_training_code()             # run code with randomization of nn parameters for k-fold
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
#  NONE AS OF 11-15-22


class MLRegressionRun(mlrt.MLRunTemplate):
    def __init__(self, standard_config, mlr_config, SUPER_RAW_NUMPY_LIST, RAW_SUPOBJS, SUPER_WORKING_NUMPY_LIST,
                 WORKING_SUPOBJS, data_run_orientation, target_run_orientation, refvecs_run_orientation, WORKING_CONTEXT,
                 WORKING_KEEP, TRAIN_SWNL, DEV_SWNL, TEST_SWNL, split_method, LABEL_RULES, number_of_labels, event_value,
                 negative_value, conv_kill, pct_change, conv_end_method, rglztn_type, rglztn_fctr, bypass_validation,
                 DATA_TRANSPOSE, TARGET_TRANSPOSE, TARGET_AS_LIST, XTX, batch_method, batch_size, intcpt_col_idx, TRAIN_RESULTS):


        super().__init__(standard_config, mlr_config, SUPER_RAW_NUMPY_LIST, RAW_SUPOBJS, SUPER_WORKING_NUMPY_LIST, WORKING_SUPOBJS,

                         data_run_orientation, target_run_orientation, refvecs_run_orientation,

                         WORKING_CONTEXT, WORKING_KEEP, TRAIN_SWNL, DEV_SWNL, TEST_SWNL, split_method, LABEL_RULES,
                         number_of_labels, event_value, negative_value, 'dum_mlr_conv_kill', 'dum_mlr_pct_change',
                         'dum_mlr_conv_end_method', rglztn_type, rglztn_fctr, bypass_validation, __name__)


        self.this_module = gmn.get_module_name(str(sys.modules[__name__]))
        fxn = '__init__'

        self.DATA_HEADER = self.WORKING_SUPOBJS[0][msod.QUICK_POSN_DICT()["HEADER"]]
        self.TARGET_HEADER = self.WORKING_SUPOBJS[1][msod.QUICK_POSN_DICT()["HEADER"]]
        self.DATA_TRANSPOSE = DATA_TRANSPOSE
        self.TARGET_TRANSPOSE = TARGET_TRANSPOSE
        self.TARGET_AS_LIST = TARGET_AS_LIST
        self.XTX = XTX

        # MLR PARAMETERS:
        self.batch_method = batch_method
        self.batch_size = batch_size
        self.intcpt_col_idx = intcpt_col_idx

        # PLACEHOLDERS
        self.WINNING_COLUMNS = []
        self.COEFFS = []

        self.TRAIN_DATA_TRANSPOSE = None
        self.TRAIN_XTX = None
        self.TRAIN_TARGET_TRANSPOSE = None
        self.TRAIN_TARGET_AS_LIST = None

        self.tc_method = 'MLRegression'

        # 2-23-22 CREATE UNIQUE OBJECT FOR MLR TO ALLOW FOR DISPLAY/DUMP OF TRAIN RESULTS
        self.TRAIN_RESULTS = TRAIN_RESULTS



    #  INHERITS FOR NOW ####################################################################################################
    # MODULE-SPECIFIC MAIN MENU DECLARATIONS ############################################################################
    def module_specific_main_menu_cmds(self):
        # module-specific top-level menu options
        # CANT USE 'ABCDEFGHIJKLNOPQRSTUVWXZ'
        return {'0': 'set regularization factor'}
    
    
    def module_specific_main_menu_operations(self):
        # execution code for post-run cmds                  # CANT USE 'ABCDEFGHIJKLNOPQRSTUVWXZ'
        if self.post_run_select == '0':
            self.rglztn_fctr = vui.validate_user_float(f'Enter regularization factor (currently {self.rglztn_fctr}) > ', min=0)

    # END MODULE-SPECIFIC MAIN MENU DECLARATIONS ############################################################################

    # OVERWRITES METHODS #######################################################################################3
    # GENERIC FUNCTIONS ####################################################################################################################
    def hyperparameter_display_module(self):
        # print hyperparameter settings to screen
        print(f'\nMULTIPLE LINEAR REGRESSION HYPERPARAMETER SETTINGS:')
        mlrc.MLRegressionConfig(self.standard_config, self.sub_config, self.SUPER_WORKING_NUMPY_LIST, self.WORKING_SUPOBJS,
            self.data_run_orientation, self.rglztn_type, self.rglztn_fctr, self.batch_method, self.batch_size,
            self.intcpt_col_idx, self.bypass_validation).print_parameters()


    '''def row_column_display_select(self):
        INHERITED '''

    '''def filedump_general_ml_setup_module(self):
        INHERITED'''


    def filedump_package_specific_setup_module(self):
        # module for filedump of setup for specific package, used for train, dev, & test filedump,
        self.wb = mlrsd.mlregression_setup_dump(self.wb, self.rglztn_type, self.rglztn_fctr, self.batch_method,
                                                self.batch_size, self.intcpt_col_idx)

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
        '''return module for package-specific core output algorithm, returns final output vector, overwritten in child'''
        # not in use
        pass


    def train_data_calc_algorithm(self):
        '''module for passing train data thru package-specific core algorithm, returns final output vector'''
        return movc.mlr_output_vector_calc(self.TRAIN_SWNL[0], self.data_run_orientation, self.COEFFS)


    def dev_data_calc_algorithm(self):
        '''module for passing dev data thru package-specific core algorithm, returns final output vector'''
        return movc.mlr_output_vector_calc(self.DEV_SWNL[0], self.data_run_orientation, self.COEFFS)


    def test_data_calc_algorithm(self):
        '''module for passing test data thru package-specific core algorithm, returns final output vector'''
        return movc.mlr_output_vector_calc(self.TEST_SWNL[0], self.data_run_orientation, self.COEFFS)


    def generic_ml_core_error_algorithm(self):
        # module for package-specific core error algorithm, returns total error
        # not in use
        pass


    def train_data_error_algorithm(self):
        '''module for passing train data thru package-specific error algorithm, returns total error'''
        self.TRAIN_OUTPUT_VECTOR = movc.mlr_output_vector_calc(self.TRAIN_SWNL[0], self.data_run_orientation, self.COEFFS)

        return mec.mlr_error_calc(self.TRAIN_OUTPUT_VECTOR, self.TRAIN_SWNL[1])


    def dev_data_error_algorithm(self):
        '''module for passing dev data thru package-specific error algorithm, returns total error'''

        self.DEV_OUTPUT_VECTOR = movc.mlr_output_vector_calc(self.DEV_SWNL[0], self.data_run_orientation, self.COEFFS)

        return mec.mlr_error_calc(self.DEV_OUTPUT_VECTOR, self.DEV_SWNL[1])


    def test_data_error_algorithm(self):
        '''module for passing test data thru package-specific error algorithm, returns total error'''

        self.TEST_OUTPUT_VECTOR = movc.mlr_output_vector_calc(self.TEST_SWNL[0], self.data_run_orientation, self.COEFFS)

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
        '''unique run code for particular ML package'''

        while True:
            if gs.get_shape('TARGET', self.TRAIN_SWNL[1], self.target_run_orientation)[1] > 1:
                # CANT DO MULTIPLE LINEAR REGRESSION ON MULTI-LABEL TARGET
                print(f'\n*** TARGET ENTERING MLRegressionCoreRunCode HAS MULTIPLE VECTORS, CANNOT DO MULTIPLE LINEAR REGRESSION. ***\n')
                break

            if len(self.DEV_SWNL) == 0 and len(self.TEST_SWNL) == 0:
                self.WINNING_COLUMNS, self.COEFFS, self.TRAIN_RESULTS = \
                    mlrcrc.MLRegressionCoreRunCode(*self.TRAIN_SWNL[:2], self.DATA_TRANSPOSE, self.TARGET_TRANSPOSE,
                        self.TARGET_AS_LIST, self.XTX, self.DATA_HEADER, self.data_run_orientation, self.target_run_orientation,
                        self.rglztn_type, self.rglztn_fctr, self.batch_method, self.batch_size, self.intcpt_col_idx,
                        self.bypass_validation).run()
            else:
                # 6/14/23 self.DATA_TRANSPOSE, self.TARGET_TRANSPOSE, self.TARGET_AS_LIST, self.XTX ARE CARRIED THRU FROM
                # ObjectOrienter IN ConfigRun. THEY ARE SHAPED FOR FULL DATA, AND WILL ONLY WORK WHEN TRAIN_SWNL HOLDS
                # THE FULL OBJECTS.  WILL EXCEPT IF TRAIN HAS BEEN CHOPPED TO BUILD DEV AND/OR TEST. GIVEN ALL THE SOURCES OF
                # CHANGE THAT IMPACT THE SIZE OF TRAIN_SWNL, IT IS NOT FEASIBLE TO BUILD TRANSPOSES ET AL FOR EVERY
                # CONTINGENCY, AND EVEN THEN, THE CONTINGENCIES HAPPEN SO CLOSE TO ENTRY TO CoreRunCode IT IS PROBABLY
                # BETTER TO JUST LET IT BE HANDLED IN THERE. SO IF EXCEPTS ABOVE, MOST LIKELY IS FOR MIS-SHAPES, JUST PASS
                # None FOR ALL THE HELPERS.

                self.WINNING_COLUMNS, self.COEFFS, self.TRAIN_RESULTS = \
                    mlrcrc.MLRegressionCoreRunCode(*self.TRAIN_SWNL[:2], None, None, None, None, self.DATA_HEADER,
                        self.data_run_orientation, self.target_run_orientation, self.rglztn_type, self.rglztn_fctr,
                        self.batch_method, self.batch_size, self.intcpt_col_idx, self.bypass_validation).run()

            break


    def kfold_core_training_code(self):
        # unique run code for kfold CV for particular ML package
        # 11/16/22 DONT NEED DEV ERROR HERE, CANT EARLY STOP.  cost_fxn IS ALWAYS SUM OF SQUARES

        if self.rebuild_kernel or self.batch_method=='M':
            # USE ObjectOrienter TO RECALCULATE XTX AND MAKE DATA_TRANSPOSE, TARGET_TRANSPOSE, TARGET_AS_LIST
            # if FULL BATCH, DATA & TARGET WILL BE DIFFERENT WHEN MOVING TO ANOTHER PARTITION (self.rebuild_kernel is True)
            # IF MINIBATCH, DATA & TARGET WILL BE DIFFERENT ON EVERY PASS, MUST RECREATE OBJECTS EVERY TIME (self.batch_method='M')
            print(f'\n*** BEAR ORIENTING JUNK IN MLRegressionRun KFOLD ***')

            t0 = time.time()  # BEAR

            # NOTES 4/23/23
            # MLRegressionCoreRunCode WAS TAKING LONGER THAN EXPECTED WHEN PASSED ALL OBJECTS, AND NONE OF THEM NEEDED TO
            # BE REORIENTED. FOUND THAT PASSING XTX TO ObjectOrienter AND DOING DATA VALIDATION ON IT WAS THE CULPRIT.
            # WHAT WAS TAKING 5.5 SECONDS WITH VALIDATION ON TAKES 0.02 SECONDS WITH VALIDATION OFF.  SO SET
            # bypass_validation TO True 4/23/23.

            KernelClass = mloo.MLObjectOrienter(
                                                DATA=self.TRAIN_SWNL[0],
                                                data_given_orientation=self.data_run_orientation,
                                                data_return_orientation=self.data_run_orientation,
                                                data_return_format='AS_GIVEN',

                                                DATA_TRANSPOSE=None,
                                                data_transpose_given_orientation=None,
                                                data_transpose_return_orientation=self.data_run_orientation,
                                                data_transpose_return_format='AS_GIVEN',

                                                XTX=None,
                                                xtx_return_format='ARRAY',

                                                XTX_INV=None,
                                                xtx_inv_return_format=None,

                                                target_is_multiclass=False,
                                                TARGET=self.TRAIN_SWNL[1],
                                                target_given_orientation=self.target_run_orientation,
                                                target_return_orientation=self.target_run_orientation,
                                                target_return_format='AS_GIVEN',

                                                TARGET_TRANSPOSE=None,
                                                target_transpose_given_orientation=None,
                                                target_transpose_return_orientation=self.target_run_orientation,
                                                target_transpose_return_format='AS_GIVEN',

                                                TARGET_AS_LIST=None,
                                                target_as_list_given_orientation=None,
                                                target_as_list_return_orientation=self.target_run_orientation,

                                                RETURN_OBJECTS=['DATA_TRANSPOSE', 'XTX',
                                                                'TARGET_TRANSPOSE', 'TARGET_AS_LIST'],

                                                bypass_validation=True,
                                                calling_module=self.this_module,
                                                calling_fxn=inspect.stack()[0][3]
            )

            self.TRAIN_DATA_TRANSPOSE = KernelClass.DATA_TRANSPOSE
            self.TRAIN_XTX = KernelClass.XTX
            self.TRAIN_TARGET_TRANSPOSE = KernelClass.TARGET_TRANSPOSE
            self.TRAIN_TARGET_AS_LIST = KernelClass.TARGET_AS_LIST

            del KernelClass

            print(f'\n*** END BEAR ORIENTING JUNK IN MLRegressionRun KFOLD (t = {time.time() - t0} sec) ***')

        self.WINNING_COLUMNS, self.COEFFS, self.TRAIN_RESULTS = \
            mlrcrc.MLRegressionCoreRunCode(*self.TRAIN_SWNL[:2], self.TRAIN_DATA_TRANSPOSE, self.TRAIN_TARGET_TRANSPOSE,
                self.TRAIN_TARGET_AS_LIST, self.TRAIN_XTX, self.DATA_HEADER, self.data_run_orientation,
                self.target_run_orientation, self.rglztn_type, self.rglztn_fctr, self.batch_method, self.batch_size,
                self.intcpt_col_idx, self.bypass_validation).run()


        self.TRAIN_DATA_TRANSPOSE = None
        self.TRAIN_XTX = None
        self.TRAIN_TARGET_TRANSPOSE = None
        self.TRAIN_TARGET_AS_LIST = None

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
        # return module with package-specific commands to run current test matrix
        if self.test_calc_select == 'S':
            # 3-22-22 BEAR FIX
            self.CSUTM_DF = sc.test_cases_calc_standard_configs(self.standard_config, self.TEST_SWNL[0],
                            self.TRAIN_SWNL[0], self.WORKING_SUPOBJS[0][msod.QUICK_POSN_DICT()["HEADER"]])


    def sub_test_calc_cmds(self):
        # return list with package - specific test calc commands
        return {
            's': 'run special MLRegression test from standard configs'
        }                       # SPECIFIED IN CHILDREN   CANT USE 'NA'

    # END TEST CALC DEFINITIONS ##################################################################################################

    # TRAIN DATA DISPLAY ##############################################################################################################
    def train_summary_statistics_module(self):
        # returns module for printing summary statistics of train data
        mlrss.MLRegressionSummaryStatisticsPrint(self.TRAIN_OUTPUT_VECTOR, self.TRAIN_SWNL[1], 'None',
                                                        self.train_data_error_algorithm()).print()


    def print_train_results_module(self):
        # returns module for printing train results for particular ML package

        while True:
            if self.TRAIN_RESULTS.equals({}):
                print(f'\n*** TRAIN RESULTS HAVE NOT BEEN CALCULATED YET ***')
                break

            try:
                pd.set_option('colheader_justify', 'center')
                pd.set_option('display.max_columns', 8)
                pd.set_option('display.max_rows', None)
                pd.set_option('display.width', 0)

                SORT_DICT = {'P':'p VALUE','C':'COEFFS', 'A':'ABSOLUTE'} if self.rglztn_fctr==0 else {'C':'COEFFS', 'A':'ABSOLUTE'}
                allowed = "".join(list(SORT_DICT.keys()))
                _text = f'\nSort by{" p-values(p), " if self.rglztn_fctr==0 else " "}coeffs(c), or absolute value of coeffs(a)? > '
                sort_column = SORT_DICT[vui.validate_user_str(_text, allowed)]
                sort_order = {'A':True,'D':False}[vui.validate_user_str(f'\nSort ascending(a) or descending(d)? > ', 'AD')]

                del SORT_DICT, allowed, _text

                DUM = self.TRAIN_RESULTS.copy()

                # GET A COPY OF FULL REGRESSION RESULTS (TOP ROW IN 'OVERALL R', 'OVERALL R2', 'OVERALL ADJ R2', 'OVERALL F')
                OVERALL_HOLDER = DUM.iloc[0, -4:].copy()
                # SET THOSE POSITIONS TO '-'
                DUM.iloc[0, -4:] = '-'  # SEE .fillna('-') IN MLRegressionCoreRunCode

                # DO THE SORT
                if sort_column in ['p VALUE', 'COEFFS']:
                    DUM.sort_values(by=[('      ',sort_column)], ascending=sort_order, inplace=True)
                elif sort_column == 'ABSOLUTE':
                    DUM.sort_values(by=[('      ','COEFFS')], key=abs, ascending=sort_order, inplace=True)

                # FILL TOP ROW IN 'OVERALL R', 'OVERALL R2', 'OVERALL ADJ R2', 'OVERALL F'
                DUM.iloc[0, -4:] = OVERALL_HOLDER

                print(DUM)

                del sort_column, sort_order, DUM, OVERALL_HOLDER

                print()

            except:
                print(f'\n*** ERROR PRINTING RESULTS OBJECT ***')

            break


    def train_filedump_module(self):
        # returns module for filedump of train results for particular ML package
        if not self.TRAIN_RESULTS.equals(pd.DataFrame({})):
            try:
                self.filedump_general_ml_setup_module()
                self.wb = mlrsd.mlregression_setup_dump(self.wb, self.rglztn_type, self.rglztn_fctr, self.batch_method,
                    self.batch_size, self.intcpt_col_idx)
                self.wb = mlrtrd.mlregression_train_results_dump(self.wb, self.TRAIN_RESULTS, self.rglztn_fctr)
                error = self.train_data_error_algorithm()  # DO THIS OUT HERE SO THAT self.TRAIN_OUTPUT_VECTOR IS CALCULATED B4 gss
                self.wb = mlrss.MLRegressionSummaryStatisticsDump(self.wb, self.TRAIN_OUTPUT_VECTOR, self.TRAIN_SWNL[1], 'None',
                                                            error, 'TRAIN STATISTICS').dump()
            except:
                print(f'\n*** EXCEPTION TRYING TO DUMP TRAIN RESULTS TO FILE IN MLRegressionRun.train_filedump_module() ***')
        else:
            print(f'\n*** TRAIN RESULTS HAVE NOT BEEN GENERATED YET ***\n')

    # END TRAIN DATA DISPLAY ##############################################################################################################

    # DEV DATA DISPLAY ##############################A##############################A##############################A##############
    def dev_summary_statistics_module(self):
        # returns module for printing summary statistics of dev data
        mlrss.MLRegressionSummaryStatisticsPrint(self.DEV_OUTPUT_VECTOR, self.DEV_SWNL[1], 'None',
                                                      self.dev_data_error_algorithm()).print()


    '''def print_dev_results_module(self):
        # returns module for printing dev results to screen for all ML packages
        pass
        INHERITED'''


    def dev_filedump_module(self):
        # returns module for filedump of dev results for all ML packages
        # BEAR FIX MLRegressionSummaryStatistics
        try:
            self.wb = gdrd.general_dev_results_dump(self.wb, self.DEV_ERROR, self.RGLZTN_FACTORS)
            try:
                self.wb = mlrss.MLRegressionSummaryStatisticsDump(self.wb, self.DEV_OUTPUT_VECTOR, self.DEV_SWNL[1], 'None',
                                                      self.dev_data_error_algorithm(), 'DEV STATISTICS').dump()
            except:
                print(f'\n*** MLRegressionRun.dev_filedump_module() >>> EXCEPTION CALLING MLRegressionSummaryStatisticsDump() ***')

        except:
            print(f'\n*** MLRegressionRun.dev_filedump_module() >>> EXCEPTION CALLING general_dev_results_dump() ***')



    # END DEV DATA DISPLAY ##############################A##############################A##############################A##############
    def test_summary_statistics_module(self):
        # returns module for printing summary statistics of test data
        error = self.test_data_error_algorithm()  # DO THIS OUT HERE TO ENSURE self.TEST_OUTPUT_VECTOR IS UPDATED B4 gss
        mlrss.MLRegressionSummaryStatisticsPrint(self.TEST_OUTPUT_VECTOR, self.TEST_SWNL[1], 'None', error).print()


    '''
    def print_test_results_module(self):
        pass
        INHERITED
        '''


    def test_filedump_module(self):
        # package-specific module for saving test results
        # self.wb = ggtrd.mlr_generic_test_results_dump(self.wb, self.CSUTM_DF, self.DISPLAY_COLUMNS, self.display_criteria, self.display_rows)
        self.wb = gterd.general_test_results_dump(self.wb, self.CSUTM_DF, self.DISPLAY_COLUMNS, self.display_select, self.display_rows)

        error = self.test_data_error_algorithm()  # DO THIS OUT HERE TO ENSURE self.TEST_OUTPUT_VECTOR IS UPDATED B4 gss

        self.wb = mlrss.MLRegressionSummaryStatisticsDump(self.wb, self.TEST_OUTPUT_VECTOR, self.TEST_SWNL[1], 'None',
                                                error, 'TEST STATISTICS').dump()


    # END CALC DATA DISPLAY #############################################################A##############################A##############

    '''def base_return_fxn(self):
        INHERITED'''


    def return_fxn(self):

        return *self.base_return_fxn(), self.WINNING_COLUMNS, self.COEFFS, self.TRAIN_RESULTS


    '''def run(self):  # MLRunTemplate loop
        INHERITED'''












if __name__ == '__main__':

    import numpy as np
    from MLObjects.TestObjectCreators.SXNL import CreateSXNL as csxnl
    from debug import IdentifyObjectAndPrint as ioap

    DATA = pd.read_csv(r'C:\Users\Bill\Documents\WORK STUFF\RESUME\1 - OTHER\SRP\beer_reviews.csv',
                      nrows=100,
                      header=0).dropna(axis=0)

    DATA = DATA[DATA.keys()[[3, 4, 5, 8, 9, 11]]] #[[3,4,5,7,8 , 9, 11]]]


    TARGET = DATA['review_overall']
    TARGET_HEADER = [['review_overall']]
    TARGET = TARGET.to_numpy().reshape((1,-1))
    TARGET_TRANSPOSE = None
    TARGET_AS_LIST = None

    DATA = DATA.drop(columns=['review_overall'])
    DATA_TRANSPOSE = None

    RAW_DATA = DATA.copy()
    RAW_DATA_HEADER = np.fromiter(RAW_DATA.keys(), dtype='<U50').reshape((1,-1))
    RAW_DATA = RAW_DATA.to_numpy()
    XTX = None

    data_given_orientation = 'ROW'
    target_given_orientation = 'ROW'
    refvecs_given_orientation = 'COLUMN'

    SXNLClass = csxnl.CreateSXNL(rows=None,
                                 bypass_validation=False,
                                 data_return_format='ARRAY',
                                 data_return_orientation=data_given_orientation,
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
                                 target_return_orientation=target_given_orientation,
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
                                 refvecs_return_orientation=refvecs_given_orientation,
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
    data_run_orientation = SXNLClass.data_current_orientation
    target_run_orientation = SXNLClass.target_current_orientation
    refvecs_run_orientation = SXNLClass.refvecs_current_orientation

    SXNLClass.expand_data(expand_as_sparse_dict={'P':True,'A':False}[vui.validate_user_str(f'\nExpand as sparse dict(p) or array(a) > ', 'AP')],
                          auto_drop_rightmost_column=False)
    SWNL = SXNLClass.SXNL
    WORKING_SUPOBJS = SXNLClass.SXNL_SUPPORT_OBJECTS

    del SXNLClass

    try:
        DumClass = mlo.MLObject(
            np.insert(SWNL[0], len(SWNL[0]) if data_given_orientation=="COLUMN" else len(SWNL[0][0]), 1, axis=0 if data_given_orientation=="COLUMN" else 1),
            data_run_orientation, name="SWNL[0]", return_orientation='AS_GIVEN', return_format='AS_GIVEN',
            bypass_validation=True, calling_module="MLRegressionRun", calling_fxn='test')
        XTX_INV = DumClass.return_XTX_INV(return_format="ARRAY")
        del DumClass, XTX_INV
        print(f'\n*** SWNL[0] INVERTED ***\n')
    except:
        raise Exception(f'*** SWNL[0] DOES NOT INVERT ***')




    WORKING_CONTEXT = []
    WORKING_KEEP = RAW_SUPOBJS[0][msod.QUICK_POSN_DICT()["HEADER"]]

    TRAIN_SWNL = []
    DEV_SWNL = []
    TEST_SWNL = []

    standard_config = 'None'
    mlr_config = 'None'

    split_method = 'None'
    LABEL_RULES = []
    number_of_labels = 1
    event_value = ''
    negative_value = ''
    mlr_conv_kill = 1
    mlr_pct_change = 0
    mlr_conv_end_method = 'KILL'
    mlr_rglztn_type = 'RIDGE'
    mlr_rglztn_fctr = 10000
    mlr_batch_method = 'M' #'B'
    mlr_batch_size = 1000
    intcpt_col_idx = None
    TRAIN_RESULTS = pd.DataFrame({})

    bypass_validation = False

    TRAIN_SWNL, DEV_SWNL, TEST_SWNL, WINNING_COLUMNS, COEFFS, TRAIN_RESULTS = \
        MLRegressionRun(standard_config, mlr_config, SRNL, RAW_SUPOBJS, SWNL, WORKING_SUPOBJS, data_run_orientation,
            target_run_orientation, refvecs_run_orientation, WORKING_CONTEXT, WORKING_KEEP, TRAIN_SWNL, DEV_SWNL,
            TEST_SWNL, split_method, LABEL_RULES, number_of_labels, event_value, negative_value, 'dum_conv_kill',
            'dum_pct_change', 'dum_conv_end_method', mlr_rglztn_type, mlr_rglztn_fctr, bypass_validation, DATA_TRANSPOSE,
            TARGET_TRANSPOSE, TARGET_AS_LIST, XTX, mlr_batch_method, mlr_batch_size, intcpt_col_idx, TRAIN_RESULTS).run()


    print(f'\nFINAL RETURN OF WORKING DATA LOOKS LIKE:')
    ioap.IdentifyObjectAndPrint(SWNL[0], 'DATA', 'MLRegressionConfigRun', 20, 10, 0, 0)



    print(f'\nFINAL RETURN OF WORKING DATA SUPPORT OBJECT LOOKS LIKE:')
    print(WORKING_SUPOBJS[0])





























