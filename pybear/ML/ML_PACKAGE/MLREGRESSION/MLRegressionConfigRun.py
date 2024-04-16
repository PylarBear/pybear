import sys, inspect, warnings
from copy import deepcopy
import numpy as np, pandas as pd
from debug import get_module_name as gmn, IdentifyObjectAndPrint as ioap
from data_validation import validate_user_input as vui, arg_kwarg_validater as akv
from general_data_ops import get_shape as gs
from MLObjects import MLObject as mlo
from MLObjects.SupportObjects import master_support_object_dict as msod
from MLObjects.ObjectOrienter import MLObjectOrienter as mloo
from ML_PACKAGE import MLConfigRunTemplate as mlcrt
from ML_PACKAGE.MLREGRESSION import MLRegressionConfig as mlrc, MLRegressionRun as mlrr
from ML_PACKAGE.GENERIC_PRINT import DictMenuPrint as dmp, train_results_excel_dump as tred


# INHERITS #############################################################################################################
# dataclass_mlo_loader()             Loads an instance of MLObject for DATA.
# intercept_manager()                Locate columns of constants in DATA & handle. As of 11/15/22 only for MLR, MI, and GMLR.
# insert_intercept()                 Insert a column of ones in the 0 index of DATA.
# delete_column()                    Delete a column from DATA and respective holder objects.
# run_module_input_tuple()           tuple of base params that pass into run_module for all ML packages
# base_post_run_options_module()     holds post run options applied to all ML packages
# return_fxn_base()                  values returned from all ML packages
# configrun()                        runs config_module() & run_module()

# OVERWRITES ###########################################################################################################
# config_module()                    gets configuration source, returns configuration parameters for particular ML package
# run_module()                       returns run module for particular ML package
# return_fxn()                       returns user-specified output, in addition to return_fxn_base()
# sub_post_run_cmds()                package-specific options available to modify WORKING_DATA after run
# sub_post_run_options_module()      holds post run options unique to particular ML package

# UNIQUE ###############################################################################################################



#CALLED BY ML
class MLRegressionConfigRun(mlcrt.MLConfigRunTemplate):

    def __init__(self, standard_config, mlr_config, SUPER_RAW_NUMPY_LIST, RAW_SUPOBJS, SUPER_WORKING_NUMPY_LIST,
                 WORKING_SUPOBJS, data_given_orientation, target_given_orientation, refvecs_given_orientation, WORKING_CONTEXT,
                 WORKING_KEEP, split_method, LABEL_RULES, number_of_labels, event_value, negative_value, mlr_rglztn_type,
                 mlr_rglztn_fctr,
                 # UNIQUE TO MLRegression
                 mlr_batch_method, mlr_batch_size, MLR_OUTPUT_VECTOR):

        # ORIENT AS [ [] = ROWS ] FOR SPEED IN MLRegression, SINCE DONT HAVE TO JOCKEY COLUMNS HERE LIKE GMLR & MI
        # TRANSPOSE TARGET SO DONT HAVE TO REPEATEDLY TRANSPOSE IN MLRegression
        # 10/13/22, 4/17/23 KEEP TARGET AS LIST, MLRegression CONVERTS ALL TARGETS TO LIST

        self.this_module = gmn.get_module_name(str(sys.modules[__name__]))
        fxn = '__init__'

        data_run_format='AS_GIVEN'
        data_run_orientation='ROW'  # 5/5/23 THIS CANNOT CHANGE BECAUSE OF matmuls AND statsmodels!!!!
        target_run_format='ARRAY' # 4/16/23 CHANGED FROM DEPENDENCE ON is_list/is_dict TO ALWAYS ARRAY
        target_run_orientation='ROW'  # 5/5/23 THIS CANNOT CHANGE BECAUSE OF matmuls AND statsmodels!!!!
        refvecs_run_format='ARRAY'
        refvecs_run_orientation='COLUMN'
        xtx_run_format='ARRAY'    # ALWAYS RETURN AS ARRAY, USES sd.hybrid_matmul IF is_dict IS True

        super().__init__(standard_config, mlr_config, SUPER_RAW_NUMPY_LIST, RAW_SUPOBJS, SUPER_WORKING_NUMPY_LIST,
            WORKING_SUPOBJS, data_given_orientation, target_given_orientation, refvecs_given_orientation,
            data_run_orientation, target_run_orientation, refvecs_run_orientation, data_run_format, target_run_format,
            refvecs_run_format, WORKING_CONTEXT, WORKING_KEEP, split_method, LABEL_RULES, number_of_labels, event_value,
            negative_value, 'dum_mlr_conv_kill', 'dum_mlr_pct_change', 'dum_mlr_conv_end_method', mlr_rglztn_type,
            mlr_rglztn_fctr, __name__)

        self.xtx_run_format = akv.arg_kwarg_validater(xtx_run_format, 'xtx_run_format', ['ARRAY', 'SPARSE_DICT'],
                                                      self.this_module, fxn)

        self.batch_method = mlr_batch_method
        self.batch_size = mlr_batch_size
        self.OUTPUT_VECTOR = MLR_OUTPUT_VECTOR     # 11/13/22 JUST PASSES THRU

        ########################################################################################################################
        # ORIENT/FORMAT TARGET & REFVEC OBJECTS ################################################################################

        # SRNL IS IN super(), SWNL HANDLED IN CHILDREN FOR EASIER HANDLING OF DIFFERENT "RETURN_OBJECTS" FOR DIFFERENT PACKAGES

        # TARGET & REFVECS CAN BE SET HERE ONCE AND FOR ALL. BECAUSE OF THE COLUMN CHOP AND INTERCEPT OPTIONS IN & AROUND Config,
        # INITIALLY SET DATA, OTHER DATA OBJECTS, AND data_run_format/orientation HERE. ABSOLUTELY NEED selfs FOR data_format &
        # orientation TO BE MADE HERE, SO CAN BE CALLED BY mloo IN config_module().
        # ANOTHER mloo FOR DATA MUST BE AFTER Config TO REBUILD DATA_TRANSPOSE & XTX IN CASE THERE ARE CHANGES TO COLUMNS

        print(f'\n    BEAR IN MLRegressionConfigRun Orienting WORKING DATA & TARGET IN init.  Patience...')

        # BREAK CONNECTIONS OF SUPER_WORKING_NUMPY_LIST WITH SUPER_WORKING_NUMPY_LIST IN OUTER SCOPE. DO NOT CHANGE THIS,
        # BLEW AN ENTIRE SATURDAY 6/24/23 DIAGNOSING THIS.
        # SUPER_WORKING_NUMPY_LIST IN ML() WAS GETTING INEXTRICABLY TRANSPOSED FROM BEFORE MLRegressionConfigRun TO AFTER IT,
        # EVEN THOUGH IT IS SIMPLY PASSED TO IT, THEN RETURNED AS SUPER_WORKING_NUMPY_LIST_HOLDER. TURNS OUT THIS ObjectOrienter
        # STEP WAS TRANSPOSING SUPER_WORKING_NUMPY_LIST IN THE ML() SCOPE VIA self.SUPER_WORKING_NUMPY_LIST IN THIS SCOPE.
        self.SUPER_WORKING_NUMPY_LIST = [deepcopy(_) if isinstance(_, dict) else _.copy() for _ in SUPER_WORKING_NUMPY_LIST]

        SWNLOrienterClass = mloo.MLObjectOrienter(
                                                    DATA=self.SUPER_WORKING_NUMPY_LIST[0],
                                                    data_given_orientation=self.data_given_orientation,
                                                    data_return_orientation=self.data_run_orientation,
                                                    data_return_format=self.data_run_format,

                                                    DATA_TRANSPOSE=None,
                                                    data_transpose_given_orientation=None,
                                                    data_transpose_return_orientation=self.data_run_orientation,
                                                    data_transpose_return_format=self.data_run_format,

                                                    target_is_multiclass=self.working_target_is_multiclass,
                                                    TARGET=self.SUPER_WORKING_NUMPY_LIST[1],
                                                    target_given_orientation=self.target_given_orientation,
                                                    target_return_orientation=self.target_run_orientation,
                                                    target_return_format=self.target_run_format,

                                                    TARGET_TRANSPOSE = None,
                                                    target_transpose_given_orientation = None,
                                                    target_transpose_return_orientation = self.target_run_orientation,
                                                    target_transpose_return_format = self.target_run_format,

                                                    TARGET_AS_LIST = None,
                                                    target_as_list_given_orientation = None,
                                                    target_as_list_return_orientation = self.target_run_orientation,

                                                    XTX=None,
                                                    xtx_return_format=self.xtx_run_format,

                                                    RETURN_OBJECTS=['DATA', 'DATA_TRANSPOSE', 'TARGET', 'TARGET_TRANSPOSE',
                                                                    'TARGET_AS_LIST', 'XTX'],

                                                    bypass_validation=self.bypass_validation,
                                                    calling_module=self.this_module,
                                                    calling_fxn=fxn
        )

        # BECAUSE ObjectOrienter WASNT BUILT TO HANDLE REFVECS
        RefVecsClass = mlo.MLObject(self.SUPER_WORKING_NUMPY_LIST[2], self.refvecs_given_orientation,
            name='REFVECS', return_orientation=self.refvecs_run_orientation, return_format=self.refvecs_run_format,
            bypass_validation=self.bypass_validation, calling_module=self.this_module, calling_fxn=fxn)

        # BECAUSE ANY OF THESE COULD BE 'AS_GIVEN', GET ACTUALS
        self.SUPER_WORKING_NUMPY_LIST[0] = SWNLOrienterClass.DATA
        self.data_run_format = SWNLOrienterClass.data_return_format
        self.data_run_orientation = SWNLOrienterClass.data_return_orientation
        self.DATA_TRANSPOSE = SWNLOrienterClass.DATA_TRANSPOSE
        self.XTX = SWNLOrienterClass.XTX

        self.SUPER_WORKING_NUMPY_LIST[1] = SWNLOrienterClass.TARGET
        self.target_run_format = SWNLOrienterClass.target_return_format
        self.target_run_orientation = SWNLOrienterClass.target_return_orientation
        self.TARGET_TRANSPOSE = SWNLOrienterClass.TARGET_TRANSPOSE
        self.TARGET_AS_LIST = SWNLOrienterClass.TARGET_AS_LIST

        del SWNLOrienterClass

        self.SUPER_WORKING_NUMPY_LIST[2] = RefVecsClass.OBJECT
        self.refvecs_run_format = RefVecsClass.current_format
        self.refvecs_run_orientation = RefVecsClass.current_orientation

        del RefVecsClass

        print(f'\n    BEAR IN MLRegressionConfigRun Orienting WORKING DATA & TARGET IN init.  Done')

        # END ORIENT SWNL & MAKE OTHER OBJECTS #################################################################################
        ########################################################################################################################

        self.WINNING_COLUMNS = []
        self.COEFFS = []
        self.TRAIN_RESULTS = pd.DataFrame({})

        self.context_text_holder = ''    # CREATE A STRING THAT RECORDS ALL OPERATIONS HERE, THEN IS APPENDED TO CONTEXT
        self.CONTEXT_UPDATE_HOLDER = []      # CREATE THIS TO HOLD CHOPS, THEN PUT INTO CONTEXT AFTER context_text_holder


    # INHERITS #############################################################################################################
    # dataclass_mlo_loader()             Loads an instance of MLObject for DATA.
    # intercept_manager()                Locate columns of constants in DATA & handle. As of 11/15/22 only for MLR, MI, and GMLR.
    # run_module_input_tuple()           tuple of base params that pass into run_module for all ML packages
    # base_post_run_options_module()     holds post run options applied to all ML packages
    # return_fxn_base()                  values returned from all ML packages
    # configrun()                        runs config_module() & run_module()
    # delete_column()                    Delete a column from DATA and respective holder objects.

    # OVERWRITES #######################################################################################################
    def config_module(self):
        # config_module()                    gets configuration source, returns configuration parameters for particular ML package

        fxn = inspect.stack()[0][3]

        # GET NUMBER OF COLUMNS B4 GOING INTO Config & intercept_manager(), WHERE COLUMNS COULD CHANGE
        start_data_cols = gs.get_shape('DATA', self.SUPER_WORKING_NUMPY_LIST[0], self.data_run_orientation)[1]
        start_batch_method = self.batch_method

        self.SUPER_WORKING_NUMPY_LIST, self.WORKING_SUPOBJS, self.rglztn_type, self.rglztn_fctr, self.batch_method, \
            self.batch_size, self.intcpt_col_idx = \
                mlrc.MLRegressionConfig(self.standard_config, self.sub_config, self.SUPER_WORKING_NUMPY_LIST,
                self.WORKING_SUPOBJS, self.data_run_orientation, self.rglztn_type, self.rglztn_fctr, self.batch_method,
                self.batch_size, self.intcpt_col_idx, self.bypass_validation).config()

        self.intercept_manager()

        end_data_cols = gs.get_shape('DATA', self.SUPER_WORKING_NUMPY_LIST[0], self.data_run_orientation)[1]
        end_batch_method = self.batch_method

        # IF NUMBER OF COLUMNS IN DATA CHANGED AND BATCH METHOD IS FULL BATCH (MINIBATCH WOULD BE REBUILT IN CoreRun ANYWAY, OR
        # JUST CHANGED FROM MINI-BATCH TO FULL, REBUILD DATA_TRANSPOSE & XTX
        if (start_data_cols != end_data_cols and self.batch_method=='B') or (end_batch_method=='B' and start_batch_method=='M'):
            ########################################################################################################################
            # REMAKE OTHER OBJECTS #################################################################################################

            # BECAUSE OF THE COLUMN CHOP AND INTERCEPT OPTION, A mloo MUST BE AFTER Config TO REBUILD DATA_TRANSPOSE & XTX
            # IN CASE THERE ARE CHANGES TO DATA COLUMNS.

            print(f'\n    BEAR IN MLRegressionConfigRun building DATA_TRANSPOSE & XTX AFTER Config.  Patience...')

            SWNLOrienterClass = mloo.MLObjectOrienter(
                                                        DATA=self.SUPER_WORKING_NUMPY_LIST[0],
                                                        data_given_orientation=self.data_run_orientation,
                                                        data_return_orientation=self.data_run_orientation,
                                                        data_return_format=self.data_run_format,

                                                        DATA_TRANSPOSE=None,
                                                        data_transpose_given_orientation=None,
                                                        data_transpose_return_orientation=self.data_run_orientation,
                                                        data_transpose_return_format=self.data_run_format,

                                                        XTX=None,
                                                        xtx_return_format=self.xtx_run_format,

                                                        RETURN_OBJECTS=['DATA_TRANSPOSE', 'XTX'],

                                                        bypass_validation=True,
                                                        calling_module=self.this_module,
                                                        calling_fxn=fxn
            )

            print(f'\n    BEAR IN MLRegressionConfigRun building DATA_TRANSPOSE & XTX AFTER Config.  Done')

            self.DATA_TRANSPOSE = SWNLOrienterClass.DATA_TRANSPOSE
            self.XTX = SWNLOrienterClass.XTX

            del SWNLOrienterClass
            # END REMAKE OTHER OBJECTS #############################################################################################
            ########################################################################################################################

        del start_data_cols, end_data_cols, start_batch_method, end_batch_method


    def run_module(self):
        # run_module()                       returns run module for particular ML package

        self.TRAIN_SWNL, self.DEV_SWNL, self.TEST_SWNL, self.PERTURBER_RESULTS, self.WINNING_COLUMNS, self.COEFFS, self.TRAIN_RESULTS = \
            mlrr.MLRegressionRun(*self.run_module_input_tuple(), self.DATA_TRANSPOSE, self. TARGET_TRANSPOSE, self.TARGET_AS_LIST,
                self.XTX, self.batch_method, self.batch_size, self.intcpt_col_idx, self.TRAIN_RESULTS).run()

        self.context_text_holder += f' Ran MLRegression.'


    def return_fxn(self):
        # return_fxn()                       returns user-specified output, in addition to return_fxn_base()

        return *self.return_fxn_base(), self.batch_method, self.batch_size, self.OUTPUT_VECTOR

    # return_fxn_base() is self.SUPER_RAW_NUMPY_LIST, self.RAW_SUPOBJS, self.SUPER_WORKING_NUMPY_LIST, self.WORKING_SUPOBJS, \
    #     self.WORKING_CONTEXT, self.WORKING_KEEP, self.split_method, self.SUPER_RAW_NUMPY_LIST, self.RAW_SUPOBJS,
    #     self.SUPER_WORKIelf.LABEL_RULES, self.number_of_labels, self.event_value, self.negative_value, self.conv_kill,
    #     self.pct_change, self.conv_end_method, self.rglztn_type, self.rglztn_fctr


    def sub_post_run_cmds(self):
    # package-specific options available to modify WORKING_DATA after run
        return {'k': 'operations for MLR WINNING COLUMNS'}


    def sub_post_run_options_module(self):
    # holds post run options unique to particular ML package

        fxn = inspect.stack()[0][3]

        if self.post_configrun_select == 'K':

            POST_MLR_MENU = {
                'w': 'keep all columns used in MLR',
                'p': 'keep only WINNING COLUMNS below certain p-value',
                'c': 'keep only WINNING COLUMNS above certain abs(COEFF)',
                'x': 'keep top x from WINNING COLUMNS based on p-value',
                'y': 'keep top x from WINNING COLUMNS based on abs(COEFF)',
                'r': 'print TRAIN RESULTS',
                'd': 'dump TRAIN RESULTS to file',
                'a': 'accept and continue'
            }

            # MLRCoreRunCode DOESNT ACTUALLY SELECT WINNING COLUMNS, CAN ONLY RUN EVERYTHING GIVEN, SO WINNING_COLUMNS
            # IS ALWAYS range(0, _cols).

            # TRAIN_RESULTS IS COMING IN FROM MLRegressionCoreRunCode > MLRegressionRun > run_module() AS DF WITH
            # ROWS HOLDING INFO ABT A COLUMN IN DATA, SORTED ASCENDING BY p VALUE WITH INTERCEPT AT TOP, IF GIVEN

            # WINNING_COLUMNS IS COMING IN FROM MLRegressionCoreRunCOde > MLRegressionRun > run_module() AS ARRAY OF
            # WINNING IDXS IN DATA WITH intcpt_col_idx FIRST IF GIVEN, SORTED ASCENDING BY p VALUE (SO WHAT DOES THAT
            # MEAN WHEN RIDGE IS USED --- ALL P VALUES ARE '-', SO DOES SORT DO ANYTHING?)


            def sort_mask_builder(column, key=None, ascending=True):
                # IF INTERCEPT IS IN WINNING_DATA, ALWAYS PUTS IT AT THE TOP OF TRAIN_RESULTS AND COL IDX 0 IN DATA
                if key==None: SORT_MASK = np.argsort(self.TRAIN_RESULTS[column].to_numpy())
                elif key==abs: SORT_MASK = np.argsort(self.TRAIN_RESULTS[column].abs().to_numpy())
                if ascending==False: SORT_MASK = np.flip(SORT_MASK)
                if not self.intcpt_col_idx is None:
                    # IF THERE WAS AN INTERCEPT, MOVE TO FIRST POSITION IN SORT_MASK
                    SORT_MASK = np.insert(SORT_MASK[SORT_MASK != 0], 0, 0, axis=0)

                return SORT_MASK


            def sort_train_results(column, SORT_MASK=None, key=None, ascending=True):
                # KEEP OVERALL R, R2, ADJ R2, F AT THE TOP
                OVERALL_HOLDER = self.TRAIN_RESULTS.iloc[0, -4:].copy()                     # CAPTURE OVERALL SCORES
                # SORT BASED ON column W INTERCEPT ALWAYS AT THE TOP
                if SORT_MASK is None: SORT_MASK = sort_mask_builder(column, key, ascending)
                self.TRAIN_RESULTS.iloc[0, -4:] = '-'                      # SEE MLRegressionCoreRunCode fillna('-')
                self.TRAIN_RESULTS = self.TRAIN_RESULTS.iloc[SORT_MASK, :]
                self.TRAIN_RESULTS.iloc[0, -4:] = OVERALL_HOLDER
                del OVERALL_HOLDER, SORT_MASK

            disallowed = ''

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if np.array_equiv(self.TRAIN_RESULTS[('      ','p VALUE')].to_numpy(), ['-' for _ in range(len(self.TRAIN_RESULTS))]):
                    disallowed += 'PX'

            while True:

                if len(self.WINNING_COLUMNS) == 0:
                    print(f'\nMLR HAS NOT BEEN RUN, WINNING COLUMNS LIST IS EMPTY. NO COLUMNS DELETED.\n')
                    break

                _cols = len(self.TRAIN_RESULTS)

                print()
                post_mlr_select = dmp.DictMenuPrint(POST_MLR_MENU, disp_len=140, disallowed=disallowed).select(f'')

                # BUILD SORT_MASK FROM argsort(TRAIN_RESULTS) TO SORT WC & TR INTO WINNING ORDER (SEE sort_mask_builder)
                # BUILD KEEP_MASK AFTER TRAIN_RESULTS IS SORTED TO INDICATE POSITIONS IN TR & WC TO BE KEPT/CHOPPED
                # APPLY SORT_MASK FIRST, THEN KEEP_MASK, TO WC AND TR

                if post_mlr_select == 'R':
                    print(f'\nTRAIN RESULTS:')
                    print(self.TRAIN_RESULTS)
                    print()

                if post_mlr_select == 'D':
                    tred.train_results_excel_dump(self.TRAIN_RESULTS, 'TRAIN_RESULTS')

                if post_mlr_select == 'W':    # 'w': 'keep all columns used in MLR'
                    # NO SORTING OF self.TRAIN_RESULTS
                    # NO MASKING OF self.WINNING_COLUMNS

                    __ = vui.validate_user_str(f'User selected to keep all winning columns, '
                                               f'Accept(a) Abort(b) > ', 'AB')

                    if __ == 'A':
                        SORT_MASK = np.arange(0, _cols, dtype=np.int32)
                        KEEP_MASK = np.ones(_cols, dtype=bool)
                        self.context_text_holder += f' Kept all columns.'
                    elif __ == 'B': continue

                if post_mlr_select == 'P':    # 'keep only WINNING COLUMNS below certain p-value'
                    p_value_cutoff = vui.validate_user_float(f'Enter value of maximum p-value (delete all columns greater than) > ', min=1e-99, max=1)

                    if p_value_cutoff < self.TRAIN_RESULTS[('      ', 'p VALUE')].min(): print(f'\nAll columns will be deleted.\n')
                    elif p_value_cutoff >= self.TRAIN_RESULTS[('      ', 'p VALUE')].max(): print(f'\nNo columns will be deleted.\n')
                    else: print(f"\n{np.sum(self.TRAIN_RESULTS[('      ', 'p VALUE')] > p_value_cutoff)} columns of {_cols} will be deleted.\n")

                    __ = vui.validate_user_str(f'User entered delete columns with p value above {p_value_cutoff}. '
                                               f'Accept(a) Abort(b) > ', 'AB')

                    if __ == 'A':
                        # TRAIN_RESULTS INDEX IS COLUMN NAME SO CANT USE IT TO GET ARGSORT. GET ARGSORT SEPARATE.
                        # SORT BY p VALUE ASCENDING
                        SORT_MASK = sort_mask_builder(('      ', 'p VALUE'), key=None, ascending=True)
                        sort_train_results(('      ', 'p VALUE'), SORT_MASK=SORT_MASK, key=None, ascending=True)
                        KEEP_MASK = np.array(self.TRAIN_RESULTS[('      ', 'p VALUE')] <= p_value_cutoff, dtype=bool)
                        self.context_text_holder += f' Kept {np.sum(KEEP_MASK)} columns of {_cols} where p value <= {p_value_cutoff}.'
                        del p_value_cutoff

                    elif __ == 'B': continue

                if post_mlr_select == 'C':      # 'keep only WINNING COLUMNS above certain abs(COEFF)'
                    coeff_cutoff = vui.validate_user_float(f'Enter minimum abs(COEFF) (delete all columns less than) > ', min=1e-99)

                    if coeff_cutoff > self.TRAIN_RESULTS[('      ', 'COEFFS')].abs().max(): print(f'\nAll columns will be deleted.\n')
                    elif coeff_cutoff <= self.TRAIN_RESULTS[('      ', 'COEFFS')].abs().min(): print(f'\nNo columns will be deleted.\n')
                    else: print(f"\n{np.sum(self.TRAIN_RESULTS[('      ', 'COEFFS')].abs() < coeff_cutoff)} columns of {_cols} will be deleted.\n")

                    __ = vui.validate_user_str(f'User entered to keep columns with abs(COEFF) >= {coeff_cutoff}, '
                                               f'Accept(a) Abort(b) > ', 'AB')

                    if __ == 'A':
                        # TRAIN_RESULTS INDEX IS COLUMN NAME SO CANT USE IT TO GET ARGSORT. GET ARGSORT SEPARATE.
                        # SORT BY abs(COEFF) DESCENDING
                        SORT_MASK = sort_mask_builder(('      ', 'COEFFS'), key=abs, ascending=False)
                        sort_train_results(('      ', 'COEFFS'), SORT_MASK=SORT_MASK)
                        KEEP_MASK = np.array(self.TRAIN_RESULTS[('      ', 'COEFFS')].abs() >= coeff_cutoff, dtype=bool)
                        self.context_text_holder += f' Kept {np.sum(KEEP_MASK)} columns of {_cols} where abs(COEFF) >= {coeff_cutoff}.'
                        del coeff_cutoff

                    elif __ == 'B': continue


                if post_mlr_select == 'X':      # 'keep top x from WINNING COLUMNS based on p-value'
                    top_winners = vui.validate_user_int(f'Enter number of top p value winners to keep (of {_cols} columns) > ',
                                                        min=1+int(not self.intcpt_col_idx is None), max=_cols)

                    __ = vui.validate_user_str(f'User entered top {top_winners} of {_cols} columns, Accept(a) Abort(b) > ', 'AB')

                    if __ == 'A':
                        # TRAIN_RESULTS INDEX IS COLUMN NAME SO CANT USE IT TO GET ARGSORT. GET ARGSORT SEPARATE.
                        # SORT BY p VALUE ASCENDING
                        SORT_MASK = sort_mask_builder(('      ', 'p VALUE'), key=None, ascending=True)
                        sort_train_results(('      ', 'p VALUE'), SORT_MASK=SORT_MASK)
                        KEEP_MASK = np.zeros(_cols, dtype=bool)
                        KEEP_MASK[:top_winners] = True
                        self.context_text_holder += f' Kept top {top_winners} of {_cols} columns based on p value.'
                        del top_winners

                    elif __ == 'B': continue

                if post_mlr_select == 'Y':  # 'keep top x from WINNING COLUMNS based on abs(COEFF)'
                    top_winners = vui.validate_user_int(f'Enter number of top abs(COEFF) winners to keep (of {_cols} columns) > ',
                                                        min=1+int(not self.intcpt_col_idx is None), max=_cols)

                    __ = vui.validate_user_str(f'User entered {top_winners} of {_cols}, Accept(a) Abort(b) > ', 'AB')

                    if __ == 'A':
                        # TRAIN_RESULTS INDEX IS COLUMN NAME SO CANT USE IT TO GET ARGSORT. GET ARGSORT SEPARATE THEN sort_values().
                        # SORT BY abs(COEFF) DESCENDING
                        SORT_MASK = sort_mask_builder(('      ', 'COEFFS'), key=abs, ascending=False)
                        sort_train_results(('      ', 'COEFFS'), SORT_MASK=SORT_MASK)
                        KEEP_MASK = np.zeros(_cols, dtype=bool)
                        KEEP_MASK[:top_winners] = True
                        self.context_text_holder += f' Kept only top {top_winners} of {_cols} winning columns based on abs(COEFF).'
                        print(f'\n *** TOP {top_winners} ABS(COEFF) CHOP COMPLETE ***\n')

                    elif __ == 'B': continue


                # MLR TRAIN_RESULTS HAS COME OUT OF CoreRunCode WITH THE RESULTS FOR ALL COLUMNS SELECTED TO BE USED,
                # SORTED BY P VALUE ASCENDING.
                # THE STEPS ABOVE SORT TRAIN_RESULTS BASED ON USER PICK "TOP X" OR "p VALUE <" OR "abs(COEFF) >" AND CREATE
                # A SORT MASK TO BE APPLIED TO WINNING_COLUMNS, AND ALSO CREATES A KEEP MASK TO APPLY TO TR & WC.
                # AFTER MASKS ARE APPLIED TO WC AND TR (KEEP ONLY FOR TR), WINNING_COLUMNS THEN SORTS/CHOPS SWNL TO THE
                # FINAL WINNING_COLUMNS, IF USER SELECTED TO DO SO

                if post_mlr_select in 'WPCXY':
                    # 'w': 'keep all columns used in MLR'
                    # 'p': 'keep only WINNING COLUMNS below certain p-value'
                    # 'c': 'keep only WINNING COLUMNS above certain abs(COEFF)'
                    # 'x': 'keep top x from WINNING COLUMNS based on p-value'
                    # 'y': 'keep top x from WINNING COLUMNS based on abs(COEFF)'


                    self.WINNING_COLUMNS = self.WINNING_COLUMNS[SORT_MASK]

                    if (not self.intcpt_col_idx is None) and \
                            KEEP_MASK[self.WINNING_COLUMNS==self.intcpt_col_idx][0] == False:
                        __ = vui.validate_user_str(f'\nIntercept is not in WINNING COLUMNS and will be deleted. '
                                                   f'Allow? (y/n) (n keeps intercept) > ', 'YN')
                        if __ == 'Y': self.intcpt_col_idx = None
                        elif __ == 'N':
                            # INTERCEPT MUST BE FIRST, RETURNED FROM MICoreRunCode FIRST
                            KEEP_MASK[0] = True


                    self.WINNING_COLUMNS = self.WINNING_COLUMNS[KEEP_MASK]
                    self.TRAIN_RESULTS = self.TRAIN_RESULTS[KEEP_MASK]
                    del SORT_MASK, KEEP_MASK

                    # IF WINNING_COLUMNS CURRENTLY REPRESENTS THE CURRENT STATE OF DATA, BYPASS COLUMN PULL CODE
                    if not np.array_equiv(range(len(self.WORKING_SUPOBJS[0][0])), self.WINNING_COLUMNS):

                        # 3-23-22, CANT DELETE COLUMNS IN KEEP (see delete_columns()) SO NOTING IN "CONTEXT"
                        # THE COLUMNS THAT FAILED MLR
                        ACTV_HDR = self.WORKING_SUPOBJS[0][msod.QUICK_POSN_DICT()["HEADER"]]
                        for col_idx in range(_cols):  # _cols IS COLUMNS RETURNED FROM MLRegRun WHICH IS ALL COLUMNS IN DATA
                            if col_idx not in self.WINNING_COLUMNS:
                                self.CONTEXT_UPDATE_HOLDER.append(
                                                f'Deleted DATA - {ACTV_HDR[col_idx]} for failing ML Regression.')
                        del ACTV_HDR

                        # CHOP WORKING_SUPOBJS TO WINNING COLUMNS
                        self.WORKING_SUPOBJS[0] = self.WORKING_SUPOBJS[0][..., self.WINNING_COLUMNS]

                        # USE GYMNASTICS TO CHOP WORKING DATA, TRAIN DATA, DEV DATA, AND TEST DATA TO WINNING COLUMNS
                        NAMES = ('WORKING_DATA', 'TRAIN_DATA', 'DEV_DATA', 'TEST_DATA')
                        DATA_OBJS = (self.SUPER_WORKING_NUMPY_LIST, self.TRAIN_SWNL, self.DEV_SWNL, self.TEST_SWNL)

                        for idx, (name, DATA_OBJ) in enumerate(zip(NAMES, DATA_OBJS)):

                            # PASS OBJECTS THAT ARE EMPTY, WOULD EXCEPT WHEN TRYING TO INDEX INTO IT
                            if np.array_equiv(DATA_OBJ, []): continue

                            WinningSelectorClass = mlo.MLObject(DATA_OBJ[0],
                                                                self.data_run_orientation,
                                                                name,
                                                                return_orientation='AS_GIVEN',
                                                                return_format='AS_GIVEN',
                                                                bypass_validation=self.bypass_validation,
                                                                calling_module=self.this_module,
                                                                calling_fxn=fxn)

                            WINNING_COLUMNS_HOLDER = \
                                WinningSelectorClass.return_columns(self.WINNING_COLUMNS,
                                                                    return_orientation='AS_GIVEN',
                                                                    return_format='AS_GIVEN')

                            # RESET TRAIN/DEV/TEST OBJECTS TO FORCE REFILL IN MLRegressionRun
                            if idx==0: self.SUPER_WORKING_NUMPY_LIST[0] = WINNING_COLUMNS_HOLDER
                            if idx==1: self.TRAIN_SWNL[0] = WINNING_COLUMNS_HOLDER
                            if idx==2: self.DEV_SWNL[0] = WINNING_COLUMNS_HOLDER
                            if idx==3: self.TEST_SWNL[0] = WINNING_COLUMNS_HOLDER

                        del WinningSelectorClass, WINNING_COLUMNS_HOLDER

                    self.context_text_holder += f' Retained top {len(self.WINNING_COLUMNS)} ML winning columns.'

                    # RESET BACK TO DEFAULT, DONT ALLOW COLUMN CHOPS AGAIN WITH SAME WINNING_COLUMNS
                    self.WINNING_COLUMNS = np.arange(0, len(self.WORKING_SUPOBJS[0][0]), dtype=np.int32)
                    if not self.intcpt_col_idx is None: self.intcpt_col_idx = 0

                    print(f'\n*** DELETE OF NON-WINNING COLUMNS FROM DATA SUCCESSFUL ***\n')


                if post_mlr_select == 'A':              # 'accept and continue(a)'

                    self.WORKING_CONTEXT.append(self.context_text_holder.strip())

                    self.WORKING_CONTEXT += self.CONTEXT_UPDATE_HOLDER

                    self.CONTEXT_UPDATE_HOLDER = []

                    break

            del sort_train_results, sort_mask_builder


# UNIQUE ###############################################################################################################
# NONE YET










if __name__ == '__main__':
    from MLObjects.TestObjectCreators.SXNL import CreateSXNL as csxnl

    DATA = pd.read_csv(r'C:\Users\Bill\Documents\WORK STUFF\RESUME\1 - OTHER\SRP\beer_reviews.csv',
                      nrows=10000,
                      header=0).dropna(axis=0)

    DATA = DATA[DATA.keys()[[3,4,5,7,8,9,11]]]   # ,7


    TARGET = DATA['review_overall']
    TARGET_HEADER = [['review_overall']]
    TARGET = TARGET.to_numpy().reshape((1,-1))

    DATA = DATA.drop(columns=['review_overall'])

    RAW_DATA = DATA.copy()
    RAW_DATA_HEADER = np.fromiter(RAW_DATA.keys(), dtype='<U50').reshape((1,-1))
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
    data_given_orientation = SXNLClass.data_current_orientation
    target_given_orientation = SXNLClass.target_current_orientation
    refvecs_given_orientation = SXNLClass.refvecs_current_orientation

    SXNLClass.expand_data(expand_as_sparse_dict={'P':True,'A':False}[vui.validate_user_str(f'\nExpand as sparse dict(p) or array(a) > ', 'AP')],
                          auto_drop_rightmost_column=True)
    SWNL = SXNLClass.SXNL
    WORKING_SUPOBJS = SXNLClass.SXNL_SUPPORT_OBJECTS

    # DELETE AN ARBITRARY BIN COLUMN TO REMOVE MULTICOLINEARITY
    # del_idx = 20
    # SWNL[0] = np.delete(SWNL[0], del_idx, axis=0)
    # WORKING_SUPOBJS[0] = np.delete(WORKING_SUPOBJS[0], del_idx, axis=1)


    del SXNLClass

    try:
        DumClass = mlo.MLObject(
            np.insert(SWNL[0], len(SWNL[0]) if data_given_orientation=="COLUMN" else len(SWNL[0][0]), 1, axis=0 if data_given_orientation=="COLUMN" else 1),
                      data_given_orientation, name="SWNL[0]", return_orientation='AS_GIVEN', return_format='AS_GIVEN',
            bypass_validation=True, calling_module="MLRegressionRun", calling_fxn='test')
        XTX_INV = DumClass.return_XTX_INV(return_format="ARRAY")
        del DumClass, XTX_INV
        print(f'\n*** SWNL[0] INVERTED ***\n')
    except:
        raise Exception(f'*** SWNL[0] DOES NOT INVERT ***')


    WORKING_CONTEXT = []
    WORKING_KEEP = RAW_SUPOBJS[0][msod.QUICK_POSN_DICT()["HEADER"]]


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
    mlr_batch_size = 1000 #len(SWNL[0][0])
    MLR_OUTPUT_VECTOR = []

    SUPER_RAW_NUMPY_LIST, RAW_SUPOBJS, SUPER_WORKING_NUMPY_LIST, WORKING_SUPOBJS, WORKING_CONTEXT, WORKING_KEEP, \
    split_method, LABEL_RULES, number_of_labels, event_value, negative_value, conv_kill, pct_change, conv_end_method, \
    mlr_rglztn_type, mlr_rglztn_fctr, mlr_batch_method, mlr_batch_size, MLR_OUTPUT_VECTOR = \
    MLRegressionConfigRun(standard_config, mlr_config, SRNL, RAW_SUPOBJS, SWNL, WORKING_SUPOBJS, data_given_orientation,
        target_given_orientation, refvecs_given_orientation, WORKING_CONTEXT, WORKING_KEEP, split_method, LABEL_RULES,
        number_of_labels, event_value, negative_value, mlr_rglztn_type, mlr_rglztn_fctr, mlr_batch_method, mlr_batch_size,
        MLR_OUTPUT_VECTOR).configrun()



    print(f'\nFINAL RETURN OF WORKING DATA LOOKS LIKE:')
    ioap.IdentifyObjectAndPrint(SUPER_WORKING_NUMPY_LIST[0], 'DATA', 'MLRegressionConfigRun', 20, 10, 0, 0)



    print(f'\nFINAL RETURN OF WORKING DATA SUPPORT OBJECT LOOKS LIKE:')
    print(WORKING_SUPOBJS[0])


















