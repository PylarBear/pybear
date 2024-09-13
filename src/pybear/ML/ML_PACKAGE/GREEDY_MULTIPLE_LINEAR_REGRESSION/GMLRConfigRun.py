import sys, inspect, warnings
from copy import deepcopy
import numpy as np, pandas as pd
from data_validation import validate_user_input as vui
from general_data_ops import get_shape as gs
from ML_PACKAGE import MLConfigRunTemplate as mlcrt
from ML_PACKAGE.GREEDY_MULTIPLE_LINEAR_REGRESSION import GMLRConfig as gmlrc, GMLRRun as gmlrr
from ML_PACKAGE.GENERIC_PRINT import DictMenuPrint as dmp, train_results_excel_dump as tred
from MLObjects import MLObject as mlo
from MLObjects.ObjectOrienter import MLObjectOrienter as mloo
from MLObjects.SupportObjects import master_support_object_dict as msod



# INHERITS #############################################################################################################
# dataclass_mlo_loader()             Loads an instance of MLObject for DATA.
# intercept_manager()                Locate columns of constants in DATA & handle. As of 11/15/22 only for MLR, MI, and GMLR.
# insert_intercept()                 Insert a column of ones in the 0 index of DATA.
# delete_columns()                    Delete a column from DATA and respective holder objects.
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
# intbin_remove_verbage()             verbage put into CONTEXT when int and bin are removed from DATA
# intbin_reappend_verbage()           verbage put into CONTEXT when int and bin are re-appended to DATA


#CALLED BY ML
class GMLRConfigRun(mlcrt.MLConfigRunTemplate):

    def __init__(self, standard_config, gmlr_config, SUPER_RAW_NUMPY_LIST, RAW_SUPOBJS, SUPER_WORKING_NUMPY_LIST, WORKING_SUPOBJS,
                 data_given_orientation, target_given_orientation, refvecs_given_orientation, WORKING_CONTEXT, WORKING_KEEP,
                 split_method, LABEL_RULES, number_of_labels, event_value, negative_value, gmlr_conv_kill, gmlr_pct_change,
                 gmlr_conv_end_method, gmlr_rglztn_type, gmlr_rglztn_fctr,
                 # UNIQUE TO GMLR
                 gmlr_batch_method, gmlr_batch_size, gmlr_type, gmlr_score_method, gmlr_float_only, gmlr_max_columns,
                 gmlr_bypass_agg, GMLR_OUTPUT_VECTOR):

        data_run_orientation = 'ROW'
        data_run_format = 'AS_GIVEN'
        target_run_orientation = 'ROW'
        target_run_format = 'ARRAY'
        refvecs_run_orientation = 'COLUMN'
        refvecs_run_format = 'ARRAY'


        super().__init__(standard_config, gmlr_config, SUPER_RAW_NUMPY_LIST, RAW_SUPOBJS, SUPER_WORKING_NUMPY_LIST,
            WORKING_SUPOBJS, data_given_orientation, target_given_orientation, refvecs_given_orientation, data_run_orientation,
            target_run_orientation, refvecs_run_orientation, data_run_format, target_run_format, refvecs_run_format,
            WORKING_CONTEXT, WORKING_KEEP, split_method, LABEL_RULES, number_of_labels, event_value, negative_value,
            gmlr_conv_kill, gmlr_pct_change, gmlr_conv_end_method, gmlr_rglztn_type, gmlr_rglztn_fctr, __name__)

        fxn = '__init__'

        # IF INTS/BIN IN MOD_DTYPES, FORCE CREATE OF BACKUPS REGARDLESS OF WHAT USER CHOSE AT BACKUP PROMPT IN super() TO
        # ENABLE FLOAT RESTORE FUNCTIONALITY
        if True in map(lambda x: x in self.WORKING_SUPOBJS[0][msod.QUICK_POSN_DICT()['MODIFIEDDATATYPES']], ['INT','BIN']):
            self.SUPER_WORKING_NUMPY_LIST_BACKUP = [_.copy() if isinstance(_, np.ndarray) else deepcopy(_) for _ in SUPER_WORKING_NUMPY_LIST]
            self.WORKING_SUPOBJS_BACKUP = WORKING_SUPOBJS.copy()


        self.batch_method = gmlr_batch_method
        self.batch_size = gmlr_batch_size
        self.type = gmlr_type
        self.gmlr_score_method = gmlr_score_method
        self.float_only = gmlr_float_only
        self.max_columns = gmlr_max_columns
        self.bypass_agg = gmlr_bypass_agg
        self.OUTPUT_VECTOR = GMLR_OUTPUT_VECTOR     # 11/13/22 JUST PASSES THRU




        ########################################################################################################################
        # ORIENT/FORMAT TARGET & REFVEC OBJECTS ################################################################################

        # SRNL IS IN super(), SWNL HANDLED IN CHILDREN FOR EASIER HANDLING OF DIFFERENT "RETURN_OBJECTS" FOR DIFFERENT PACKAGES

        # TARGET & REFVECS CAN BE SET HERE ONCE AND FOR ALL. BECAUSE OF THE COLUMN CHOP AND INTERCEPT OPTIONS IN & AROUND Config,
        # INITIALLY SET DATA, OTHER DATA OBJECTS, AND data_run_format/orientation HERE.
        # NO POINT IN MAKING DATA_TRANSPOSE OR XTX BECAUSE DATA WILL ALWAYS BE CHANGING IN CoreRun

        # BREAK CONNECTIONS OF SUPER_WORKING_NUMPY_LIST WITH SUPER_WORKING_NUMPY_LIST IN OUTER SCOPE. DO NOT CHANGE THIS,
        # SEE NOTES IN MLRegressionConfigRun.
        self.SUPER_WORKING_NUMPY_LIST = [deepcopy(_) if isinstance(_, dict) else _.copy() for _ in SUPER_WORKING_NUMPY_LIST]

        data_rows, data_cols = gs.get_shape('DATA', self.SUPER_WORKING_NUMPY_LIST[0], data_given_orientation)
        target_rows, target_cols = gs.get_shape('TARGET', self.SUPER_WORKING_NUMPY_LIST[1], target_given_orientation)

        if data_cols == 0: raise Exception(f'{self.this_module}.{fxn}() >>> DATA IN HAS NO COLUMNS')
        if not data_rows == target_rows: raise Exception(f'{self.this_module}.{fxn}() >>>  TARGET ROWS != DATA ROWS. TERMINATE.')

        del data_rows, data_cols, target_rows, target_cols


        SWNLOrienterClass = mloo.MLObjectOrienter(
                                                    DATA=self.SUPER_WORKING_NUMPY_LIST[0],
                                                    data_given_orientation=self.data_given_orientation,
                                                    data_return_orientation=self.data_run_orientation,
                                                    data_return_format=self.data_run_format,

                                                    DATA_TRANSPOSE=None,
                                                    data_transpose_given_orientation=None,
                                                    data_transpose_return_orientation=None,
                                                    data_transpose_return_format=None,

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
                                                    xtx_return_format=None,

                                                    RETURN_OBJECTS=['DATA', 'TARGET', 'TARGET_TRANSPOSE', 'TARGET_AS_LIST'],

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

        # END ORIENT SWNL & MAKE OTHER OBJECTS #################################################################################
        ########################################################################################################################


        self.WINNING_COLUMNS = []
        self.COEFFS = []
        self.TRAIN_RESULTS = pd.DataFrame({})

        self.context_text_holder = ''    # CREATE A STRING THAT RECORDS ALL OPERATIONS HERE, THEN IS APPENDED TO CONTEXT
        self.CONTEXT_UPDATE_HOLDER = []      # CREATE THIS TO HOLD CHOPS, THEN PUT INTO CONTEXT AFTER context_text_holder

        # BEAR HASHED THIS OUT 7/1/23, THINKING THIS IS REDUNDANT WITH THE ONE AFTER Config(), AND ALL THE OTHER MODULES W
        # ML_intercept_manager() DONT HAVE THIS.
        # OK TO DELETE IF EVERYTHING IS WORKING FINE.
        # self.intercept_manager()

    # END init #############################################################################################################
    ########################################################################################################################
    ########################################################################################################################



    # INHERITS #############################################################################################################
    # dataclass_mlo_loader()             Loads an instance of MLObject for DATA.
    # intercept_manager()                Locate columns of constants in DATA & handle. As of 11/15/22 only for MLR, MI, and GMLR.
    # run_module_input_tuple()           tuple of base params that pass into run_module for all ML packages
    # base_post_run_options_module()     holds post run options applied to all ML packages
    # return_fxn_base()                  values returned from all ML packages
    # configrun()                        runs config_module() & run_module()


    # OVERWRITES #######################################################################################################
    def config_module(self):
        # config_module()                    gets configuration source, returns configuration parameters for particular ML package

        self.conv_kill, self.pct_change, self.conv_end_method, self.rglztn_type, self.rglztn_fctr, self.batch_method, \
        self.batch_size, self.type, self.gmlr_score_method, self.float_only, self.max_columns, self.bypass_agg, \
        self.intcpt_col_idx = \
            gmlrc.GMLRConfig(self.standard_config, self.sub_config, self.SUPER_WORKING_NUMPY_LIST, self.WORKING_SUPOBJS,
                self.data_run_orientation, self.conv_kill, self.pct_change, self.conv_end_method, self.rglztn_type,
                self.rglztn_fctr, self.batch_method, self.batch_size, self.type, self.gmlr_score_method, self.float_only,
                self.max_columns, self.bypass_agg, self.intcpt_col_idx, self.bypass_validation).config()

        self.intercept_manager()


    def run_module(self):
        # run_module()                       returns run module for particular ML package

        while True:  # TO ALLOW ABORT

            if self.float_only:
                __ = vui.validate_user_str(f'\nGMLR is currently set for FLOAT ONLY.  Proceed(p), change(c), abort(a)? > ', 'PCA')
                if __ == 'P': pass
                elif __ == 'C':
                    self.float_only = {'T':True,'F':False}[vui.validate_user_str(f'Set GMLR float only: True(t) False(f) > ', 'TF')]
                elif __ == 'A': break

                print(f'\nDeleting INT and BIN columns from DATA (if any) and preserving intercept...')
                W_MOD_DS = self.WORKING_SUPOBJS[0][msod.QUICK_POSN_DICT()["MODIFIEDDATATYPES"]]
                BININT_TO_DELETE = []
                start_len = len(W_MOD_DS)
                for col_idx in range(len(W_MOD_DS)-1, -1, -1):
                    if col_idx == self.intcpt_col_idx: continue
                    # FIND OUT IF IS INT/BIN
                    if W_MOD_DS[col_idx] in ['INT','BIN']: BININT_TO_DELETE.append(int(col_idx))

                self.delete_columns(BININT_TO_DELETE, update_context=False)  # CONTEXT UPDATE IS HANDLED EXTERNALLY

                if not self.intcpt_col_idx is None: self.intcpt_col_idx -= sum(np.array(BININT_TO_DELETE < self.intcpt_col_idx))

                self.context_text_holder += ' ' + self.intbin_remove_verbage()
                print(f'\n*** DELETE OF INT AND BIN COLUMNS COMPLETE (DELETED {start_len-len(self.WORKING_SUPOBJS[0][0])} COLUMNS OF {start_len}) ***')

                del W_MOD_DS, BININT_TO_DELETE, start_len

            self.TRAIN_SWNL, self.DEV_SWNL, self.TEST_SWNL, self.PERTURBER_RESULTS, self.WINNING_COLUMNS, self.COEFFS, self.TRAIN_RESULTS = \
                gmlrr.GMLRRun(*self.run_module_input_tuple(), self.TARGET_TRANSPOSE, self.TARGET_AS_LIST, self.batch_method,
                              self.batch_size, self.type, self.gmlr_score_method, self.float_only, self.max_columns,
                              self.bypass_agg, self.intcpt_col_idx, self.TRAIN_RESULTS).run()

            self.context_text_holder += f' Ran GMLR.'
            break


    def return_fxn(self):

        # return_fxn()                       returns user-specified output, in addition to return_fxn_base()
        return *self.return_fxn_base()[:-5], self.batch_method, self.batch_size, self.type, self.gmlr_score_method, \
                self.float_only, self.max_columns, self.bypass_agg, self.OUTPUT_VECTOR


    def sub_post_run_cmds(self):
    # package-specific options available to modify WORKING_DATA after run
        return {'k':'operations for GMLR WINNING COLUMNS'}


    def intbin_remove_verbage(self):
        # verbage put into CONTEXT when int and bin are removed from DATA
        return f'Removed INT and BIN columns from DATA prior to GMLR.'


    def intbin_reappend_verbage(self):
        # verbage put into CONTEXT when int and bin are re-appended to DATA
        return f'Restored INT and BIN columns to DATA after GMLR.'


    def sub_post_run_options_module(self):
    # holds post run options unique to particular ML package

        fxn = inspect.stack()[0][3]

        if self.post_configrun_select == 'K':

            POST_GMLR_MENU = {
                              'w': 'keep only WINNING COLUMNS from GMLR',
                              'p': 'keep only WINNING COLUMNS below certain p-value',
                              'x': 'select only top x from WINNING COLUMNS',
                              'y': 'keep top x from WINNING COLUMNS based on p-value',
                              'r': 'print TRAIN RESULTS',
                              'd': 'dump TRAIN RESULTS to file',
                              'a': 'accept and continue',
                              }

            # self.WINNING_COLUMNS IS ALREADY SORTED BY WINNING ORDER (GMLR SCORE DESCENDING, COULD BE R2, ADJ R2, F,
            # --- THE ORDER THAT TRAIN_RESULTS IS NOW IN) BUT DATA IS NOT SORTED

            def sort_mask_builder(column, key=None, ascending=True):
                # IF INTERCEPT IS IN WINNING_DATA, ALWAYS PUTS IT AT THE TOP OF TRAIN_RESULTS AND COL IDX 0 IN DATA
                if key==None: SORT_MASK = np.argsort(self.TRAIN_RESULTS[column].to_numpy())
                elif key==abs: SORT_MASK = np.argsort(self.TRAIN_RESULTS[column].abs().to_numpy())
                if ascending==False: SORT_MASK = np.flip(SORT_MASK)
                if not self.intcpt_col_idx is None:
                    # TAKE OUT INTERCEPT AND PUT IN FIRST POSITION
                    SORT_MASK = np.insert(SORT_MASK[SORT_MASK != 0], 0, 0, axis=0)

                return SORT_MASK


            def sort_train_results(column, SORT_MASK=None, key=None, ascending=True):
                # KEEP OVERALL R, R2, ADJ R2, F AT THE TOP
                OVERALL_HOLDER = self.TRAIN_RESULTS.iloc[0, -4:].copy()                     # CAPTURE OVERALL SCORES
                # COMPLICATION TO GET INTERCEPT ALWAYS AT THE TOP
                if SORT_MASK is None: SORT_MASK = sort_mask_builder(column, key, ascending)

                self.TRAIN_RESULTS.iloc[0, -4:] = '-'                      # SEE MLRegressionCoreRunCode fillna('-')
                self.TRAIN_RESULTS = self.TRAIN_RESULTS.iloc[SORT_MASK, :]
                self.TRAIN_RESULTS.iloc[0, -4:] = OVERALL_HOLDER
                del OVERALL_HOLDER, SORT_MASK

            disallowed = ''

            # 6/9/23 IF RIDGE WAS USED, P VALUES WILL ALL BE '-'.  IF SO, DISALLOW FILTERING BY P VALUES. MAYBE COULD USE
            # self.rglztn_fctr, BUT WITH ABILITY TO CHANGE IT ON THE FLY IN GMLRRun AND ITS NOT RETURNED TO HERE, THIS
            # MAY NOT BE ROBUST.  LOOK TO SEE IF '-' IN P VALUE COLUMN, THEN DISALLOW FILTERING.

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if np.array_equiv(self.TRAIN_RESULTS[('FINAL','p VALUE')].to_numpy(), ['-' for _ in range(len(self.TRAIN_RESULTS))]):
                    disallowed += 'PY'

            while True:

                if len(self.WINNING_COLUMNS) == 0:
                    print(f'\nGMLR HAS NOT BEEN RUN, WINNING COLUMNS LIST IS EMPTY.\n')

                _cols = len(self.TRAIN_RESULTS)  # _cols IS NOT ACTUAL # COLUMNS IN DATA! IS NUMBER OF COLS RETURNED FROM GMLRRun

                post_gmlr_select = dmp.DictMenuPrint(POST_GMLR_MENU, disallowed=disallowed, disp_len=140).select('')

                # BUILD SORT_MASK FROM argsort(TRAIN_RESULTS) TO SORT TR & WC INTO WINNING ORDER
                # BUILD KEEP_MASK AFTER TRAIN_RESULTS IS SORTED TO INDICATE POSITIONS IN TR & WC TO BE KEPT/CHOPPED
                # APPLY SORT_MASK FIRST, THEN KEEP_MASK, TO WINNING_COLUMNS. TRAIN_RESULTS IS SORTED IN-PROCESS, SO ONLY
                # APPLY KEEP_MASK AT THE END

                if post_gmlr_select == 'R':
                    print(f'\nTRAIN RESULTS:')
                    print(self.TRAIN_RESULTS)
                    print()

                # TRAIN_RESULTS IS COMING IN FROM run_module() AS DF, ROWS HOLDING INFO ABT A COLUMN IN DATA, SORTED DESC
                # BY SCORE, WHICH COULD BE RSQ, ADJ RSQ, OR F

                if post_gmlr_select == 'D':
                    tred.train_results_excel_dump(self.TRAIN_RESULTS, 'TRAIN_RESULTS')

                if post_gmlr_select == 'W':    # 'keep only WINNING COLUMNS from GMLR'

                    __ = vui.validate_user_str(f'User entered to keep all winning columns from GMLR, '
                                               f'Accept(a) Abort(b) > ', 'AB')

                    if __ == 'A':
                        # BUILD MASK FOR WINNING COLUMNS... POSN IN WINNING_COLUMNS MATCHES ROW POSN IN TRAIN_RESULTS
                        # DONT DONT SORT WC & TR

                        SORT_MASK = np.arange(0, _cols, dtype=np.int32)
                        KEEP_MASK = np.ones(_cols, dtype=bool)
                        self.context_text_holder += f' Kept all GMLR winning columns.'

                    elif __ == 'B': continue

                if post_gmlr_select == 'P':     # 'keep only WINNING COLUMNS below certain p-value(p)'
                    p_value_cutoff = vui.validate_user_float(f'Enter value of maximum p-value (delete all columns greater than) > ', min=1e-99, max=1)

                    if p_value_cutoff < self.TRAIN_RESULTS[('FINAL', 'p VALUE')].min(): print(f'\nAll columns will be deleted.\n')
                    elif p_value_cutoff > self.TRAIN_RESULTS[('FINAL', 'p VALUE')].max(): print(f'\nNo columns will be deleted.\n')
                    else: print(f"\n{np.sum(self.TRAIN_RESULTS[('FINAL', 'p VALUE')] > p_value_cutoff)} columns of {_cols} will be deleted.\n")

                    __ = vui.validate_user_str(f'User entered delete columns with p value above {p_value_cutoff}. '
                                                f'Accept(a) Abort(b) > ', 'AB')

                    if __ == 'A':
                        # TRAIN_RESULTS INDEX IS COLUMN NAME SO CANT USE IT TO GET ARGSORT. GET ARGSORT SEPARATE.
                        # SORT BY p VALUE ASCENDING
                        SORT_MASK = sort_mask_builder(('FINAL', 'p VALUE'), key=None, ascending=True)
                        sort_train_results(('FINAL', 'p VALUE'), SORT_MASK=SORT_MASK)
                        KEEP_MASK = np.array(self.TRAIN_RESULTS[('FINAL', 'p VALUE')] <= p_value_cutoff, dtype=bool)
                        self.context_text_holder += f' Kept {np.sum(KEEP_MASK)} columns of {_cols} where p value <= {p_value_cutoff}.'
                        del p_value_cutoff

                    elif __ == 'B': continue

                if post_gmlr_select == 'X':      # 'select only top x from WINNING COLUMNS(x)'

                    top_winners = vui.validate_user_int(f'Enter number of top GMLR winners to keep (of {_cols}) columns > ',
                                                        min=1 + int(not self.intcpt_col_idx is None), max=_cols)

                    __ = vui.validate_user_str(f'User entered top {top_winners} of {_cols} columns, Accept(a) Abort(b) > ', 'AB')
                    if __ == 'A':
                        # BUILD MASK FOR WINNING COLUMNS BASED ON top_winners... POSN IN WINNING_COLUMNS MATCHES ROW POSN IN TRAIN_RESULTS
                        # DONT SORT SORT WC & TR

                        SORT_MASK = np.arange(0, _cols, dtype=np.int32)
                        KEEP_MASK = np.zeros(_cols, dtype=bool)
                        KEEP_MASK[:top_winners] = True

                        self.context_text_holder += f' Kept only top {top_winners} of {_cols} GMLR winning columns.'

                    elif __ == 'B': continue

                if post_gmlr_select == 'Y':     # 'y': 'keep top x from WINNING COLUMNS based on p-value'
                    top_winners = vui.validate_user_int(f'Enter number of top p value winners to keep (of {_cols} columns) > ',
                                                        min=1 + int(not self.intcpt_col_idx is None), max=_cols)

                    __ = vui.validate_user_str(f'User entered top {top_winners} of {_cols} columns, Accept(a) Abort(b) > ', 'ARB')

                    if __ == 'A':
                        # TRAIN_RESULTS INDEX IS COLUMN NAME SO CANT USE IT TO GET ARGSORT. GET ARGSORT SEPARATE THEN sort_values().
                        # SORT BY p VALUE ASCENDING
                        SORT_MASK = sort_mask_builder(('FINAL', 'p VALUE'), key=None, ascending=True)
                        sort_train_results(('FINAL', 'p VALUE'), SORT_MASK=SORT_MASK)
                        KEEP_MASK = np.zeros(_cols, dtype=bool)
                        KEEP_MASK[:top_winners] = True
                        self.context_text_holder += f' Kept top {top_winners} of {_cols} columns based on p value.'
                        del top_winners

                    elif __ == 'B': continue



                # GMLR TRAIN_RESULTS HAS COME OUT OF CoreRunCode WITH THE RESULTS FOR max_columns COLUMNS,
                # SORTED BY SCORE DESCENDING (SCORE COULD BE R2, ADJ R2 OR F).
                # THE STEPS ABOVE SORT TRAIN_RESULTS BASED ON USER PICK "TOP X" OR "p VALUE <" OR "abs(COEFF) >" AND CREATE
                # A SORT MASK TO BE APPLIED TO WINNING_COLUMNS, AND ALSO CREATES A KEEP MASK TO APPLY TO TR & WC.
                # AFTER MASKS ARE APPLIED TO WC AND TR (KEEP ONLY FOR TR), WINNING_COLUMNS THEN SORTS/CHOPS SWNL TO THE
                # FINAL WINNING_COLUMNS, IF USER SELECTED TO DO SO

                if post_gmlr_select in 'WPXY':
                    # 'w': 'keep only WINNING COLUMNS from GMLR',
                    # 'p': 'keep only WINNING COLUMNS below certain p-value',
                    # 'x': 'select only top x from WINNING COLUMNS',
                    # 'y': 'keep top x from WINNING COLUMNS based on p-value'

                    self.WINNING_COLUMNS = self.WINNING_COLUMNS[SORT_MASK]

                    if (not self.intcpt_col_idx is None) and \
                        (KEEP_MASK[self.WINNING_COLUMNS==self.intcpt_col_idx][0] == False):
                        __ = vui.validate_user_str(f'\nIntercept is not in WINNING COLUMNS and will be deleted. '
                                                   f'Allow? (y/n) > ', 'YN')
                        if __ == 'Y': self.intcpt_col_idx = None
                        elif __ == 'N':
                            # INTERCEPT MUST BE FIRST, RETURNED FROM MICoreRunCode FIRST
                            KEEP_MASK[0] = True

                    self.WINNING_COLUMNS = self.WINNING_COLUMNS[KEEP_MASK]
                    self.TRAIN_RESULTS = self.TRAIN_RESULTS[KEEP_MASK]
                    del KEEP_MASK, SORT_MASK

                    # IF WINNING_COLUMNS CURRENTLY REPRESENTS THE CURRENT STATE OF DATA, BYPASS COLUMN PULL CODE
                    if not np.array_equiv(range(len(self.WORKING_SUPOBJS[0][0])), self.WINNING_COLUMNS):

                        # 4-17-22, CANT DELETE COLUMNS IN KEEP (see delete_columnS()) SO NOTING IN "CONTEXT"
                        # THE COLUMNS THAT FAILED MI
                        ACTV_HDR = self.WORKING_SUPOBJS[0][msod.QUICK_POSN_DICT()["HEADER"]]
                        # MUST USE ACTUAL COLUMNS IN DATA NOT _cols, WHICH IS THE NUMBER OF COLUMNS SELECTED BY MIRun (max_cols)
                        for col_idx in range(gs.get_shape('DATA', self.SUPER_WORKING_NUMPY_LIST[0], self.data_run_orientation)[1]):
                            if col_idx not in self.WINNING_COLUMNS:
                                self.CONTEXT_UPDATE_HOLDER.append(f'Deleted DATA - {ACTV_HDR} for low Mutual Information score.')
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


                            if idx == 0: self.SUPER_WORKING_NUMPY_LIST[0] = WINNING_COLUMNS_HOLDER
                            if idx == 1: self.TRAIN_SWNL[0] = WINNING_COLUMNS_HOLDER
                            if idx == 2: self.DEV_SWNL[0] = WINNING_COLUMNS_HOLDER
                            if idx == 3: self.TEST_SWNL[0] = WINNING_COLUMNS_HOLDER

                        del WinningSelectorClass, WINNING_COLUMNS_HOLDER

                    self.context_text_holder += f' Retained top {len(self.WINNING_COLUMNS)} GMLR winning columns.'

                    # RESET BACK TO DEFAULT, DONT ALLOW COLUMN CHOPS AGAIN WITH SAME WINNING_COLUMNS
                    self.WINNING_COLUMNS = np.arange(0, len(self.WORKING_SUPOBJS[0][0]), dtype=np.int32)
                    if not self.intcpt_col_idx is None: self.intcpt_col_idx = 0
                    # 5/17/23 GMLRCoreRunCode IS FORCING INTERCEPT TO 0 IDX AND THIS MODULE IS PRESERVING IT UPON CHOP

                    print(f'\n*** DELETE OF NON-WINNING COLUMNS FROM DATA SUCCESSFUL ***\n')


                if post_gmlr_select == 'A':
                    # 'accept and continue(a)'

                    self.WORKING_CONTEXT.append(self.context_text_holder.strip())

                    self.WORKING_CONTEXT += self.CONTEXT_UPDATE_HOLDER

                    self.CONTEXT_UPDATE_HOLDER = []


                    # INT/BIN REAPPEND ABSOLUTELY MUST BE DONE AFTER ANY MODIFICATION TO WINNING_COLUMNS (DIRECTLY ABOVE)
                    # 6-9-23 IF float_only WAS FALSE, OR NO INT/BIN IN BACKUP, OR INT/BIN ALREADY RE-APPENDED TO DATA,
                    # DONT ALLOW OPTION TO APPEND WITHHELD INT/BIN

                    mdtype_idx = msod.QUICK_POSN_DICT()["MODIFIEDDATATYPES"]

                    if (self.float_only is False) or True not in map(lambda x: x in self.WORKING_SUPOBJS_BACKUP[0][mdtype_idx], ['INT','BIN']) or \
                        (np.sum(self.WORKING_SUPOBJS[0][mdtype_idx]=='BIN') + np.sum(self.WORKING_SUPOBJS[0][mdtype_idx]=='INT')) > int(not self.intcpt_col_idx is None):
                        # COMPLICATION HERE. IF HAS INTERCEPT IT STAYS IN AND 'BIN' STAYS IN self.WORKING_SUPOBJS[0][mdtype_idx]; DURING WINNER CHOPS,
                        # INTCPT WOULD BE MOVED TO FIRST BUT THAT COULD BE BYPASSED LEAVING INTCPT WHEREVER IT WAS, SO HAVE TO COUNT INTS AND BINS IN
                        # MOD_DTYPES INSTEAD OF LOOKING BY LOCATION. IF NUMBER OF INTS/BINS IN self.WORKING_SUPOBJS[0][mdtype_idx] is > int(HAS INTERCEPT),
                        # THEN INTS/BINS HAVE ALREADY BEEN REAPPENDED
                        pass
                    else:
                        # 'append withheld INTS and BINS to WINNING COLUMNS(f)'
                        if vui.validate_user_str(f'append withheld INT/BIN columns back to DATA (y/n) > ', 'YN') == 'Y':

                            ReceiverClass = mlo.MLObject(self.SUPER_WORKING_NUMPY_LIST[0], self.data_run_orientation,
                                                         name='DATA',
                                                         return_format='AS_GIVEN', return_orientation='AS_GIVEN',
                                                         bypass_validation=self.bypass_validation,
                                                         calling_module=self.this_module,
                                                         calling_fxn=fxn)

                            DonorClass = mlo.MLObject(self.SUPER_WORKING_NUMPY_LIST_BACKUP[0],
                                                      self.data_given_orientation, name='DATA_BACKUP',
                                                      return_format='AS_GIVEN', return_orientation='AS_GIVEN',
                                                      bypass_validation=self.bypass_validation,
                                                      calling_module=self.this_module,
                                                      calling_fxn=fxn)

                            # INSERTED AT THE END OF DATA, DONT HAVE TO WORRY ABOUT CHANGING self.intcpt_col_idx
                            ins_col_idx = gs.get_shape('DATA', self.SUPER_WORKING_NUMPY_LIST[0], self.data_run_orientation)[1]

                            for orig_col_idx in range(gs.get_shape('DATA', self.SUPER_WORKING_NUMPY_LIST_BACKUP[0],
                                                                   self.data_given_orientation)[1]):

                                if self.WORKING_SUPOBJS_BACKUP[0][mdtype_idx][orig_col_idx] in ['INT','BIN']:
                                    # MUST GET ORIGINAL VALUES FOR ORIGINAL OBJECTS FROM BACKUP OBJECTS

                                    INSERT_COLUMN = DonorClass.return_columns([orig_col_idx],
                                                      return_orientation='COLUMN', return_format='ARRAY')

                                    ReceiverClass.insert_column(ins_col_idx, INSERT_COLUMN, 'COLUMN',
                                            HEADER_OR_FULL_SUPOBJ=self.WORKING_SUPOBJS[0],
                                            SUPOBJ_INSERT=self.WORKING_SUPOBJS_BACKUP[0][:, orig_col_idx])

                                    self.WORKING_SUPOBJS[0] = ReceiverClass.HEADER_OR_FULL_SUPOBJ

                                    ins_col_idx += 1

                            del ins_col_idx

                            self.SUPER_WORKING_NUMPY_LIST[0] = ReceiverClass.OBJECT

                            self.CONTEXT_UPDATE_HOLDER.append(self.intbin_reappend_verbage())
                            self.float_only = False  # SO IT DOESNT TRIGGER RE-DELETION WHEN RE-ENTERING MIRun

                            del ReceiverClass, DonorClass

                            print(f'\n*** INT/BIN COLUMNS HAVE BEEN SUCCESSFULLY RE-INTRODUCED INTO WORKING DATA ***\n')

                    del mdtype_idx

                    break

            del sort_train_results, sort_mask_builder











if __name__ == '__main__':

    from MLObjects.TestObjectCreators.SXNL import CreateSXNL as csxnl
    from ML_PACKAGE.GREEDY_MULTIPLE_LINEAR_REGRESSION import gmlr_default_config_params as gdcp

    _format = 'ARRAY'
    _orient = 'COLUMN'

    SXNLClass = csxnl.CreateSXNL(
                                    rows=100,
                                    bypass_validation=True,
                                    ####################################################################################
                                    # DATA #############################################################################
                                    data_return_format=_format,
                                    data_return_orientation=_orient,
                                    DATA_OBJECT=None,
                                    DATA_OBJECT_HEADER=None,
                                    DATA_FULL_SUPOBJ_OR_SINGLE_MDTYPES=None,
                                    data_override_sup_obj=False,
                                    # CREATE FROM GIVEN ONLY ###############################################
                                    data_given_orientation=None,
                                    # END CREATE FROM GIVEN ONLY #############################################
                                    # CREATE FROM SCRATCH_ONLY ################################
                                    data_columns=10,
                                    DATA_BUILD_FROM_MOD_DTYPES=['FLOAT', 'STR'],
                                    DATA_NUMBER_OF_CATEGORIES=5,
                                    DATA_MIN_VALUES=-10,
                                    DATA_MAX_VALUES=10,
                                    DATA_SPARSITIES=0,
                                    DATA_WORD_COUNT=None,
                                    DATA_POOL_SIZE=None,
                                    # END DATA ##############################################################
                                    #########################################################################

                                    #########################################################################
                                    # TARGET ################################################################
                                    target_return_format=_format,
                                    target_return_orientation=_orient,
                                    TARGET_OBJECT=None,
                                    TARGET_OBJECT_HEADER=None,
                                    TARGET_FULL_SUPOBJ_OR_SINGLE_MDTYPES=None,
                                    target_type='FLOAT',  # MUST BE 'BINARY','FLOAT', OR 'SOFTMAX'
                                    target_override_sup_obj=False,
                                    target_given_orientation=None,
                                    # END CORE TARGET_ARGS ########################################################
                                    # FLOAT AND BINARY
                                    target_sparsity=10,
                                    # FLOAT ONLY
                                    target_build_from_mod_dtype='FLOAT',  # COULD BE FLOAT OR INT
                                    target_min_value=-10,
                                    target_max_value=10,
                                    # SOFTMAX ONLY
                                    target_number_of_categories=None,
                                    # END TARGET ##############################################################
                                    ###########################################################################

                                    ###########################################################################
                                    # REFVECS #################################################################
                                    refvecs_return_format=_format,  # IS ALWAYS ARRAY (WAS, CHANGED THIS 4/6/23)
                                    refvecs_return_orientation=_orient,
                                    REFVECS_OBJECT=None,
                                    REFVECS_OBJECT_HEADER=None,
                                    REFVECS_FULL_SUPOBJ_OR_SINGLE_MDTYPES=None,
                                    REFVECS_BUILD_FROM_MOD_DTYPES='STR',
                                    refvecs_override_sup_obj=False,
                                    refvecs_given_orientation=None,
                                    refvecs_columns=5,
                                    REFVECS_NUMBER_OF_CATEGORIES=10,
                                    REFVECS_MIN_VALUES=None,
                                    REFVECS_MAX_VALUES=None,
                                    REFVECS_SPARSITIES=None,
                                    REFVECS_WORD_COUNT=None,
                                    REFVECS_POOL_SIZE=None
                                    # END REFVECS ##############################################################
                                    ############################################################################
    )

    # BUILD SRNL ##################################################################################################################

    SRNL = SXNLClass.SXNL.copy()
    RAW_SUPOBJS = SXNLClass.SXNL_SUPPORT_OBJECTS.copy()
    WORKING_KEEP = SXNLClass.SXNL[1][0].copy()

    # END BUILD SRNL ##################################################################################################################

    # EXPAND SRNL TO SWNL #############################################################################################################
    SXNLClass.expand_data(expand_as_sparse_dict=isinstance(SXNLClass.DATA, dict), auto_drop_rightmost_column=True)
    # END EXPAND SRNL TO SWNL #############################################################################################################


    # BUILD SWNL ##################################################################################################################
    SWNL = SXNLClass.SXNL
    WORKING_SUPOBJS = SXNLClass.SXNL_SUPPORT_OBJECTS
    data_given_orientation = SXNLClass.data_current_orientation
    target_given_orientation = SXNLClass.target_current_orientation
    refvecs_given_orientation = SXNLClass.refvecs_current_orientation
    del SXNLClass
    # END BUILD SWNL ##################################################################################################################

    standard_config = 'None'
    gmlr_config = 'None'

    gmlr_conv_kill, gmlr_pct_change, gmlr_conv_end_method, gmlr_rglztn_type, gmlr_rglztn_fctr, gmlr_batch_method, \
    gmlr_batch_size, gmlr_type, gmlr_score_method, gmlr_float_only, gmlr_max_columns, gmlr_bypass_agg, GMLR_OUTPUT_VECTOR = \
        gdcp.gmlr_default_config_params()

    WORKING_CONTEXT = []
    split_method = None
    LABEL_RULES = []
    number_of_labels = 1
    event_value = 1
    negative_value = 0


    GMLRConfigRun(standard_config, gmlr_config, SRNL, RAW_SUPOBJS, SWNL, WORKING_SUPOBJS, data_given_orientation,
                  target_given_orientation, refvecs_given_orientation, WORKING_CONTEXT, WORKING_KEEP,

                  split_method, LABEL_RULES, number_of_labels, event_value, negative_value, gmlr_conv_kill, gmlr_pct_change,
                  gmlr_conv_end_method, gmlr_rglztn_type, gmlr_rglztn_fctr,

                  gmlr_batch_method, gmlr_batch_size, gmlr_type, gmlr_score_method, gmlr_float_only, gmlr_max_columns,
                  gmlr_bypass_agg, GMLR_OUTPUT_VECTOR).configrun()

















