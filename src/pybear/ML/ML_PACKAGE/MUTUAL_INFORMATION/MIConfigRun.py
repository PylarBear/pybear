import sys, inspect
import numpy as np, pandas as pd
from copy import deepcopy
from debug import get_module_name as gmn
import sparse_dict as sd
from general_data_ops import get_shape as gs
from data_validation import validate_user_input as vui, arg_kwarg_validater as akv
from ML_PACKAGE import MLConfigRunTemplate as mlcrt
from ML_PACKAGE.MUTUAL_INFORMATION import MIConfig as mic, MIRun as mir, MICrossEntropyObjects as miceo
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
# float_remove_verbage()             verbage put into CONTEXT when floats are removed from DATA
# float_reappend_verbage()           verbage put into CONTEXT when floats are re-appended to DATA



#CALLED BY ML
class MIConfigRun(mlcrt.MLConfigRunTemplate):

    def __init__(self, standard_config, mi_config, SUPER_RAW_NUMPY_LIST, RAW_SUPOBJS, SUPER_WORKING_NUMPY_LIST, WORKING_SUPOBJS,
                 data_given_orientation, target_given_orientation, refvecs_given_orientation, WORKING_CONTEXT, WORKING_KEEP,
                 split_method, LABEL_RULES, number_of_labels, event_value, negative_value,
                 # UNIQUE TO MI
                 mi_batch_method, mi_batch_size, mi_int_or_bin_only, mi_max_columns, mi_bypass_agg, MI_OUTPUT_VECTOR):

        data_run_format = 'AS_GIVEN'
        data_run_orientation = 'COLUMN'
        target_run_format = 'ARRAY'
        target_run_orientation = 'COLUMN'
        refvecs_run_format = 'ARRAY'
        refvecs_run_orientation = 'COLUMN'

        super().__init__(standard_config, mi_config, SUPER_RAW_NUMPY_LIST, RAW_SUPOBJS, SUPER_WORKING_NUMPY_LIST,
            WORKING_SUPOBJS, data_given_orientation, target_given_orientation, refvecs_given_orientation,
            data_run_orientation, target_run_orientation, refvecs_run_orientation, data_run_format, target_run_format,
            refvecs_run_format, WORKING_CONTEXT, WORKING_KEEP, split_method, LABEL_RULES, number_of_labels,
            event_value, negative_value, 'dum_conv_kill', 'dum_pct_change', 'dum_conv_end_method', 'dum_rglztn_type',
            'dum_rglztn_fctr', __name__)

        fxn = '__init__'

        # IF FLOATS IN MOD_DTYPES, FORCE CREATE OF BACKUPS REGARDLESS OF WHAT USER CHOSE AT BACKUP PROMPT IN super() TO
        # ENABLE FLOAT RESTORE FUNCTIONALITY
        if 'FLOAT' in self.WORKING_SUPOBJS[0][msod.QUICK_POSN_DICT()['MODIFIEDDATATYPES']]:
            self.SUPER_WORKING_NUMPY_LIST_BACKUP = [_.copy() if isinstance(_, np.ndarray) else deepcopy(_) for _ in SUPER_WORKING_NUMPY_LIST]
            self.WORKING_SUPOBJS_BACKUP = WORKING_SUPOBJS.copy()

        # BREAK CONNECTIONS OF SUPER_WORKING_NUMPY_LIST WITH SUPER_WORKING_NUMPY_LIST IN OUTER SCOPE. DO NOT CHANGE THIS,
        # SEE NOTES IN MLRegressionConfigRun.
        self.SUPER_WORKING_NUMPY_LIST = [deepcopy(_) if isinstance(_, dict) else _.copy() for _ in SUPER_WORKING_NUMPY_LIST]
        self.batch_method = mi_batch_method
        self.batch_size = mi_batch_size
        self.int_or_bin_only = mi_int_or_bin_only
        self.max_columns = mi_max_columns
        self.bypass_agg = mi_bypass_agg
        self.OUTPUT_VECTOR = MI_OUTPUT_VECTOR       # 11/13/22 JUST PASSES THRU
        self.WINNING_COLUMNS = []

        ########################################################################################################################
        # ORIENT SWNL & MAKE OTHER OBJECTS #####################################################################################

        # SRNL IS IN super(), SWNL HANDLED IN CHILDREN FOR EASIER HANDLING OF DIFFERENT "RETURN_OBJECTS" FOR DIFFERENT PACKAGES

        print(f'\n    BEAR IN MIConfigRun Orienting WORKING DATA & TARGET IN __init__.  Patience...')

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

                                                    TARGET_TRANSPOSE=None,
                                                    target_transpose_given_orientation=None,
                                                    target_transpose_return_orientation=self.target_run_orientation,
                                                    target_transpose_return_format=self.target_run_format,

                                                    TARGET_AS_LIST=None,
                                                    target_as_list_given_orientation=None,
                                                    target_as_list_return_orientation=self.target_run_orientation,

                                                    RETURN_OBJECTS=['DATA', 'TARGET', 'TARGET_TRANSPOSE', 'TARGET_AS_LIST'],

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
        self.TARGET_TRANSPOSE = SWNLOrienterClass.TARGET_TRANSPOSE
        self.TARGET_AS_LIST = SWNLOrienterClass.TARGET_AS_LIST
        self.target_run_format = SWNLOrienterClass.target_return_format
        self.target_run_orientation = SWNLOrienterClass.target_return_orientation

        del SWNLOrienterClass

        self.SUPER_WORKING_NUMPY_LIST[2] = RefVecsClass.OBJECT
        self.refvecs_run_format = RefVecsClass.current_format
        self.refvecs_run_orientation = RefVecsClass.current_orientation

        del RefVecsClass

        # END ORIENT SWNL & MAKE OTHER OBJECTS #################################################################################
        ########################################################################################################################


        self.TRAIN_RESULTS = {}

        self.context_text_holder = ''    # CREATE A STRING THAT RECORDS ALL OPERATIONS HERE, THEN IS APPENDED TO CONTEXT
        self.CONTEXT_UPDATE_HOLDER = []     # CREATE THIS TO HOLD CHOPS, THEN PUT INTO CONTEXT AFTER context_text_holder


        TargetUniquesClass = mlo.MLObject(self.SUPER_WORKING_NUMPY_LIST[1], self.target_run_orientation, name='TARGET',
            return_orientation=self.target_run_orientation, return_format='ARRAY', bypass_validation=self.bypass_validation,
            calling_module=self.this_module, calling_fxn=fxn)

        TARGET_UNIQUES = TargetUniquesClass.unique(0).reshape((1,-1))
        del TargetUniquesClass

        YMiceoClass = miceo.MICrossEntropyObjects(self.SUPER_WORKING_NUMPY_LIST[1], UNIQUES=TARGET_UNIQUES,
                                                  return_as='ARRAY', bypass_validation=None)

        self.Y_OCCUR_HOLDER = YMiceoClass.OCCURRENCES
        self.Y_SUM_HOLDER = YMiceoClass.SUMS
        self.Y_FREQ_HOLDER = YMiceoClass.FREQ
        del YMiceoClass


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
        self.SUPER_WORKING_NUMPY_LIST, self.WORKING_SUPOBJS, self.batch_method, self.batch_size, self.int_or_bin_only, \
            self.max_columns, self.bypass_agg, self.intcpt_col_idx = \
                mic.MIConfig(self.standard_config, self.sub_config, self.SUPER_WORKING_NUMPY_LIST, self.WORKING_SUPOBJS,
                             self.data_run_orientation, self.batch_method, self.batch_size, self.int_or_bin_only,
                             self.max_columns, self.bypass_agg, self.intcpt_col_idx, self.bypass_validation).config()

        self.intercept_manager()


    def run_module(self):
        # run_module()                       returns run module for particular ML package

        while True:   # TO ALLOW ABORT

            if self.int_or_bin_only:
                __ = vui.validate_user_str(f'\nMI is currently set for INT OR BIN ONLY.  Proceed(p), change(c), abort(a)? > ', 'PCA')
                if __ == 'P': pass
                elif __ == 'C':
                    self.int_or_bin_only = {'T':True,'F':False}[vui.validate_user_str(f'Set MI int or bin only: True(t) False(f) > ', 'TF')]
                elif __ == 'A': break

                print(f'\nDeleting FLOAT columns from DATA (if any) and preserving intercept...')
                W_MOD_DS = self.WORKING_SUPOBJS[0][msod.QUICK_POSN_DICT()["MODIFIEDDATATYPES"]]
                FLOATS_TO_DELETE = []
                start_len = len(W_MOD_DS)
                for col_idx in range(len(W_MOD_DS)-1, -1, -1):
                    if col_idx == self.intcpt_col_idx: continue
                    # FIND OUT IF IS 'FLOAT'
                    if W_MOD_DS[col_idx] == 'FLOAT': FLOATS_TO_DELETE.append(int(col_idx))

                self.delete_columns(FLOATS_TO_DELETE, update_context=False)  # CONTEXT UPDATE IS HANDLED EXTERNALLY BELOW

                if not self.intcpt_col_idx is None: self.intcpt_col_idx -= sum(np.array(FLOATS_TO_DELETE) < self.intcpt_col_idx)

                self.context_text_holder += ' ' + self.float_remove_verbage()
                print(f'\n*** DELETE OF FLOAT COLUMNS COMPLETE (DELETED {start_len-len(self.WORKING_SUPOBJS[0][0])} COLUMNS OF {start_len}) ***')

                del W_MOD_DS, FLOATS_TO_DELETE, start_len

            self.TRAIN_SWNL, self.DEV_SWNL, self.TEST_SWNL, self.PERTURBER_RESULTS, self.WINNING_COLUMNS, self.TRAIN_RESULTS = \
                mir.MIRun(*self.run_module_input_tuple(), self.TARGET_TRANSPOSE, self.TARGET_AS_LIST, self.batch_method,
                    self.batch_size, self.int_or_bin_only, self.max_columns, self.bypass_agg, self.intcpt_col_idx,
                    self.TRAIN_RESULTS, self.Y_OCCUR_HOLDER, self.Y_SUM_HOLDER, self.Y_FREQ_HOLDER).run()

            self.context_text_holder += f' Ran MI.'
            break


    def return_fxn(self):

        # return_fxn()                       returns user-specified output, in addition to return_fxn_base()
        return *self.return_fxn_base()[:-5], self.batch_method, self.batch_size, self.int_or_bin_only, self.max_columns, \
               self.bypass_agg, self.OUTPUT_VECTOR


    def sub_post_run_cmds(self):
        return {'k': 'operations for MI WINNING COLUMNS'}
    # package-specific options available to modify WORKING_DATA after run


    def float_remove_verbage(self):
    # verbage put into CONTEXT when floats are removed from DATA
        return f'Removed FLOAT columns from DATA prior to MI.'


    def float_reappend_verbage(self):
    # verbage put into CONTEXT when floats are re-appended to DATA
        return f'Restored FLOAT columns to DATA after MI.'


    def sub_post_run_options_module(self):
    # holds post run options unique to particular ML package

        fxn = inspect.stack()[0][3]

        if self.post_configrun_select == 'K':

            POST_MI_MENU = {
                            'w': 'keep only WINNING COLUMNS from MI',
                            's': 'keep only WINNING COLUMNS above user-entered score',
                            'x': 'select only top x from WINNING COLUMNS',
                            'r': 'print TRAIN RESULTS',
                            'd': 'dump TRAIN RESULTS to file',
                            'a': 'accept and continue'
            }

            # self.WINNING_COLUMNS IS ALREADY SORTED BY WINNING ORDER (MI SCORE DESCENDING, THE ORDER THAT TRAIN_RESULTS
            # IS NOW IN) BUT DATA IS NOT SORTED

            while True:

                if len(self.WINNING_COLUMNS) == 0:
                    print(f'\nMUTUAL INFORMATION HAS NOT BEEN RUN, WINNING COLUMNS LIST IS EMPTY.\n')
                    break

                _cols = len(self.TRAIN_RESULTS)  # _cols IS NOT ACTUAL # COLUMNS IN DATA! IS NUMBER OF COLS RETURNED FROM MIRun

                print()
                post_mi_select = dmp.DictMenuPrint(POST_MI_MENU, disp_len=140).select('')

                # BUILD WC_KEEP_MASK TO INDICATE POSITIONS IN WINNING_COLUMNS TO BE KEPT/CHOPPED
                # BUILD TR_KEEP_MASK TO INDICATE POSITIONS IN TRAIN_RESULTS TO BE KEPT/CHOPPED

                if post_mi_select == 'R':
                    print(f'\nTRAIN RESULTS:')
                    print(self.TRAIN_RESULTS)
                    print()

                # TRAIN_RESULTS IS COMING IN FROM run_module() AS DF, ROWS HOLDING INFO ABT A COLUMN IN DATA

                if post_mi_select == 'D':
                    tred.train_results_excel_dump(self.TRAIN_RESULTS, 'TRAIN_RESULTS')

                if post_mi_select == 'W':    # 'keep only WINNING COLUMNS from MI'

                    __ = vui.validate_user_str(f'User entered to keep all winning columns from MI, '
                                               f'Accept(a) Abort(b) > ', 'AB')

                    if __ == 'A':
                        # BUILD MASKS FOR WC & TR... POSN IN WINNING_COLUMNS MATCHES ROW POSN IN TRAIN_RESULTS
                        # DONT MAKE TR_SORT_MASK AND DONT SORT self.TRAIN_RESULTS, self.TRAIN_RESULTS IS COMING OUT
                        # OF CoreRun WITH INTERCEPT AT TOP (IF INTERCEPT WAS GIVEN), THEN THE REST DESCENDING BY MI SCORE.
                        # THE SORT_MASK AND sort_values WANT TO CHANGE THAT IF INTERCEPT WASNT BEST (PROBABLY WASNT).
                        TR_KEEP_MASK = np.ones(_cols, dtype=bool)
                        WC_KEEP_MASK = TR_KEEP_MASK.copy()
                        self.context_text_holder += f' Kept all MI winning columns.'

                    elif __ == 'B': continue


                if post_mi_select == 'S':     # 'keep only WINNING COLUMNS above user-entered score(s)'
                    score_cutoff = vui.validate_user_float(f'Enter value of minimum score (delete all columns less than) > ')

                    if score_cutoff > self.TRAIN_RESULTS[('INDIV', 'MI SCORE')].max(): print(f'\nAll columns will be deleted.\n')
                    elif score_cutoff <= self.TRAIN_RESULTS[('INDIV', 'MI SCORE')].min(): print(f'\nNo columns will be deleted.\n')
                    else:
                        chop_count = np.sum(self.TRAIN_RESULTS[('INDIV', 'MI SCORE')] < score_cutoff)
                        print(f'\n{chop_count} columns of {_cols} will be deleted.\n')
                        del chop_count

                    __ = vui.validate_user_str(f'User entered keep columns above MI score of {score_cutoff}, '
                                               f'Accept(a) Abort(b) > ', 'AB')

                    if __ == 'A':
                        # BUILD MASKS FOR WC & TR BASED ON score_cutoff... POSN IN WINNING_COLUMNS MATCHES ROW POSN IN TRAIN_RESULTS
                        # DONT MAKE TR_SORT_MASK AND DONT SORT self.TRAIN_RESULTS, self.TRAIN_RESULTS IS COMING OUT
                        # OF CoreRun WITH INTERCEPT AT TOP (IF INTERCEPT WAS GIVEN). THEN THE REST DESCENDING BY MI SCORE.
                        # THE SORT_MASK AND sort_values WANT TO CHANGE THAT IF INTERCEPT WANT BEST (PROBABLY WASNT).
                        TR_KEEP_MASK = np.array(self.TRAIN_RESULTS[('INDIV', 'MI SCORE')] >= score_cutoff, dtype=bool)
                        WC_KEEP_MASK = TR_KEEP_MASK.copy()

                        self.context_text_holder += f' Kept only {np.sum(TR_KEEP_MASK)} columns of {_cols} where MI score >= {score_cutoff}.'

                    elif __ == 'B': continue

                if post_mi_select == 'X':      # 'select only top x from WINNING COLUMNS(x)'
                    top_winners = vui.validate_user_int(f'Enter number of top MI winners to keep (of {_cols}) columns) > ',
                                                        min=1+int(not self.intcpt_col_idx is None), max=_cols)

                    __ = vui.validate_user_str(f'User entered top {top_winners} of {_cols} columns, Accept(a) Abort(b) > ', 'AB')
                    if __ == 'A':
                        # BUILD MASKS FOR WC & TR BASED ON top_winners... POSN IN WINNING_COLUMNS MATCHES ROW POSN IN TRAIN_RESULTS
                        # DONT MAKE TR_SORT_MASK AND DONT SORT self.TRAIN_RESULTS, self.TRAIN_RESULTS IS COMING OUT
                        # OF CoreRun WITH INTERCEPT AT TOP (IF INTERCEPT WAS GIVEN). THEN THE REST DESCENDING BY MI SCORE.
                        # THE SORT_MASK AND sort_values WANT TO CHANGE THAT IF INTERCEPT WANT BEST (PROBABLY WASNT).
                        TR_KEEP_MASK = np.zeros(_cols, dtype=bool)
                        TR_KEEP_MASK[:top_winners] = True
                        WC_KEEP_MASK = TR_KEEP_MASK.copy()

                        self.context_text_holder += f' Kept only top {top_winners} of {_cols} MI winning columns.'

                    elif __ == 'B': continue

                # MI TRAIN_RESULTS HAS COME OUT OF CoreRunCode WITH THE RESULTS FOR THE TOP max_columns,
                # SORTED BY MI SCORE DESCENDING.
                # THE STEPS ABOVE SORT TRAIN_RESULTS BASED ON USER PICK "TOP X" OR "p VALUE <" OR "abs(COEFF) >" AND CREATE
                # A SORT MASK TO BE APPLIED TO WINNING_COLUMNS, AND ALSO CREATES A KEEP MASK TO APPLY TO TR & WC.
                # AFTER MASKS ARE APPLIED TO WC AND TR (KEEP ONLY FOR TR), WINNING_COLUMNS THEN SORTS/CHOPS SWNL TO THE
                # FINAL WINNING_COLUMNS, IF USER SELECTED TO DO SO

                if post_mi_select in 'WSX':
                    # 'w': 'keep only WINNING COLUMNS from MI'
                    # 's': 'keep only WINNING COLUMNS above user-entered score'
                    # 'x': 'select only top x from WINNING COLUMNS'

                    if (not self.intcpt_col_idx is None) and \
                        (WC_KEEP_MASK[self.WINNING_COLUMNS==self.intcpt_col_idx][0] == False):
                        __ = vui.validate_user_str(f'\nIntercept is not in WINNING COLUMNS and will be deleted. '
                                                   f'Allow? (y/n) (n keeps intercept) > ', 'YN')
                        if __ == 'Y': self.intcpt_col_idx = None
                        elif __ == 'N':
                            # INTERCEPT MUST BE FIRST, RETURNED FROM MICoreRunCode FIRST
                            TR_KEEP_MASK[0] = True
                            WC_KEEP_MASK[0] = True

                    self.WINNING_COLUMNS = self.WINNING_COLUMNS[WC_KEEP_MASK]
                    self.TRAIN_RESULTS = self.TRAIN_RESULTS[TR_KEEP_MASK]
                    del WC_KEEP_MASK, TR_KEEP_MASK

                    # IF WINNING_COLUMNS CURRENTLY REPRESENTS THE CURRENT STATE OF DATA, BYPASS COLUMN PULL CODE
                    if not np.array_equiv(range(len(self.WORKING_SUPOBJS[0][0])), self.WINNING_COLUMNS):

                        # 4-17-22, CANT DELETE COLUMNS IN KEEP (see delete_columns()) SO NOTING IN "CONTEXT"
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

                    self.context_text_holder += f' Retained top {len(self.WINNING_COLUMNS)} MI winning columns.'

                    # RESET BACK TO DEFAULT, DONT ALLOW COLUMN CHOPS AGAIN WITH SAME WINNING_COLUMNS
                    self.WINNING_COLUMNS = np.arange(0, len(self.WORKING_SUPOBJS[0][0]), dtype=np.int32)
                    if not self.intcpt_col_idx is None: self.intcpt_col_idx = 0
                    # 5/17/23 MiCoreRunCode IS FORCING INTERCEPT TO 0 IDX AND THIS MODULE IS PRESERVING IT UPON CHOP

                    print(f'\n*** DELETE OF NON-WINNING COLUMNS FROM DATA SUCCESSFUL ***\n')


                if post_mi_select == 'A':
                    # 'accept and continue(a)'

                    self.WORKING_CONTEXT.append(self.context_text_holder.strip())

                    self.WORKING_CONTEXT += self.CONTEXT_UPDATE_HOLDER

                    self.CONTEXT_UPDATE_HOLDER = []


                    # FLOAT REAPPEND ABSOLUTELY MUST BE DONE AFTER ANY MODIFICATION TO WINNING_COLUMNS (DIRECTLY ABOVE)
                    # 4-20-23 IF bin_or_int_only WAS FALSE, OR NO FLOATS IN BACKUP, OR FLOAT ALREADY RE-APPENDED TO DATA
                    # DONT ALLOW OPTION TO APPEND WITHHELD FLOATS

                    mdtype_idx = msod.QUICK_POSN_DICT()["MODIFIEDDATATYPES"]
                    if (self.int_or_bin_only is False) or 'FLOAT' not in self.WORKING_SUPOBJS_BACKUP[0][mdtype_idx] or \
                        'FLOAT' in self.WORKING_SUPOBJS[0][mdtype_idx]:
                        pass
                    else:
                        if vui.validate_user_str('append withheld FLOAT columns back to DATA (y/n) > ', 'YN') == 'Y':

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

                                if self.WORKING_SUPOBJS_BACKUP[0][mdtype_idx][orig_col_idx] == 'FLOAT':
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

                            self.CONTEXT_UPDATE_HOLDER.append(self.float_reappend_verbage())
                            self.int_or_bin_only = False  # SO IT DOESNT TRIGGER RE-DELETION WHEN RE-ENTERING MIRun

                            del ReceiverClass, DonorClass

                            print(f'\n*** FLOAT COLUMNS HAVE BEEN SUCCESSFULLY RE-INTRODUCED INTO WORKING DATA ***\n')

                    del mdtype_idx

                    break


















if __name__ == '__main__':
    from MLObjects.TestObjectCreators.SXNL import CreateSXNL as csxnl, NewObjsToOldSXNL as notos
    from MLObjects.SupportObjects import NEWSupObjToOLD as nsoto

    DATA = pd.read_csv(r'C:\Users\Bill\Documents\WORK STUFF\RESUME\1 - OTHER\SRP\beer_reviews.csv',
                      nrows=10000,
                      header=0).dropna(axis=0)

    DATA = DATA[DATA.keys()[[3, 4, 5, 7, 8, 9, 11]]]

    TARGET = DATA['review_overall']
    TARGET_HEADER = [['review_overall']]
    TARGET = TARGET.to_numpy().reshape((1,-1))

    DATA = DATA.drop(columns=['review_overall'])

    RAW_DATA = DATA.copy()
    RAW_DATA_HEADER = np.fromiter(RAW_DATA.keys(), dtype='<U50').reshape((1,-1))
    RAW_DATA = RAW_DATA.to_numpy()

    REFVECS = np.vstack((np.fromiter(range(len(RAW_DATA)), dtype=int).reshape((1, -1)),
                         np.random.choice(['A', 'B', 'C', 'D', 'E'], len(RAW_DATA), replace=True),
                         np.random.choice(['A', 'B', 'C', 'D', 'E'], len(RAW_DATA), replace=True),
                         np.random.choice(['A', 'B', 'C', 'D', 'E'], len(RAW_DATA), replace=True),
                         np.random.choice(['A', 'B', 'C', 'D', 'E'], len(RAW_DATA), replace=True)
    ))

    REFVECS_HEADER = ['ROW_IDX', 'JUNK1', 'JUNK2', 'JUNK3', 'JUNK4']



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
                                 refvecs_return_orientation=refvecs_given_orient,
                                 REFVECS_OBJECT=REFVECS,
                                 REFVECS_OBJECT_HEADER=REFVECS_HEADER,
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

    # BEAR HASH THIS TO GO BACK TO FULL COLUMNS
    # SWNL[0] = SWNL[0][:7]
    # WORKING_SUPOBJS[0] = WORKING_SUPOBJS[0][:, :7]

    data_given_orientation = SXNLClass.data_current_orientation
    target_given_orientation = SXNLClass.target_current_orientation
    refvecs_given_orientation = SXNLClass.refvecs_current_orientation

    WORKING_CONTEXT = []
    WORKING_KEEP = SRNL[1][0]



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
    MI_OUTPUT_VECTOR = []


    SUPER_RAW_NUMPY_LIST_HOLDER, RAW_SUPOBJS_HOLDER, SUPER_WORKING_NUMPY_LIST_HOLDER, WORKING_SUPOBJS_HOLDER, \
    WORKING_CONTEXT_HOLDER, WORKING_KEEP_HOLDER, split_method_holder, LABEL_RULES_HOLDER, number_of_labels_holder, \
    event_value_holder, negative_value_holder, batch_method, batch_size, int_or_bin_only, max_columns, bypass_agg, \
    OUTPUT_VECTOR = \
        MIConfigRun(standard_config, mi_config, SRNL, RAW_SUPOBJS, SWNL, WORKING_SUPOBJS, data_given_orientation,
            target_given_orientation, refvecs_given_orientation, WORKING_CONTEXT, WORKING_KEEP, split_method, LABEL_RULES,
            number_of_labels, event_value, negative_value, mi_batch_method, mi_batch_size, mi_int_or_bin_only,
            mi_max_columns, mi_bypass_agg, MI_OUTPUT_VECTOR).configrun()















