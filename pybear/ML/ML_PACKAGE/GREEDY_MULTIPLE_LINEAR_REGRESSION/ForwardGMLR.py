import sys, inspect, time
import itertools    # BEAR
import numpy as np, pandas as pd
from debug import get_module_name as gmn
from data_validation import arg_kwarg_validater as akv
from general_data_ops import NoImprov as ni
from MLObjects import MLObject as mlo
from ML_PACKAGE.MLREGRESSION import MLRegression as mlr
from ML_PACKAGE.GREEDY_MULTIPLE_LINEAR_REGRESSION import ApexGMLR as agmlr



class ForwardGMLR(agmlr.ApexGMLR):

    def __init__(self, DATA, DATA_HEADER, data_given_orientation, TARGET, target_given_orientation, AVAILABLE_COLUMNS=None,
        max_columns=None, intcpt_col_idx=None, rglztn_fctr=None, score_method=None, TRAIN_RESULTS=None, TARGET_TRANSPOSE=None,
        TARGET_AS_LIST=None, data_run_orientation='ROW', target_run_orientation='ROW', conv_kill=None, pct_change=None,
        conv_end_method=None, bypass_validation=None):

        # MUST BE BEFORE super ##########################################################################################
        this_module = gmn.get_module_name(str(sys.modules[__name__]))
        fxn = inspect.stack()[0][3]

        self.score_method = akv.arg_kwarg_validater(score_method, 'score_method', ['Q', 'R', 'A', 'F', None],
                                                                            this_module, fxn, return_if_none='Q')

        self.conv_kill = conv_kill
        self.pct_change = pct_change if not pct_change is None else 0
        self.conv_end_method = conv_end_method if not conv_end_method is None else 'PROMPT'
        # END MUST BE BEFORE super ######################################################################################

        super().__init__(DATA, DATA_HEADER, data_given_orientation, TARGET, target_given_orientation, AVAILABLE_COLUMNS=AVAILABLE_COLUMNS,
            max_columns=max_columns, intcpt_col_idx=intcpt_col_idx, rglztn_fctr=rglztn_fctr, TRAIN_RESULTS=TRAIN_RESULTS,
            TARGET_TRANSPOSE=TARGET_TRANSPOSE, TARGET_AS_LIST=TARGET_AS_LIST, data_run_orientation=data_run_orientation,
            target_run_orientation=target_run_orientation, bypass_validation=bypass_validation, calling_module=this_module)


    def core_run(self, DATA, DATA_HEADER, data_run_orientation, TARGET, TARGET_TRANSPOSE, TARGET_AS_LIST,
                 target_run_orientation, AVAILABLE_COLUMNS, max_columns, intcpt_col_idx, rglztn_fctr, bypass_validation,
                 this_module):

        fxn = inspect.stack()[0][3]
        bear_time_display = lambda t0: round((time.time()-t0), 3)

        self.WINNING_COLUMNS = np.empty(0, dtype=np.int32)

        # Receiver HOLDS CURRENT WINNERS DATA + NEW DATA COL TO BE TESTED
        # BUILD A DONOR (BATCH_DATA) AND RECEIVER (FWD_AGG_DATA_WIP) CLASSES TO ################################
        # DONATE AND AGGLOMERATE WINNING COLUMNS ################################################################

        FwdAggData_Donor = mlo.MLObject(
                                        DATA,
                                        data_run_orientation,
                                        name='DATA',
                                        return_orientation='AS_GIVEN',
                                        return_format='AS_GIVEN',
                                        bypass_validation=bypass_validation,
                                        calling_module=this_module,
                                        calling_fxn=fxn
        )

        FIRST_INSERT_COLUMN = FwdAggData_Donor.return_columns([AVAILABLE_COLUMNS[0]],
                                                              return_orientation='AS_GIVEN', return_format='AS_GIVEN')



        # BUILD FwdAggData_Receiver FROM FIRST COLUMN TO GO INTO FORWARD MLR (INTERCEPT, IF PRESENT)
        FwdAggData_Receiver = mlo.MLObject(
                                            FIRST_INSERT_COLUMN,
                                            data_run_orientation,
                                            name='FWD_AGG_DATA_WIP',
                                            return_orientation='AS_GIVEN',
                                            return_format='AS_GIVEN',
                                            bypass_validation=bypass_validation,
                                            calling_module=this_module,
                                            calling_fxn=fxn
        )

        del FIRST_INSERT_COLUMN






        # BEAR --- RELIC OF FINDING LAST GOOD COLUMN
        # ml_regression_errored_last_pass = False



        ni_ctr_init = 0
        ni_ctr = 0
        best_value = float('-inf')

        # FWD AGG ALGORITHM, --GET r, RSQ, RSQ-ADJ, F, PUT IN A HOLDER ARRAY AS COLUMNS #################
        for f_itr, col_idx in enumerate(AVAILABLE_COLUMNS):

            ##########################################################################################################################
            ##########################################################################################################################
            # SELECTION OF NEXT COLUMN FROM AVAILABLE ################################################################################

            ################################################################################################################
            # PICK COLUMNS AVAILABLE FOR CYCLING THE INNER LOOP TO FIND THE NEXT AGG WINNER ################################

            if len(self.WINNING_COLUMNS) == max_columns:
                print(f'\n*** GREEDY ALGORITHM ENDED FOR MAXIMUM USER-ALLOWED COLUMNS REACHED ***\n')
                break
            # INNER for (f_itr2) SEES CORRECT COLUMNS TO CHOOSE FROM THRU WIP_AVAILABLE_COLUMNS
            # IF INTCPT, ONLY ALLOW ALG TO CHOOSE INTCPT ON FIRST PASS
            elif f_itr == 0 and not intcpt_col_idx is None:
                # IF ON FIRST ITR AND INTCPT IS PRESENT (IF PRESENT, MUST BE FIRST), SET ALL CUMUL SCORES TO ZERO
                # AND BYPASS MLR
                self.TRAIN_RESULTS.loc[DATA_HEADER[0], ('CUMUL', 'R')] = 0
                self.TRAIN_RESULTS.loc[DATA_HEADER[0], ('CUMUL', 'R2')] = 0
                self.TRAIN_RESULTS.loc[DATA_HEADER[0], ('CUMUL', 'ADJ R2')] = 0
                self.TRAIN_RESULTS.loc[DATA_HEADER[0], ('CUMUL', 'F')] = 0
                self.WINNING_COLUMNS = np.hstack((self.WINNING_COLUMNS, [col_idx]))
                ni_ctr_init += 1
                continue

            # IF NO INTERCEPT, OR NOT ON 1st PASS (WHICH IS WHERE INTERCEPT WOULD BE) BUILD WIP_AVAILABLE_COLUMNS FROM UNUSED COLUMNS
            elif f_itr != 0 or intcpt_col_idx is None:
                WIP_AVAILABLE_COLUMNS = np.fromiter((_ for _ in AVAILABLE_COLUMNS if _ not in self.WINNING_COLUMNS), dtype=np.int32)

            # END PICK COLUMNS AVAILABLE FOR CYCLING THE INNER LOOP TO FIND THE NEXT AGG WINNER ############################
            ################################################################################################################

            current_round_high_score = -1

            for f_itr2, avail_col_idx in enumerate(WIP_AVAILABLE_COLUMNS):

                print(f'*' * 100)
                print(
                    f'        selecting column {f_itr + 1} of {min(len(AVAILABLE_COLUMNS), max_columns)} final winners... '
                    f'running column {f_itr2 + 1} of {len(WIP_AVAILABLE_COLUMNS)}...')

                if intcpt_col_idx is None and (f_itr==0 and f_itr2==0): pass
                else:
                    # ON FIRST PASS (IF NO INTERCEPT, WOULDNT GET HERE IF HAS INTERCEPT), DO NOT INSERT A
                    # COLUMN BECAUSE ReceiverClass WAS SEEDED WITH 1st COLUMN
                    # append avail_col_idx to Receiver
                    INSERT_COLUMN = FwdAggData_Donor.return_columns([avail_col_idx], return_orientation='AS_GIVEN',
                                                                    return_format='AS_GIVEN')

                    FwdAggData_Receiver.insert_column(f_itr, INSERT_COLUMN, data_run_orientation)

                    del INSERT_COLUMN

                if data_run_orientation == 'COLUMN':
                    FWD_AGG_DATA_AS_COLUMN = FwdAggData_Receiver.return_as_column()
                    FWD_AGG_DATA_AS_ROW = FwdAggData_Receiver.return_as_row()
                elif data_run_orientation == 'ROW':
                    FWD_AGG_DATA_AS_COLUMN = None
                    FWD_AGG_DATA_AS_ROW = FwdAggData_Receiver.return_as_row()

                print(f'        Start MLRegression()....'); t1 = time.time()
                # returned from MLRegression
                # xtx_determinant, self.COEFFS, PREDICTED, P_VALUES, R, R2, R2_ADJ, F
                DUM, COEFF_HOLDER, DUM, P_VALUE_HOLDER, R_, R2_, R2_ADJ_, F_ = \
                    mlr.MLRegression(FWD_AGG_DATA_AS_ROW,
                                     'ROW',
                                     TARGET if target_run_orientation == 'ROW' else TARGET_TRANSPOSE,
                                     'ROW',
                                     DATA_TRANSPOSE=FWD_AGG_DATA_AS_COLUMN,
                                     TARGET_TRANSPOSE=TARGET_TRANSPOSE if target_run_orientation == 'ROW' else TARGET,
                                     TARGET_AS_LIST=TARGET_AS_LIST if target_run_orientation == 'ROW' else TARGET_AS_LIST.transpose(),
                                     XTX=None,
                                     XTX_INV=None,
                                     has_intercept=False if intcpt_col_idx is None else True,
                                     intercept_math=False if intcpt_col_idx is None else True,
                                     regularization_factor=rglztn_fctr,
                                     safe_matmul=not bypass_validation,
                                     bypass_validation=bypass_validation
                ).run()

                print(f'        MLRegression done. total time = {bear_time_display(t1)} sec')
                print(f'****************************************************************************')

                if self.score_method == 'R': score = abs(R_) if isinstance(R_, (int, float)) else R_
                elif self.score_method == 'Q': score = R2_
                elif self.score_method == 'A': score = R2_ADJ_
                elif self.score_method == 'F': score = F_

                # if mlregression blows up, it returns nan. return 0.
                score = score if isinstance(score, (float, int)) else 0

                if score >= current_round_high_score:  # ">=" (as opposed to ">") allows return of something if all wip_avail cause error and return 0s
                    current_round_high_score = score
                    best_score = score
                    best_idx = avail_col_idx  # retain high score idx
                    best_r = R_
                    best_r2 = R2_
                    best_r2_adj = R2_ADJ_
                    best_f = F_

                # REMOVE THIS TRIALS COLUMN FROM THE RECEIVER B4 IT GOES BACK TO TOP TO GET NEXT COLUMN
                FwdAggData_Receiver.delete_columns([f_itr])

            # END SELECTION OF NEXT COLUMN FROM AVAILABLE ############################################################################
            ##########################################################################################################################
            ##########################################################################################################################

            print(f'      \nStart BEAR after selection of best column for current pass thru columns'); t0 = time.time()

            if not self.conv_kill is None:

                ni_ctr, gmlr_conv_kill, best_value, abort = \
                    ni.NoImprov(best_score, f_itr, ni_ctr_init, ni_ctr, self.conv_kill, best_value, self.pct_change,
                                'FORWARD GMLR', conv_end_method=self.conv_end_method).max()

                if abort:
                    max_columns = f_itr
                    # THIS SHORT CIRCUIT WILL PREVENT LAST P_VALUES AND self.COEFFS FROM BEING RECORDED IF HIT CONV KILL
                    print(f'\nGreedy algorithm ended after {f_itr} rounds for no improvement in {self.METHOD_DICT[self.score_method]}\n')
                    break

            self.COEFFS = COEFF_HOLDER
            P_VALUES = P_VALUE_HOLDER

            # AFTER TRIAL COMPLETE
            # IF conv_kill NOT TRIPPED, APPEND best_idx TO Receiver & self.WINNING_COLUMNS

            # update Receiver w best column of last round
            INSERT_COLUMN = FwdAggData_Donor.return_columns([best_idx], return_orientation='AS_GIVEN', return_format='AS_GIVEN')
            FwdAggData_Receiver.insert_column(f_itr, INSERT_COLUMN, data_run_orientation)
            del INSERT_COLUMN

            self.WINNING_COLUMNS = np.insert(self.WINNING_COLUMNS, len(self.WINNING_COLUMNS), int(best_idx), axis=0)

            best_hdr = DATA_HEADER[best_idx]

            self.TRAIN_RESULTS.loc[best_hdr, ('CUMUL', 'R')] = best_r
            self.TRAIN_RESULTS.loc[best_hdr, ('CUMUL', 'R2')] = best_r2
            self.TRAIN_RESULTS.loc[best_hdr, ('CUMUL', 'ADJ R2')] = best_r2_adj
            self.TRAIN_RESULTS.loc[best_hdr, ('CUMUL', 'F')] = best_f

            del best_r, best_r2, best_r2_adj, best_f, best_hdr

            '''
            BEAR  --- RELIC OF FINDING LAST GOOD COLUMN
            ##################################################################################################################
            ### handling if r, r2, adj_r2 or f give err or nan ################################################################

            at any point, self.coeffs, p_values, r_list, r2_list, r2_adj_list or f_list could go "err" or "nan"
            look at entire coeffs and p_values columns, since they are updated in full on every pass, but only look at
            last entry for r_list, r2_list, r2_adj_list, f_list, since these might recover to good values again
            
            # BEAR 6/9/23 THIS IS OBSOLETE, WILL HAVE TO FIGURE OUT A WAY TO PULL THIS INFO FROM self.TRAIN_RESULTS
            str_zipped_upper_data = np.char.upper(np.array(
                list(map(list,
                         itertools.zip_longest(*(self.COEFFS, P_VALUES, [R_LIST[f_itr]], [R2_LIST[f_itr]],
                                                 [R2_ADJ_LIST[f_itr]], [F_LIST[f_itr]]), fillvalue=0)
                         )),
                dtype=str))

            ml_regression_errored_this_pass = \
                True in np.fromiter((_ in str_zipped_upper_data for _ in ['nan', 'err']), dtype=bool)

            if ml_regression_errored_this_pass:
                if not ml_regression_errored_last_pass:  # if errored this pass but not last, this is a new cotoff
                    error_winrank = f_itr
                # elif ml_regression_errored_last_pass: pass   # if errored this pass and last, just carry on
                ml_regression_errored_last_pass = True

            elif not ml_regression_errored_this_pass:  # if mlr did not error, record its results because it could be the last good pass
                error_winrank = float('inf')  # allows that if subsequent passes dont error, can go back to
                # recognizing they are good. create backups that hold last good result (non-err, non-nan)
                ml_regression_errored_last_pass = False

            ### end handling if r, r2, adj_r2 or f give err or nan #########################################################
            ##################################################################################################################
            '''

            print(f'      \nEnd BEAR after selection of best column. time = {bear_time_display(t0)} sec')

        else:  # GOES WITH A for, NOT if. IF MADE IT THRU OUTER FOR LOOP W/O BREAK, EXHAUSTED ALL AVAILABLE COLUMNS
            print(f'\nGreedy algorithm ended for available columns exhausted\n')



        # DONT USE AVAILABLE_COLUMNS TO CHOP OR SORT TRAIN_RESULTS HERE
        self.TRAIN_RESULTS = self.TRAIN_RESULTS.iloc[self.WINNING_COLUMNS, :]

        self.TRAIN_RESULTS.loc[:, ('FINAL', 'COEFFS')] = self.COEFFS
        self.TRAIN_RESULTS.loc[:, ('FINAL', 'p VALUE')] = P_VALUES

        # #################################################################################################################
        # #################################################################################################################

        del FwdAggData_Donor, FwdAggData_Receiver, COEFF_HOLDER, P_VALUE_HOLDER

        print(
            f'\nSuccessfully completed Forward greedy MLR using {self.METHOD_DICT[self.score_method]}, ' + \
            f'selecting {len(self.WINNING_COLUMNS)} of {min(max_columns, self.data_cols)} ' + \
            f'allowed columns (of {self.data_cols} features in original data' + \
            f' including intercept)' if not intcpt_col_idx is None else ')'
            )















if __name__ == '__main__':

    import numpy as np, pandas as pd, sparse_dict as sd
    from MLObjects.TestObjectCreators.SXNL import CreateSXNL as csxnl
    from MLObjects.SupportObjects import master_support_object_dict as msod
    from data_validation import validate_user_input as vui
    from ML_PACKAGE.GREEDY_MULTIPLE_LINEAR_REGRESSION import build_empty_gmlr_train_results as begtr
    from general_sound import winlinsound as wls


    DATA = pd.read_csv(r'C:\Users\Bill\Documents\WORK STUFF\RESUME\1 - OTHER\SRP\beer_reviews.csv',
                      nrows=150,
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



    # GIVEN TO ForwardGMLR test, NOT CreateSXNL!
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


    DATA = SWNL[0]
    DATA_HEADER = WORKING_SUPOBJS[0][msod.QUICK_POSN_DICT()["HEADER"]].reshape((1,-1))
    TARGET = SWNL[1]

    # DELETE AN ARBITRARY BIN COLUMN TO GET RID OF MULTI-COLINEARITY
    delete_idx = 4
    DATA = np.delete(DATA, delete_idx, axis=0)
    DATA_HEADER = np.delete(DATA_HEADER, delete_idx, axis=1)

    MASTER_BYPASS_VALIDATION = [True, False]
    MASTER_MAX_COLUMNS = [10]  #[2, len(DATA)+10]  # 6/5/23 MUST BE len==1
    MASTER_INTCPT_GIVEN = [False, True]
    MASTER_INTCPT_IDX_GIVEN = [True, False]
    MASTER_RGLZTN_FCTR = [0, 100]
    MASTER_AV_COLS_GIVEN = [True, False]
    MASTER_TRAIN_RESULTS_GIVEN = [True, False]
    MASTER_TARGET_HELPERS_GIVEN = [True, False]
    MASTER_DATA_FORMAT = ['SPARSE_DICT', 'ARRAY']
    MASTER_DATA_ORIENT = ['ROW', 'COLUMN']
    MASTER_TARGET_FORMAT = ['ARRAY', 'SPARSE_DICT']
    MASTER_TARGET_ORIENT = ['ROW', 'COLUMN']





    # BUILD WINNING_COLUMNS & TRAIN_RESULTS ANSWER KEYS #########################################################################

    score_method = 'Q'

    if score_method == 'R': look_row = 0  # R
    elif score_method == 'Q': look_row = 1  # RSQ
    elif score_method == 'A': look_row = 2  # ADJ RSQ
    elif score_method == 'F': look_row = 3  # F

    for int_idx, has_intercept in enumerate(MASTER_INTCPT_GIVEN):
        if has_intercept:
            WIP_DATA = np.insert(DATA.copy(), 0, 1, axis=0)
            WIP_HEADER = np.hstack(([['INTERCEPT']] if has_intercept else [[]], DATA_HEADER.copy()))
        else:
            WIP_DATA = DATA.copy()
            WIP_HEADER = DATA_HEADER.copy()


        for rgz_idx, rglztn_fctr in enumerate(MASTER_RGLZTN_FCTR):

            print(f'Running trial {2*(int_idx)+(rgz_idx)+1}')

            #############################################################################################################
            # CORE FORWARD ALGORITHM ####################################################################################
            AVAILABLE_COLUMNS = np.arange(0, len(WIP_DATA), dtype=np.int32)
            TRIAL_DATA_HOLDER = np.empty((0, len(WIP_DATA[0])), dtype=np.float64)
            WIP_TRAIN_HOLDER = begtr.build_empty_gmlr_train_results(WIP_HEADER)
            WINNING_COLUMNS = np.empty(0, dtype=np.int8)

            for col_idx in range(min(MASTER_MAX_COLUMNS[0], len(AVAILABLE_COLUMNS))):

                print(f'Looking for column {col_idx+1}...')

                # CREATE AN ALLOWED_AVAILABLE COLUMNS TO ITERATE OVER THAT HOLDS THE COLUMNS THAT HAVENT BEEN USED YET
                ALLOWED_AV_COLS = np.fromiter((_ for _ in AVAILABLE_COLUMNS if _ not in WINNING_COLUMNS), dtype=np.int8)

                BEST_SCORES_HOLDER = np.empty((len(ALLOWED_AV_COLS), 4), dtype=object)

                if col_idx == 0 and WIP_HEADER[0][col_idx] == 'INTERCEPT':   # IF INTERCEPT WOULD BE ONLY COLUMN GOING INTO MLR
                    TRIAL_DATA_HOLDER = WIP_DATA[0].reshape((1,-1))
                    WIP_TRAIN_HOLDER.iloc[col_idx, 6:10] = (0, 0, 0, 0)
                    WINNING_COLUMNS = np.hstack((WINNING_COLUMNS, [col_idx]))
                    continue

                for itr2, col_idx2 in enumerate(ALLOWED_AV_COLS):

                    TRIAL_DATA_HOLDER = np.vstack((TRIAL_DATA_HOLDER, WIP_DATA[col_idx2]))

                    # returned from MLRegression
                    # xtx_determinant, self.COEFFS, PREDICTED, P_VALUES, R, R2, R2_ADJ, F
                    DUM, COEFFS, DUM, P_VALUES, R_, R2_, R2_ADJ_, F_ = \
                        mlr.MLRegression(
                                         TRIAL_DATA_HOLDER.transpose(),
                                         'ROW',
                                         TARGET.reshape((-1,1)),
                                         'ROW',
                                         DATA_TRANSPOSE=TRIAL_DATA_HOLDER,
                                         TARGET_TRANSPOSE=TARGET.reshape((1,-1)),
                                         TARGET_AS_LIST=TARGET.reshape((-1,1)),
                                         XTX=None,
                                         XTX_INV=None,
                                         has_intercept=has_intercept,
                                         intercept_math=has_intercept,
                                         regularization_factor=rglztn_fctr,
                                         safe_matmul=False,     # True if bypass_validation is False
                                         bypass_validation=True
                    ).run()

                    BEST_SCORES_HOLDER[itr2] = (R_, R2_, R2_ADJ_, F_)

                    # TAKE OFF LAST COLUMN TO ALLOW REPLACEMENT BY THE NEXT ONE AT THE TOP OF THIS for LOOP
                    TRIAL_DATA_HOLDER = TRIAL_DATA_HOLDER[:-1]

                # AFTER GETTING ALL SCORES, ETC. FIND THE BEST COLUMN
                win_idx = np.argwhere(BEST_SCORES_HOLDER[:, look_row] == np.max(BEST_SCORES_HOLDER[:, look_row]))[0][0]
                win_col_idx = ALLOWED_AV_COLS[win_idx]
                WINNING_COLUMNS = np.hstack((WINNING_COLUMNS, [win_col_idx]))
                # UPDATE TRAIN_HOLDER & PERMANENTLY PUT win_col_idx IN TRIAL_DATA_HOLDER
                WIP_TRAIN_HOLDER.iloc[col_idx, 6:10] = BEST_SCORES_HOLDER[win_idx]
                TRIAL_DATA_HOLDER = np.vstack((TRIAL_DATA_HOLDER, WIP_DATA[win_col_idx]))

            else:  # ONCE GET THRU AVAILABLE_COLUMNS for LOOP, LAST COEFFS AND P_VALUES ARE THE ONES THAT MATTER
                del win_idx, win_col_idx, TRIAL_DATA_HOLDER, DUM, R_, R2_, R2_ADJ_, F_, BEST_SCORES_HOLDER
                WIP_TRAIN_HOLDER = WIP_TRAIN_HOLDER.iloc[:MASTER_MAX_COLUMNS[0], :]
                WIP_TRAIN_HOLDER.iloc[:, 4] = COEFFS
                WIP_TRAIN_HOLDER.iloc[:, 5] = P_VALUES
                WIP_TRAIN_HOLDER.index = WIP_HEADER[0][WINNING_COLUMNS]

            # END CORE FORWARD ALGORITHM ################################################################################
            #############################################################################################################

            if rgz_idx==0 and not has_intercept:
                RGZ_0_INT_0_COL_KEY = WINNING_COLUMNS.copy()
                RGZ_0_INT_0_TRAIN_KEY = WIP_TRAIN_HOLDER.copy()
            elif rgz_idx==0 and has_intercept:
                RGZ_0_INT_1_COL_KEY = WINNING_COLUMNS.copy()
                RGZ_0_INT_1_TRAIN_KEY = WIP_TRAIN_HOLDER.copy()
            elif rgz_idx==1 and not has_intercept:
                RGZ_1_INT_0_COL_KEY = WINNING_COLUMNS.copy()
                RGZ_1_INT_0_TRAIN_KEY = WIP_TRAIN_HOLDER.copy()
            elif rgz_idx==1 and has_intercept:
                RGZ_1_INT_1_COL_KEY = WINNING_COLUMNS.copy()
                RGZ_1_INT_1_TRAIN_KEY = WIP_TRAIN_HOLDER.copy()

    print()
    print(f'RGZ_0_INT_0_TRAIN_KEY:')
    print(RGZ_0_INT_0_TRAIN_KEY)
    print()
    print(f'RGZ_1_INT_0_TRAIN_KEY:')
    print(RGZ_1_INT_0_TRAIN_KEY)
    print()
    print(f'RGZ_0_INT_1_TRAIN_KEY:')
    print(RGZ_0_INT_1_TRAIN_KEY)
    print()
    print(f'RGZ_1_INT_1_TRAIN_KEY:')
    print(RGZ_1_INT_1_TRAIN_KEY)
    print()
    __ = input(' > ')
    # END BUILD WINNING_COLUMNS AND TRAIN_RESULTS ANSWER KEYS #####################################################################



    total_trials = np.product(list(map(len, (MASTER_BYPASS_VALIDATION, MASTER_MAX_COLUMNS, MASTER_INTCPT_GIVEN,
        MASTER_INTCPT_IDX_GIVEN, MASTER_RGLZTN_FCTR, MASTER_AV_COLS_GIVEN, MASTER_TRAIN_RESULTS_GIVEN,
        MASTER_TARGET_HELPERS_GIVEN, MASTER_DATA_FORMAT, MASTER_DATA_ORIENT, MASTER_TARGET_FORMAT, MASTER_TARGET_ORIENT))))

    ctr = 0
    for bypass_validation in MASTER_BYPASS_VALIDATION:
        for max_columns in MASTER_MAX_COLUMNS:
            for intcpt_is_given in MASTER_INTCPT_GIVEN:
                for intcpt_idx_is_given in MASTER_INTCPT_IDX_GIVEN:
                    for rglztn_fctr in MASTER_RGLZTN_FCTR:
                        for av_cols_is_given in MASTER_AV_COLS_GIVEN:
                            for train_results_given in MASTER_TRAIN_RESULTS_GIVEN:
                                for target_helpers_given in MASTER_TARGET_HELPERS_GIVEN:
                                    for data_format in MASTER_DATA_FORMAT:
                                        for data_orient in MASTER_DATA_ORIENT:
                                            for target_format in MASTER_TARGET_FORMAT:
                                                for target_orient in MASTER_TARGET_ORIENT:


                                                    ctr += 1
                                                    print(f'*'*140)
                                                    print(f'\nRunning trial {ctr} of {total_trials}...')

                                                    # BUILD / ORIENT / FORMAT GIVENS ################################################################
                                                    print(f'\nCreating given objects...')

                                                    GIVEN_DATA = DATA.copy()
                                                    GIVEN_DATA_HEADER = DATA_HEADER.copy()
                                                    intcpt_col_idx = None
                                                    if intcpt_is_given:
                                                        GIVEN_DATA = np.insert(GIVEN_DATA, 0, 1, axis=0)
                                                        if intcpt_idx_is_given: intcpt_col_idx = 0
                                                        # elif not intcpt_idx_is_given: intcpt_col_idx STAYS None
                                                        GIVEN_DATA_HEADER = np.hstack(([['INTERCEPT']], GIVEN_DATA_HEADER))
                                                    if data_orient == 'COLUMN': pass
                                                    elif data_orient == 'ROW': GIVEN_DATA = GIVEN_DATA.transpose()
                                                    if data_format == 'ARRAY': pass
                                                    elif data_format == 'SPARSE_DICT': GIVEN_DATA = sd.zip_list_as_py_float(GIVEN_DATA)

                                                    GIVEN_TARGET = TARGET.copy()
                                                    GIVEN_TARGET_TRANSPOSE = None
                                                    GIVEN_TARGET_AS_LIST = None
                                                    if target_orient == 'COLUMN':
                                                        if target_helpers_given:
                                                            GIVEN_TARGET_TRANSPOSE = GIVEN_TARGET.copy().transpose()
                                                            GIVEN_TARGET_AS_LIST = GIVEN_TARGET.copy()
                                                    elif target_orient == 'ROW':
                                                        GIVEN_TARGET = GIVEN_TARGET.transpose()
                                                        if target_helpers_given:
                                                            GIVEN_TARGET_TRANSPOSE = GIVEN_TARGET.copy().transpose()
                                                            GIVEN_TARGET_AS_LIST = GIVEN_TARGET.copy()
                                                    if target_format == 'ARRAY': pass
                                                    elif target_format == 'SPARSE_DICT':
                                                        GIVEN_TARGET = sd.zip_list_as_py_float(GIVEN_TARGET)
                                                        if target_helpers_given:
                                                            GIVEN_TARGET_TRANSPOSE = sd.zip_list_as_py_float(GIVEN_TARGET_TRANSPOSE)

                                                    if av_cols_is_given: AVAILABLE_COLUMNS = np.fromiter(range(len(DATA)+int(intcpt_is_given)), dtype=np.int32)
                                                    else: AVAILABLE_COLUMNS = None

                                                    if train_results_given:
                                                        GIVEN_TRAIN_RESULTS = begtr.build_empty_gmlr_train_results(GIVEN_DATA_HEADER)
                                                    else:
                                                        GIVEN_TRAIN_RESULTS = None

                                                    print(f'\nDone creating given objects...')
                                                    # END BUILD / ORIENT / FORMAT GIVENS ############################################################

                                                    # GET EXPECTEDS #################################################################################
                                                    if intcpt_is_given:
                                                        if rglztn_fctr==MASTER_RGLZTN_FCTR[0]:
                                                            EXP_WINNING_COLUMNS = RGZ_0_INT_1_COL_KEY
                                                            EXP_TRAIN_RESULTS = RGZ_0_INT_1_TRAIN_KEY
                                                        elif rglztn_fctr==MASTER_RGLZTN_FCTR[1]:
                                                            EXP_WINNING_COLUMNS = RGZ_1_INT_1_COL_KEY
                                                            EXP_TRAIN_RESULTS = RGZ_1_INT_1_TRAIN_KEY
                                                    else:
                                                        if rglztn_fctr==MASTER_RGLZTN_FCTR[0]:
                                                            EXP_WINNING_COLUMNS = RGZ_0_INT_0_COL_KEY
                                                            EXP_TRAIN_RESULTS = RGZ_0_INT_0_TRAIN_KEY
                                                        elif rglztn_fctr==MASTER_RGLZTN_FCTR[1]:
                                                            EXP_WINNING_COLUMNS = RGZ_1_INT_0_COL_KEY
                                                            EXP_TRAIN_RESULTS = RGZ_1_INT_0_TRAIN_KEY

                                                    EXP_WINNING_COLUMNS = EXP_WINNING_COLUMNS[:max_columns]
                                                    EXP_TRAIN_RESULTS = EXP_TRAIN_RESULTS.iloc[:max_columns, :]
                                                    # END GET EXPECTEDS #############################################################################


                                                    ACT_WINNING_COLUMNS, ACT_TRAIN_RESULTS, COEFFS = \
                                                        ForwardGMLR(GIVEN_DATA,
                                                                     GIVEN_DATA_HEADER,
                                                                     data_orient,
                                                                     GIVEN_TARGET,
                                                                     target_orient,
                                                                     AVAILABLE_COLUMNS=AVAILABLE_COLUMNS,
                                                                     max_columns=max_columns,
                                                                     intcpt_col_idx=intcpt_col_idx,
                                                                     rglztn_fctr=rglztn_fctr,
                                                                     score_method=score_method,
                                                                     TRAIN_RESULTS=GIVEN_TRAIN_RESULTS,
                                                                     TARGET_TRANSPOSE=GIVEN_TARGET_TRANSPOSE,
                                                                     TARGET_AS_LIST=GIVEN_TARGET_AS_LIST,
                                                                     data_run_orientation='ROW', #'AS_GIVEN',
                                                                     target_run_orientation='ROW', #'AS_GIVEN',
                                                                     conv_kill=None,
                                                                     pct_change=None,
                                                                     conv_end_method=None,
                                                                     bypass_validation=bypass_validation).run()


                                                    if not np.array_equiv(ACT_WINNING_COLUMNS, EXP_WINNING_COLUMNS):

                                                        print(f'\033[91m')
                                                        print(f'ACTUAL WINNING COLUMNS VIA ACT_TRAIN_RESULTS:')
                                                        print(ACT_TRAIN_RESULTS)
                                                        print()
                                                        print(f'EXPECTED WINNING COLUMNS:')
                                                        print(EXP_TRAIN_RESULTS)
                                                        print()
                                                        wls.winlinsound(444, 1000)
                                                        # if vui.validate_user_str(f'\nContinue(c) or kill(k) > ', 'CK') == 'K':
                                                        raise Exception(f'*** EXP/ACT WINNING COLUMNS NOT EQUAL ***')

                                                    # COEFFS ARE CHECKED WItHIN TRAIN_RESULTS
                                                    if intcpt_is_given and rglztn_fctr == 0:
                                                        ACT_TRAIN_RESULTS.to_numpy()[:, 4:].astype(np.float64)
                                                        PASS_TRAIN_RESULTS_TEST = np.allclose(
                                                            ACT_TRAIN_RESULTS.to_numpy()[:, 4:].astype(np.float64),
                                                            EXP_TRAIN_RESULTS.to_numpy()[:, 4:].astype(np.float64)
                                                        )
                                                    elif intcpt_is_given and rglztn_fctr != 0:
                                                        PASS_TRAIN_RESULTS_TEST = np.allclose(
                                                            ACT_TRAIN_RESULTS.to_numpy()[:, [4, 6, 7, 8, 9]].astype(np.float64),
                                                            EXP_TRAIN_RESULTS.to_numpy()[:, [4, 6, 7, 8, 9]].astype(np.float64)
                                                        )
                                                    elif not intcpt_is_given and rglztn_fctr == 0:
                                                        PASS_TRAIN_RESULTS_TEST = np.allclose(
                                                            ACT_TRAIN_RESULTS.to_numpy()[:, [4, 5, 7, 9]].astype(np.float64),
                                                            EXP_TRAIN_RESULTS.to_numpy()[:, [4, 5, 7, 9]].astype(np.float64)
                                                        )
                                                    elif not intcpt_is_given and rglztn_fctr != 0:
                                                        PASS_TRAIN_RESULTS_TEST = np.allclose(
                                                            ACT_TRAIN_RESULTS.to_numpy()[:, [4, 7, 9]].astype(np.float64),
                                                            EXP_TRAIN_RESULTS.to_numpy()[:, [4, 7, 9]].astype(np.float64)
                                                        )

                                                    # not ACT_TRAIN_RESULTS.equals(EXP_TRAIN_RESULTS):
                                                    if not PASS_TRAIN_RESULTS_TEST:
                                                        print(f'\033[91m')
                                                        print(f'ACTUAL TRAIN RESULTS:')
                                                        print(ACT_TRAIN_RESULTS)
                                                        print()
                                                        print(f'EXPECTED TRAIN RESULTS:')
                                                        print(EXP_TRAIN_RESULTS)
                                                        wls.winlinsound(444, 2000)
                                                        print(f'*** EXP/ACT TRAIN RESULTS NOT EQUAL ***')
                                                        if vui.validate_user_str(f'\nContinue(c) or kill(k) > ', 'CK') == 'K':
                                                            raise Exception(f'*** EXP/ACT TRAIN RESULTS NOT EQUAL ***')

                                                    print(f'*'*140)


    print(f'\n\033[92m*** ALL TESTS PASSED ***\033[0m')
    for _ in range(3): wls.winlinsound(888, 500); time.sleep(1)





