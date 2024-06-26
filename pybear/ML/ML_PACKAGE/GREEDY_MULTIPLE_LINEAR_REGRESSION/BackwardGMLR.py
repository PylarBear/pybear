import sys, inspect, time
import numpy as np, pandas as pd
import sparse_dict as sd
from debug import get_module_name as gmn
from general_data_ops import get_shape as gs, NoImprov as ni
from data_validation import arg_kwarg_validater as akv
from MLObjects import MLObject as mlo
from ML_PACKAGE.MLREGRESSION import MLRegression as mlr
from ML_PACKAGE.GREEDY_MULTIPLE_LINEAR_REGRESSION import ApexGMLR as agmlr, LazyAggGMLR as lagg



class BackwardGMLR(agmlr.ApexGMLR):
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


        # RUN A ONE-OFF MLRegression ON FULL DATA TO GET STARTING best_value #############################################
        # BackwardDataWip HOLDS FULL DATA, TO HAVE LOSING COLUMNS DROPPED
        BackwardDataWip = mlo.MLObject(
                                        DATA,
                                        data_run_orientation,
                                        name='DATA',
                                        return_orientation='AS_GIVEN',
                                        return_format='AS_GIVEN',
                                        bypass_validation=bypass_validation,
                                        calling_module=this_module,
                                        calling_fxn=fxn
        )

        R_, R2_, R2_ADJ_, F_ = \
                mlr.MLRegression(
                                 BackwardDataWip.return_as_row(),
                                 'ROW',
                                 TARGET if target_run_orientation == 'ROW' else TARGET_TRANSPOSE,
                                 'ROW',
                                 DATA_TRANSPOSE=BackwardDataWip.return_as_column(),
                                 TARGET_TRANSPOSE=TARGET_TRANSPOSE if target_run_orientation == 'ROW' else TARGET,
                                 TARGET_AS_LIST=TARGET_AS_LIST if target_run_orientation == 'ROW' else TARGET_AS_LIST.transpose(),
                                 XTX=None,
                                 XTX_INV=None,
                                 has_intercept=False if intcpt_col_idx is None else True,
                                 intercept_math=False if intcpt_col_idx is None else True,
                                 regularization_factor=rglztn_fctr,
                                 safe_matmul=not bypass_validation,
                                 bypass_validation=bypass_validation
        ).run()[4:]

        if self.score_method == 'R': best_score = abs(R_)
        elif self.score_method == 'Q': best_score = R2_
        elif self.score_method == 'A': best_score = R2_ADJ_
        elif self.score_method == 'F': best_score = F_

        best_score = best_score if isinstance(best_score, (int, float)) else best_score

        del R_, R2_, R2_ADJ_, F_

        # END RUN A ONE-OFF MLRegression ON FULL DATA TO GET STARTING best_score #############################################




        ni_ctr_init = 0
        ni_ctr = 0
        # best_score --- ABOVE

        # MUST BE BEFORE AV_COLS CHOP TO KEEP INTERCEPT AS A WINNER
        self.WINNING_COLUMNS = AVAILABLE_COLUMNS.copy()

        # DONT ALLOW INTCPT, IF PRESENT, TO BE DROPPED
        if not intcpt_col_idx is None:
            AVAILABLE_COLUMNS = AVAILABLE_COLUMNS[AVAILABLE_COLUMNS!=intcpt_col_idx]


        # BACKWARD ALGORITHM, --GET r, RSQ, RSQ-ADJ, F, PUT IN A HOLDER ARRAY AS COLUMNS #################
        abort = False
        for b_itr, col_idx in enumerate(AVAILABLE_COLUMNS):

            # INNER for (b_itr2) SEES CORRECT COLUMNS TO CHOOSE FROM THRU WINNING_COLUMNS, WHICH IS CHOPPED WHEN COLUMN IS DROPPED
            current_round_high_score = float('-inf')

            for b_itr2, avail_col_idx in enumerate(self.WINNING_COLUMNS):

                print(f'*' * 100)
                print(
                    f'        selecting column number {b_itr + 1} to drop... '
                    f'running column {b_itr2 + 1} of {len(self.WINNING_COLUMNS)}...')

                if avail_col_idx == intcpt_col_idx: continue

                COLUMN_BACKUP = BackwardDataWip.return_columns([b_itr2], return_orientation='AS_GIVEN',
                                                                    return_format='AS_GIVEN')

                BackwardDataWip.delete_columns([b_itr2])


                print(f'        Start MLRegression()....'); t1 = time.time()
                # returned from MLRegression
                # xtx_determinant, self.COEFFS, PREDICTED, P_VALUES, R, R2, R2_ADJ, F
                R_, R2_, R2_ADJ_, F_ = \
                    mlr.MLRegression(BackwardDataWip.return_as_row(),
                                     'ROW',
                                     TARGET if target_run_orientation == 'ROW' else TARGET_TRANSPOSE,
                                     'ROW',
                                     DATA_TRANSPOSE=BackwardDataWip.return_as_column(),
                                     TARGET_TRANSPOSE=TARGET_TRANSPOSE if target_run_orientation == 'ROW' else TARGET,
                                     TARGET_AS_LIST=TARGET_AS_LIST if target_run_orientation == 'ROW' else TARGET_AS_LIST.transpose(),
                                     XTX=None,
                                     XTX_INV=None,
                                     has_intercept=False if intcpt_col_idx is None else True,
                                     intercept_math=False if intcpt_col_idx is None else True,
                                     regularization_factor=rglztn_fctr,
                                     safe_matmul=not bypass_validation,
                                     bypass_validation=bypass_validation
                ).run()[4:]

                print(f'        MLRegression done. total time = {bear_time_display(t1)} sec')
                print(f'****************************************************************************')

                if self.score_method == 'R': score = abs(R_)
                elif self.score_method == 'Q': score = R2_
                elif self.score_method == 'A': score = R2_ADJ_
                elif self.score_method == 'F': score = F_

                # IF MLRegression() BLOWS UP, IT RETURNS nan. RETURN 0.
                score = score if isinstance(score, (float, int)) else 0

                if score >= current_round_high_score:
                    current_round_high_score = score
                    best_idx = b_itr2  # retain high score idx

                # PUT COLUMN BACK
                BackwardDataWip.insert_column(b_itr2, COLUMN_BACKUP, insert_orientation=data_run_orientation)

                del COLUMN_BACKUP

            # END SELECTION OF NEXT COLUMN FROM AVAILABLE ############################################################################
            ##########################################################################################################################
            ##########################################################################################################################

            if not self.conv_kill is None:
                ni_ctr, self.conv_kill, best_score, abort = \
                    ni.NoImprov(score, b_itr, ni_ctr_init, ni_ctr, self.conv_kill, best_score, self.pct_change,
                                'BACKWARD GMLR', conv_end_method=self.conv_end_method).max()

                # IF SIGNAL TO ABORT BUT NOT CHOPPED DOWN TO max_columns YET, KEEP GOING
                abort = False if len(self.WINNING_COLUMNS) > max_columns else abort


            if abort:
                # len(self.WINNING_COLUMNS) MUST BE <= max_columns TO GET HERE
                print(f'\nBackward search algorithm ended after {b_itr+1} rounds for no improvement in {self.METHOD_DICT[self.score_method]} '
                      f'and columns ({len(self.WINNING_COLUMNS)}) <= max_columns ({max_columns})\n')
                break

            if not abort:  # ONLY DELETE IF IT IMPROVES score
                # PERMANENTLY REMOVE best_idx FROM WINNING_COLUMNS  (best_idx IS THE IDX THAT MAXIMIZES score THE MOST UPON REMOVAL)
                self.WINNING_COLUMNS = np.delete(self.WINNING_COLUMNS, best_idx, axis=0)
                BackwardDataWip.delete_columns([best_idx])
                if len(self.WINNING_COLUMNS) >= max_columns: pass
                    # HAVENT REACHED PEAK YET AND STILL HAVE MORE COLUMNS TO CHOP, JUST BUSINESS AS USUAL
                elif len(self.WINNING_COLUMNS) < max_columns:
                    max_columns -= 1

            # len(WINNING_COLUMNS) HAS ONLY ONE COLUMN (NOT COUNTING INTERCEPT)
            if len(self.WINNING_COLUMNS) == 1 + [1 if not intcpt_col_idx is None else 0][0]:
                print(f'\n*** Backward search algorithm ended after {b_itr + 1} rounds for one column remaining ***\n')
                break

            if self.conv_kill is None and len(self.WINNING_COLUMNS)==max_columns:
                print(f'\n*** Backward search algorithm ended after {b_itr + 1} rounds for reaching max columns ***\n')



        # RUN LazyAgg ON WINNING COLUMNS TO BUILD TRAIN_RESULTS ##################################################################

        del BackwardDataWip

        # ENSURE intcpt_col_idx IS FIRST IF PRESENT
        if not intcpt_col_idx is None:
            self.WINNING_COLUMNS = np.insert(self.WINNING_COLUMNS[self.WINNING_COLUMNS!=intcpt_col_idx], 0, intcpt_col_idx, axis=0)


        self.TRAIN_RESULTS = self.TRAIN_RESULTS.iloc[self.WINNING_COLUMNS, :]


        print(f'      \nStart BEAR LazyAgg for WINNING_COLUMNS'); t0 = time.time()

        WinningDataClass = mlo.MLObject(
                                        DATA,
                                        data_run_orientation,
                                        name='BACKWARD_DATA',
                                        return_orientation='AS_GIVEN',
                                        return_format='AS_GIVEN',
                                        bypass_validation=True,
                                        calling_module=gmn.get_module_name(str(sys.modules[__name__])),
                                        calling_fxn='GUARD TEST'
        )

        WIP_HEADER = DATA_HEADER[self.WINNING_COLUMNS]

        WINNING_DATA_AS_ROW = WinningDataClass.return_columns(self.WINNING_COLUMNS, return_orientation='ROW', return_format='AS_GIVEN')

        self.TRAIN_RESULTS, self.COEFFS = \
                    lagg.LazyAggGMLR(
                                    WINNING_DATA_AS_ROW,
                                    WIP_HEADER,
                                    'ROW',
                                    TARGET if target_run_orientation == 'ROW' else TARGET_TRANSPOSE,
                                    'ROW',
                                    AVAILABLE_COLUMNS=np.arange(0, len(self.WINNING_COLUMNS)),
                                    max_columns=None,
                                    intcpt_col_idx=None if intcpt_col_idx is None else 0,  # WAS INSERTED INTO 0 SLOT IN DATA WHEN USED
                                    rglztn_fctr=rglztn_fctr,
                                    score_method=self.score_method,
                                    TRAIN_RESULTS=self.TRAIN_RESULTS,
                                    TARGET_TRANSPOSE=TARGET_TRANSPOSE if target_run_orientation == 'ROW' else TARGET,
                                    TARGET_AS_LIST=TARGET_AS_LIST if target_run_orientation == 'ROW' else TARGET_AS_LIST.transpose(),
                                    data_run_orientation='ROW',
                                    target_run_orientation='ROW',
                                    conv_kill=None,
                                    pct_change=None,
                                    conv_end_method=None,
                                    bypass_validation=False
        ).run()[1:]

        del WinningDataClass, WINNING_DATA_AS_ROW

        print(f'      \nEnd BEAR LazyAgg for WINNING_COLUMNS. time = {bear_time_display(t0)} sec')
        # END RUN LazyAgg ON WINNING COLUMNS TO BUILD TRAIN_RESULTS ##################################################################

        # #################################################################################################################
        # #################################################################################################################


        print(
            f'\nSuccessfully completed backward search MLR using {self.METHOD_DICT[self.score_method]}, ' + \
            f'selecting {len(self.WINNING_COLUMNS)} of {min(max_columns, self.data_cols)} ' + \
            f'allowed columns (of {self.data_cols} features in original data' + \
            f' including intercept)' if not intcpt_col_idx is None else ')'
            )
















if __name__ == '__main__':

    # MODULE & TEST CODE GOOD 6/11/23

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



    # GIVEN TO BackwardGMLR tests, NOT CreateSXNL!
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
    MASTER_INTCPT_GIVEN = [True, False]
    MASTER_INTCPT_IDX_GIVEN = [True, False]
    MASTER_RGLZTN_FCTR = [0, 100]
    MASTER_AV_COLS_GIVEN = [True, False]
    MASTER_TRAIN_RESULTS_GIVEN = [True, False]
    MASTER_TARGET_HELPERS_GIVEN = [True, False]
    MASTER_DATA_FORMAT = ['ARRAY', 'SPARSE_DICT']
    MASTER_DATA_ORIENT = ['ROW', 'COLUMN']
    MASTER_TARGET_FORMAT = ['ARRAY', 'SPARSE_DICT']
    MASTER_TARGET_ORIENT = ['ROW', 'COLUMN']


    # BUILD WINNING_COLUMNS & TRAIN_RESULTS ANSWER KEYS #########################################################################

    score_method = 'F'   # BEAR

    if score_method == 'R': look_row = 0  # R
    elif score_method == 'Q': look_row = 1  # RSQ
    elif score_method == 'A': look_row = 2  # ADJ RSQ
    elif score_method == 'F': look_row = 3  # F





    for int_idx, has_intercept in enumerate(MASTER_INTCPT_GIVEN):
        if has_intercept:
            WIP_DATA = np.insert(DATA.copy(), 0, 1, axis=0)
            WIP_HEADER = np.hstack(([['INTERCEPT']] if has_intercept else [[]], DATA_HEADER.copy()))
            intcpt_col_idx = 0
        else:
            WIP_DATA = DATA.copy()
            WIP_HEADER = DATA_HEADER.copy()
            intcpt_col_idx = None

        for rgz_idx, rglztn_fctr in enumerate(MASTER_RGLZTN_FCTR):

            print(f'Running key builder trial {2*(int_idx)+(rgz_idx)+1}')

            #############################################################################################################
            # CORE BACKWARD ALGORITHM ####################################################################################

            # RUN A ONE-OFF MLRegression ON FULL DATA TO GET STARTING best_value #############################################
            R_, R2_, R2_ADJ_, F_ = \
                                    mlr.MLRegression(
                                                     WIP_DATA.transpose(),
                                                     'ROW',
                                                     TARGET.reshape((-1, 1)),
                                                     'ROW',
                                                     DATA_TRANSPOSE=WIP_DATA,
                                                     TARGET_TRANSPOSE=TARGET.reshape((1, -1)),
                                                     TARGET_AS_LIST=TARGET.reshape((-1, 1)),
                                                     XTX=None,
                                                     XTX_INV=None,
                                                     has_intercept=has_intercept,
                                                     intercept_math=has_intercept,
                                                     regularization_factor=rglztn_fctr,
                                                     safe_matmul=False,
                                                     bypass_validation=True
            ).run()[4:]

            if score_method == 'R': best_value = abs(R_)
            elif score_method == 'Q': best_value = R2_
            elif score_method == 'A': best_value = R2_ADJ_
            elif score_method == 'F': best_value = F_

            best_value = best_value if isinstance(best_value, (float, int)) else 0

            del R_, R2_, R2_ADJ_, F_

            # END RUN A ONE-OFF MLRegression ON FULL DATA TO GET STARTING best_value #############################################

            AVAILABLE_COLUMNS = np.arange(0, len(WIP_DATA), dtype=np.int32)
            WIP_TRAIN_HOLDER = begtr.build_empty_gmlr_train_results(WIP_HEADER)
            WINNING_COLUMNS = AVAILABLE_COLUMNS.copy()

            conv_kill = 1
            pct_change = 0
            conv_end_method = 'KILL'
            ni_ctr_init = 0
            ni_ctr = 0
            # best_value --- ABOVE

            for b_itr, col_idx in enumerate(AVAILABLE_COLUMNS):

                print(f'Looking to drop column {b_itr+1}...')

                # CREATE AN ALLOWED_AVAILABLE COLUMNS TO ITERATE OVER THAT HOLDS THE COLUMNS THAT HAVENT BEEN USED YET

                BEST_SCORES_HOLDER = np.empty((len(WINNING_COLUMNS), 4), dtype=object)

                if col_idx == intcpt_col_idx:   # IF INTERCEPT, DONT TAKE IT OUT
                    ni_ctr_init += 1
                    continue

                for b_itr2, col_idx2 in enumerate(WINNING_COLUMNS):

                    if col_idx2 == intcpt_col_idx:
                        BEST_SCORES_HOLDER[b_itr2] = [0,0,0,0]
                        continue

                    COLUMN_BACKUP = WIP_DATA[b_itr2].copy()
                    WIP_DATA = np.delete(WIP_DATA, b_itr2, axis=0)

                    # returned from MLRegression
                    # xtx_determinant, self.COEFFS, PREDICTED, P_VALUES, R, R2, R2_ADJ, F
                    R_, R2_, R2_ADJ_, F_ = \
                        mlr.MLRegression(
                                         WIP_DATA.transpose(),
                                         'ROW',
                                         TARGET.reshape((-1,1)),
                                         'ROW',
                                         DATA_TRANSPOSE=WIP_DATA,
                                         TARGET_TRANSPOSE=TARGET.reshape((1,-1)),
                                         TARGET_AS_LIST=TARGET.reshape((-1,1)),
                                         XTX=None,
                                         XTX_INV=None,
                                         has_intercept=has_intercept,
                                         intercept_math=has_intercept,
                                         regularization_factor=rglztn_fctr,
                                         safe_matmul=False,     # True if bypass_validation is False
                                         bypass_validation=True
                    ).run()[4:]

                    BEST_SCORES_HOLDER[b_itr2] = (R_, R2_, R2_ADJ_, F_)

                    # PUT COLUMN BACK
                    WIP_DATA = np.insert(WIP_DATA, b_itr2, COLUMN_BACKUP, axis=0)

                del COLUMN_BACKUP

                best_idx = np.argwhere(BEST_SCORES_HOLDER[:, look_row] == np.max(BEST_SCORES_HOLDER[:, look_row]))[0][0]

                if score_method == 'R': score = BEST_SCORES_HOLDER[best_idx][0]
                elif score_method == 'Q': score = BEST_SCORES_HOLDER[best_idx][1]
                elif score_method == 'A': score = BEST_SCORES_HOLDER[best_idx][2]
                elif score_method == 'F': score = BEST_SCORES_HOLDER[best_idx][3]

                score = score if isinstance(score, (int, float)) else 0


                ni_ctr, gmlr_conv_kill, best_value, abort = \
                    ni.NoImprov(score, b_itr, ni_ctr_init, ni_ctr, conv_kill, best_value, pct_change,
                                'BACKWARD GMLR', conv_end_method=conv_end_method).max()

                if abort:
                    # DO NOT REMOVE A COLUMN FROM WINNING_COLUMNS
                    if len(WINNING_COLUMNS) <= MASTER_MAX_COLUMNS[0]: break
                    else: abort = False

                elif not abort:
                    lose_col_idx = WINNING_COLUMNS[best_idx]
                    # PERMANENTLY REMOVE lose_col_idx FROM TRIAL
                    WINNING_COLUMNS = WINNING_COLUMNS[WINNING_COLUMNS!=lose_col_idx]
                    WIP_DATA = np.delete(WIP_DATA, best_idx, axis=0)

                if len(WINNING_COLUMNS) - int(has_intercept) == 1:
                    print(f'\nBackward GMLR ended for only one column remaining')
                    break

            del best_idx, lose_col_idx, WIP_DATA, R_, R2_, R2_ADJ_, F_, BEST_SCORES_HOLDER, ni_ctr, ni_ctr_init, best_value, abort

            # STILL HAVE WINNING_COLUMNS, RUN LAZYAGG ON THEM TO MAKE TRAIN_RESULTS, START WITH FRESH WIP_DATA
            if has_intercept: WIP_DATA = np.insert(DATA.copy(), 0, 1, axis=0)
            else: WIP_DATA = DATA.copy()

            DataClass = mlo.MLObject(
                                     WIP_DATA,
                                     data_given_orientation,
                                     name='BACKWARD_DATA',
                                     return_orientation='AS_GIVEN',
                                     return_format='AS_GIVEN',
                                     bypass_validation=True,
                                     calling_module=gmn.get_module_name(str(sys.modules[__name__])),
                                     calling_fxn='GUARD TEST'
            )

            WINNING_HEADER = WIP_HEADER[0][WINNING_COLUMNS]
            WIP_TRAIN_HOLDER = WIP_TRAIN_HOLDER.iloc[WINNING_COLUMNS, :]

            WIP_TRAIN_HOLDER, COEFFS = \
                lagg.LazyAggGMLR(
                                 DataClass.return_columns(WINNING_COLUMNS, return_orientation='ROW', return_format='AS_GIVEN'),
                                 WINNING_HEADER,
                                 'ROW',
                                 TARGET.transpose(),
                                 'ROW',
                                 AVAILABLE_COLUMNS=np.arange(0, len(WINNING_COLUMNS)),
                                 max_columns=None,
                                 intcpt_col_idx=0 if has_intercept else None,  # WAS INSERTED INTO 0 SLOT IN DATA WHEN USED
                                 rglztn_fctr=rglztn_fctr,
                                 score_method=score_method,
                                 TRAIN_RESULTS=WIP_TRAIN_HOLDER,
                                 TARGET_TRANSPOSE=TARGET,
                                 TARGET_AS_LIST=TARGET.transpose(),
                                 data_run_orientation='ROW',
                                 target_run_orientation='ROW',
                                 conv_kill=None,
                                 pct_change=None,
                                 conv_end_method=None,
                                 bypass_validation=False
            ).run()[1:]

            # END CORE BACKWARD ALGORITHM ################################################################################
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

            del WIP_TRAIN_HOLDER

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

                                                    # EXP_WINNING_COLUMNS = EXP_WINNING_COLUMNS[:max_columns]
                                                    # EXP_TRAIN_RESULTS = EXP_TRAIN_RESULTS.iloc[:max_columns, :]
                                                    # END GET EXPECTEDS #############################################################################


                                                    ACT_WINNING_COLUMNS, ACT_TRAIN_RESULTS, COEFFS = \
                                                        BackwardGMLR(GIVEN_DATA,
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
                                                                     conv_kill=None, #1, #None,
                                                                     pct_change=0, #None,
                                                                     conv_end_method='KILL', #None,
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




