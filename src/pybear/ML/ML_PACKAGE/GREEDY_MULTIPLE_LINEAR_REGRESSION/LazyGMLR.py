import sys, inspect, time
import numpy as np, pandas as pd
import sparse_dict as sd
from general_data_ops import get_shape as gs
from debug import get_module_name as gmn
from MLObjects import MLObject as mlo
from ML_PACKAGE.MLREGRESSION import MLRegression as mlr
from ML_PACKAGE.GREEDY_MULTIPLE_LINEAR_REGRESSION import ApexGMLR as agmlr


# gmlr_score_type DOES NOT MATTER.  ADJUSTMENTS FOR RSQ_ADJ AND F WOULD BE THE SAME FOR EVERY COLUMN, SO RSQ_ADJ AND F
# WILL ALWAYS BE PROPORTIONAL TO R2.  R WOULD REQUIRE abs ADJUSTMENT, JUST TO STILL BE PROPORTIONAL TO R2.
# USING ONLY R2 FOR SORTING.




class LazyGMLR(agmlr.ApexGMLR):

    def __init__(self, DATA, DATA_HEADER, data_given_orientation, TARGET, target_given_orientation, AVAILABLE_COLUMNS=None,
             max_columns=None, intcpt_col_idx=None, rglztn_fctr=None, TRAIN_RESULTS=None, TARGET_TRANSPOSE=None,
             TARGET_AS_LIST=None, data_run_orientation='ROW', target_run_orientation='ROW', bypass_validation=None):

        this_module = gmn.get_module_name(str(sys.modules[__name__]))


        super().__init__(DATA, DATA_HEADER, data_given_orientation, TARGET, target_given_orientation,
                         AVAILABLE_COLUMNS=AVAILABLE_COLUMNS,max_columns=max_columns, intcpt_col_idx=intcpt_col_idx,
                         rglztn_fctr=rglztn_fctr, TRAIN_RESULTS=TRAIN_RESULTS, TARGET_TRANSPOSE=TARGET_TRANSPOSE,
                         TARGET_AS_LIST=TARGET_AS_LIST, data_run_orientation=data_run_orientation,
                         target_run_orientation=target_run_orientation, bypass_validation=bypass_validation,
                         calling_module=this_module)

        fxn = inspect.stack()[0][3]

    # END init ##########################################################################################################
    #####################################################################################################################
    #####################################################################################################################


    def core_run(self, DATA, DATA_HEADER, data_run_orientation, TARGET, TARGET_TRANSPOSE, TARGET_AS_LIST,
                target_run_orientation, AVAILABLE_COLUMNS, max_columns, intcpt_col_idx, rglztn_fctr, bypass_validation,
                this_module):

        fxn = inspect.stack()[0][3]

        LAZY_SCORES = np.zeros(len(AVAILABLE_COLUMNS), dtype=np.float64)     # FILLED RSQs FROM EACH COLUMN

        for lazy_itr, col_idx in enumerate(AVAILABLE_COLUMNS):

            actv_hdr = DATA_HEADER[col_idx]

            # IF ON FIRST ITR (INTCPT, IF PRESENT, MUST BE FIRST), SET ALL SCORES TO ZERO AND BYPASS MLR
            if col_idx == intcpt_col_idx:
                LAZY_SCORES[lazy_itr] = 0
                self.TRAIN_RESULTS.loc[actv_hdr, ('INDIV', 'R')] = 0
                self.TRAIN_RESULTS.loc[actv_hdr, ('INDIV', 'R2')] = 0
                self.TRAIN_RESULTS.loc[actv_hdr, ('INDIV', 'ADJ R2')] = 0
                self.TRAIN_RESULTS.loc[actv_hdr, ('INDIV', 'F')] = 0
                continue

            if (lazy_itr + 1) % 100 == 0:
                print(f'Running column {lazy_itr + 1} of {len(AVAILABLE_COLUMNS)}...')

            print(f'********************************************************************************************************')
            print(f'        BEAR Start MLRegression() IN {this_module}.{fxn}()'); t1 = time.time()

            # IF DATASET ORIGINALLY HAD INTCPT, USE IN EACH ITR OF MLReg (EXCEPT INTCPT TRIAL)... AND VICE VERSA
            # IF ON INTCPT (SHOULD BE FIRST) PASS, DONT DOUBLE UP INTCPT
            if intcpt_col_idx is None: COL_IDXS_TO_PULL = [col_idx]
            elif not intcpt_col_idx is None: COL_IDXS_TO_PULL = [col_idx, intcpt_col_idx]

            ColumnPuller = mlo.MLObject(
                                        DATA,
                                        data_run_orientation,
                                        name='DATA',
                                        return_orientation='AS_GIVEN',
                                        return_format='AS_GIVEN',
                                        bypass_validation=bypass_validation,
                                        calling_module=this_module,
                                        calling_fxn=fxn
            )

            # IF BATCH_DATA IS COLUMN, SINCE HAVE TO TRANSPOSE TO ROW ANYWAY, GET TRIAL DATA TRANSPOSE FROM COLUMN
            # FORMAT THEN BUILD TRIAL DATA AS ROW. IF DATA IS ROW, JUST GET ROW AND LET MLRegression DEAL WITH IT.
            if data_run_orientation == 'COLUMN':
                LAZY_TRIAL_AS_COLUMN = ColumnPuller.return_columns(COL_IDXS_TO_PULL,
                                                                   return_orientation='COLUMN',
                                                                   return_format='AS_GIVEN')
                LAZY_TRIAL_AS_ROW = ColumnPuller.return_columns(COL_IDXS_TO_PULL,
                                                                return_orientation='ROW', return_format='AS_GIVEN')

            elif data_run_orientation == 'ROW':
                LAZY_TRIAL_AS_COLUMN = None

                LAZY_TRIAL_AS_ROW = ColumnPuller.return_columns(COL_IDXS_TO_PULL,
                                                                return_orientation='ROW', return_format='AS_GIVEN')

            del ColumnPuller


            # RETURNED FROM MLRegression
            # XTX_determinant, self.COEFFS, PREDICTED, P_VALUES, r, R2, R2_adj, F
            R_, R2_, R2_ADJ_, F_ = \
                mlr.MLRegression(LAZY_TRIAL_AS_ROW,
                                 data_run_orientation,
                                 TARGET,
                                 target_run_orientation,
                                 DATA_TRANSPOSE=LAZY_TRIAL_AS_COLUMN,
                                 TARGET_TRANSPOSE=TARGET_TRANSPOSE,
                                 TARGET_AS_LIST=TARGET_AS_LIST,
                                 XTX=None, XTX_INV=None,
                                 has_intercept=False if intcpt_col_idx is None else True,
                                 intercept_math=False if intcpt_col_idx is None else True,
                                 regularization_factor=rglztn_fctr,
                                 safe_matmul=not bypass_validation,  # True if bypass_validation is False
                                 bypass_validation=bypass_validation
                                 ).run()[4:]

            del COL_IDXS_TO_PULL, LAZY_TRIAL_AS_ROW, LAZY_TRIAL_AS_COLUMN

            # 6/4/23 KEEP score SEPARATE FROM TRAIN_RESULTS FOR MASKING OF AVAILABLE_COLUMNS
            score = R2_

            # 1-17-2022 -- THIS ALGORITHM IS REACHING "INTERCEPT" COLUMN (WHEN USED) AND RETURNING nan, WHICH IS
            # CAUSING THE > FUNCTIONALITY OF SORT TO FAIL.  ALSO, IF MLRegression BLOWS UP, IT RETURNS nan. RETURN 0 INSTEAD.
            score = score if isinstance(score, (float, int)) else 0


            print(f'        BEAR MLRegression in {this_module}.{fxn}() Done. total time = {(time.time() - t1)} sec')
            print(f'********************************************************************************************************')

            LAZY_SCORES[lazy_itr] = score
            self.TRAIN_RESULTS.loc[actv_hdr, ('INDIV', 'R')] = R_
            self.TRAIN_RESULTS.loc[actv_hdr, ('INDIV', 'R2')] = R2_
            self.TRAIN_RESULTS.loc[actv_hdr, ('INDIV', 'ADJ R2')] = R2_ADJ_
            self.TRAIN_RESULTS.loc[actv_hdr, ('INDIV', 'F')] = F_


        del R_, R2_, R2_ADJ_, F_, score, actv_hdr


        # ########################################################################################################################
        # SORT & SELECT #####################################################################################################
        print(f'\nProceeding to sort and select of GMLR winners...')

        # SORT ALL SCORES DESCENDING, BUT OBSERVE NEED TO SORT HEADER ASCENDING WHEN VALUES ARE EQUAL, SO USE A SORTER DF
        SORTER_DF = pd.DataFrame(data=np.vstack((np.round(LAZY_SCORES, 6),
                                                 self.TRAIN_RESULTS.index.to_numpy())).transpose(),
                                 columns=['VALUES','HEADER'], dtype=object
                                ).sort_values(by=['VALUES','HEADER'], ascending=[False, True])
        MASTER_SORT_DESC = SORTER_DF.index.to_numpy()

        del LAZY_SCORES, SORTER_DF

        # IF DATASET HAD INTERCEPT FORCE IT INTO WINNERS, MOVE IT TO FIRST IN MASTER_SORT_DESC NO MATTER WHAT ITS SCORE
        # WAS. IF PRESENT, ITS IDX WAS ZERO IN AVAILABLE_COLUMNS, SO MOVE THE ZERO TO FIRST IN MASTER_SORT_DESC.
        # ITS BETTER FOR AGGLOMERATIVE MLRegression IF INTERCEPT IS INCLUDED IF WAS IN DATA AND IS RUN FIRST, SO
        # THAT ALL AGGLOMERATIONS SEE INTERCEPT NATURALLY WITHOUT CODE GYMNASTICS
        if not intcpt_col_idx is None:
            MASTER_SORT_DESC = np.insert(MASTER_SORT_DESC[MASTER_SORT_DESC != 0], 0, 0, axis=0)

        # AFTER COLLECTING ALL INDIVIDUAL SCORES, THE WINNERS HERE BECOME THE AVAILABLE_COLUMNS FOR FULL, IF NOT BYPASSED
        self.WINNING_COLUMNS = AVAILABLE_COLUMNS[MASTER_SORT_DESC][:max_columns]

        del AVAILABLE_COLUMNS, MASTER_SORT_DESC


        # BEAR THIS STEP IS THE DIFFERENCE-MAKER BETWEEN HAVING A CLOSED LazyGMLR AND ONE THAT MUST OPERATE AS PART OF GMLRCoreRunCode
        self.TRAIN_RESULTS = self.TRAIN_RESULTS.iloc[self.WINNING_COLUMNS, :]

        # ########################################################################################################################
        # RUN ONE-SHOT ML REGRESSION ON WINNING COLUMNS ###################################################@######################

        print(f'\nCalculating overall "{["no intercept" if intcpt_col_idx is None else "intercept"][0]}" style '
              f'ml regression results for lazy winners and building results table... \n')


        DataOrienter = mlo.MLObject(DATA, data_run_orientation, name='WINNING_DATA',
                                    return_orientation='AS_GIVEN', return_format='AS_GIVEN',
                                    bypass_validation=bypass_validation, calling_module=this_module,
                                    calling_fxn=fxn
                                    )

        if data_run_orientation=='ROW':
            WINNING_DATA_AS_COLUMN = None

            WINNING_DATA_AS_ROW = DataOrienter.return_columns(self.WINNING_COLUMNS, return_orientation='ROW',
                                                                return_format='AS_GIVEN')

        elif data_run_orientation=='COLUMN':
            WINNING_DATA_AS_COLUMN = DataOrienter.return_columns(self.WINNING_COLUMNS, return_orientation='COLUMN',
                                                                return_format='AS_GIVEN')
            WINNING_DATA_AS_ROW = DataOrienter.return_columns(self.WINNING_COLUMNS, return_orientation='ROW',
                                                                return_format='AS_GIVEN')

        del DataOrienter


        # RETURNED FROM MLRegression
        # XTX_determinant, self.COEFFS, PREDICTED, P_VALUES, r, R2, R2_adj, F
        DUM, self.COEFFS, DUM, P_VALUES, R_, R2_, R2_ADJ_, F_ = \
            mlr.MLRegression(
                             WINNING_DATA_AS_ROW,
                             'ROW',
                             TARGET if target_run_orientation == 'ROW' else TARGET_TRANSPOSE,
                             'ROW',
                             DATA_TRANSPOSE=WINNING_DATA_AS_COLUMN,
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


        del WINNING_DATA_AS_ROW, WINNING_DATA_AS_COLUMN, TARGET, TARGET_TRANSPOSE, TARGET_AS_LIST, DUM

        self.TRAIN_RESULTS.iloc[0, 6] = R_
        self.TRAIN_RESULTS.iloc[0, 7] = R2_
        self.TRAIN_RESULTS.iloc[0, 8] = R2_ADJ_
        self.TRAIN_RESULTS.iloc[0, 9] = F_
        self.TRAIN_RESULTS[('FINAL', 'COEFFS')] = self.COEFFS
        self.TRAIN_RESULTS[('FINAL', 'p VALUE')] = P_VALUES

        del P_VALUES, R_, R2_, R2_ADJ_, F_

        # END RUN ONE-SHOT ML REGRESSION ON WINNING COLUMNS ###################################################@##################
        # ########################################################################################################################














if __name__ == '__main__':

    from MLObjects.TestObjectCreators.SXNL import CreateSXNL as csxnl
    from MLObjects.SupportObjects import master_support_object_dict as msod
    from data_validation import validate_user_input as vui
    from ML_PACKAGE.GREEDY_MULTIPLE_LINEAR_REGRESSION import build_empty_gmlr_train_results as begtr
    from general_sound import winlinsound as wls


    DATA = pd.read_csv(r'C:\Users\Bill\Documents\WORK STUFF\RESUME\1 - OTHER\SRP\beer_reviews.csv',
                      nrows=100,
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



    # GIVEN TO LazyGMLR tests, NOT CreateSXNL!
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


    # DELETE AN ARBITRARY BIN COLUMN TO GET RID OF MULTI-COLINEARTIY
    DATA = np.delete(DATA, 4, axis=0)
    DATA_HEADER = np.delete(DATA_HEADER, 4, axis=1)


    MASTER_BYPASS_VALIDATION = [True, False]
    MASTER_MAX_COLUMNS = [len(DATA)+10, 2]
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

    for int_idx, has_intercept in enumerate(MASTER_INTCPT_GIVEN):
        if has_intercept:
            WIP_DATA = np.insert(DATA, 0, 1, axis=0)
            WIP_HEADER = np.hstack(([['INTERCEPT']] if has_intercept else [[]], DATA_HEADER.copy()))
        else:
            WIP_DATA = DATA
            WIP_HEADER = DATA_HEADER.copy()

        for rgz_idx, rglztn_fctr in enumerate(MASTER_RGLZTN_FCTR):
            WIP_SCORE_HOLDER = np.empty(0, dtype=np.float64)
            WIP_TRAIN_HOLDER = begtr.build_empty_gmlr_train_results(WIP_HEADER)

            for col_idx, COLUMN in enumerate(WIP_DATA):

                column_name = WIP_HEADER[0][col_idx]

                if column_name == 'INTERCEPT':
                    WIP_SCORE_HOLDER = np.insert(WIP_SCORE_HOLDER, len(WIP_SCORE_HOLDER), 0, axis=0)
                    WIP_TRAIN_HOLDER.iloc[col_idx, 0:4] = (0, 0, 0, 0)
                    continue

                if has_intercept: COLUMN = np.insert(COLUMN.reshape((1,-1)), 1, 1, axis=0)
                else: COLUMN = COLUMN.reshape((1,-1))

                R_, R2_, R2_ADJ_, F_ = mlr.MLRegression(COLUMN.transpose(),
                                                         'ROW',
                                                         TARGET.reshape((-1,1)),
                                                         'ROW',
                                                         DATA_TRANSPOSE=COLUMN,
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

                WIP_SCORE_HOLDER = np.insert(WIP_SCORE_HOLDER, len(WIP_SCORE_HOLDER), R2_, axis=0)
                WIP_TRAIN_HOLDER.iloc[col_idx, 0:4] = (R_, R2_, R2_ADJ_, F_)

            AVAILABLE_COLUMNS = np.fromiter(range(len(WIP_DATA)), dtype=np.int32)

            SORTER_DF = pd.DataFrame(data=np.vstack((np.round(WIP_SCORE_HOLDER, 6),
                                                     WIP_TRAIN_HOLDER.index.to_numpy())).transpose(),
                                     columns=['VALUES', 'HEADER'], dtype=object
                                     ).sort_values(by=['VALUES', 'HEADER'], ascending=[False, True])
            WIP_ARGSORT = SORTER_DF.index.to_numpy()

            del WIP_SCORE_HOLDER, SORTER_DF

            if has_intercept: WIP_ARGSORT = np.insert(WIP_ARGSORT[WIP_ARGSORT!=0], 0, 0, axis=0)

            if rgz_idx==0 and int_idx==0:   # HAS INTERCEPT
                RGZ_0_INT_0_COL_KEY = AVAILABLE_COLUMNS.copy()[WIP_ARGSORT]
                RGZ_0_INT_0_TRAIN_KEY = WIP_TRAIN_HOLDER.copy().iloc[WIP_ARGSORT, :]
            elif rgz_idx==0 and int_idx==1:
                RGZ_0_INT_1_COL_KEY = AVAILABLE_COLUMNS.copy()[WIP_ARGSORT]
                RGZ_0_INT_1_TRAIN_KEY = WIP_TRAIN_HOLDER.copy().iloc[WIP_ARGSORT, :]
            elif rgz_idx==1 and int_idx==0:   # HAS INTERCEPT
                RGZ_1_INT_0_COL_KEY = AVAILABLE_COLUMNS.copy()[WIP_ARGSORT]
                RGZ_1_INT_0_TRAIN_KEY = WIP_TRAIN_HOLDER.copy().iloc[WIP_ARGSORT, :]
            elif rgz_idx==1 and int_idx==1:
                RGZ_1_INT_1_COL_KEY = AVAILABLE_COLUMNS.copy()[WIP_ARGSORT]
                RGZ_1_INT_1_TRAIN_KEY = WIP_TRAIN_HOLDER.copy().iloc[WIP_ARGSORT, :]

    del WIP_DATA, WIP_ARGSORT, WIP_TRAIN_HOLDER, AVAILABLE_COLUMNS

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
                                                            EXP_WINNING_COLUMNS = RGZ_0_INT_0_COL_KEY
                                                            EXP_TRAIN_RESULTS = RGZ_0_INT_0_TRAIN_KEY
                                                        elif rglztn_fctr==MASTER_RGLZTN_FCTR[1]:
                                                            EXP_WINNING_COLUMNS = RGZ_1_INT_0_COL_KEY
                                                            EXP_TRAIN_RESULTS = RGZ_1_INT_0_TRAIN_KEY
                                                    else:
                                                        if rglztn_fctr==MASTER_RGLZTN_FCTR[0]:
                                                            EXP_WINNING_COLUMNS = RGZ_0_INT_1_COL_KEY
                                                            EXP_TRAIN_RESULTS = RGZ_0_INT_1_TRAIN_KEY
                                                        elif rglztn_fctr==MASTER_RGLZTN_FCTR[1]:
                                                            EXP_WINNING_COLUMNS = RGZ_1_INT_1_COL_KEY
                                                            EXP_TRAIN_RESULTS = RGZ_1_INT_1_TRAIN_KEY

                                                    EXP_WINNING_COLUMNS = EXP_WINNING_COLUMNS[:max_columns]
                                                    EXP_TRAIN_RESULTS = EXP_TRAIN_RESULTS.iloc[:max_columns, :]
                                                    # END GET EXPECTEDS #############################################################################


                                                    ACT_WINNING_COLUMNS, ACT_TRAIN_RESULTS, COEFFS = \
                                                        LazyGMLR(
                                                                    GIVEN_DATA,
                                                                    GIVEN_DATA_HEADER,
                                                                    data_orient,
                                                                    GIVEN_TARGET,
                                                                    target_orient,
                                                                    AVAILABLE_COLUMNS=AVAILABLE_COLUMNS,
                                                                    max_columns=max_columns,
                                                                    intcpt_col_idx=intcpt_col_idx,
                                                                    rglztn_fctr=rglztn_fctr,
                                                                    TRAIN_RESULTS=GIVEN_TRAIN_RESULTS,
                                                                    TARGET_TRANSPOSE=GIVEN_TARGET_TRANSPOSE,
                                                                    TARGET_AS_LIST=GIVEN_TARGET_AS_LIST,
                                                                    data_run_orientation='ROW', #'AS_GIVEN',
                                                                    target_run_orientation='ROW', #'AS_GIVEN',
                                                                    bypass_validation=bypass_validation
                                                        ).run()


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

                                                    try:
                                                        ACT_TRAIN_RESULTS.to_numpy()[:, :4].astype(np.float64) # SEE IF THIS EXCEPTS
                                                        # THIS WILL EXCEPT IF ANY NON-NUMS (AS WOULD HAPPEN IF NON-INTERCEPT MLR)
                                                        PASS_TRAIN_RESULTS_TEST = np.allclose(
                                                            ACT_TRAIN_RESULTS.to_numpy()[:, :4].astype(np.float64),
                                                            EXP_TRAIN_RESULTS.to_numpy()[:, :4].astype(np.float64)
                                                        )
                                                    except:
                                                        PASS_TRAIN_RESULTS_TEST = np.allclose(
                                                            ACT_TRAIN_RESULTS.to_numpy()[:, [1,3]].astype(np.float64),
                                                            EXP_TRAIN_RESULTS.to_numpy()[:, [1,3]].astype(np.float64)
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


    print(f'\n\033[92m*** ALL LazyGMLR TESTS PASSED ***\033[0m')
    for _ in range(3): wls.winlinsound(888, 500); time.sleep(1)

























