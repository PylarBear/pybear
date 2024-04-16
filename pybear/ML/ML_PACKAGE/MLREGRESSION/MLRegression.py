import sys, time
import numpy as np, pandas as pd
import statsmodels.api as sm
import sparse_dict as sd
from debug import get_module_name as gmn
from data_validation import arg_kwarg_validater as akv
from ML_PACKAGE._data_validation import list_dict_validater as ldv
from general_data_ops import get_shape as gs
from linear_algebra import XTX_determinant as xd
from MLObjects.ObjectOrienter import MLObjectOrienter as mloo

# BEAR 6/1/2023  THINK ABOUT 3 THINGS.....
# 1) TAKING EVERYTHING OUT OF run() AND PUT IT IN init LIKE MutualInformation
# 2) LEAVE run() BUT HAVE init VALIDATE/ORIENT TARGET ET AL AND PASS DATA ET AL TO run()
# 3) LEAVE run() BUT HAVE init VALIDATE/ORIENT DATA ET AL AND PASS TARGET ET AL TO run()



# SEE NOTES AT BOTTOM
# DATA IS PROCESSED AS [ [] = ROWS], TARGET IS PROCESSED AS [ [] = ROWS]
class MLRegression:
    def __init__(self,
                 DATA,
                 data_given_orientation,
                 TARGET,
                 target_given_orientation,
                 DATA_TRANSPOSE=None,
                 TARGET_TRANSPOSE=None,
                 TARGET_AS_LIST=None,
                 XTX=None,
                 XTX_INV=None,
                 has_intercept=None,
                 intercept_math=None,
                 regularization_factor=0,
                 safe_matmul=True,
                 bypass_validation=None):
        
        self.bear_time_display = lambda t0: round((time.time() - t0), 2)

        self.this_module = gmn.get_module_name(str(sys.modules[__name__]))
        fxn = '__init__'

        bypass_validation = akv.arg_kwarg_validater(bypass_validation, 'bypass_validation', [True, False, None],
                                                         self.this_module, fxn, return_if_none=False)

        if bypass_validation:
            self.is_list = isinstance(DATA, (np.ndarray, tuple, list))
            self.is_dict = isinstance(DATA, dict)

        elif not bypass_validation:
            data_given_orientation = akv.arg_kwarg_validater(data_given_orientation, 'data_given_orientation', ['ROW', 'COLUMN'],
                                                             self.this_module, fxn)
            target_given_orientation = akv.arg_kwarg_validater(target_given_orientation, 'target_given_orientation', ['ROW', 'COLUMN'],
                                                             self.this_module, fxn)

            data_format, DATA = ldv.list_dict_validater(DATA, 'DATA')
            data_transpose_format, DATA_TRANSPOSE = ldv.list_dict_validater(DATA_TRANSPOSE, 'DATA_TRANSPOSE')

            if data_format is None and data_transpose_format is None:
                raise Exception(f'\n*** {self.this_module}() MUST HAVE AT LEAST ONE DATA OBJECT AS ARGUMENT ***\n')
            elif not data_format is None and not data_transpose_format is None:
                if not data_format==data_transpose_format:
                    raise Exception(f'\n*** {self.this_module}() DATA AND DATA_TRANSPOSE FORMATS MUST BE THE SAME ***\n')

            self.is_list = data_format == 'ARRAY' if not data_format is None else data_transpose_format == 'ARRAY'
            self.is_dict = data_format == 'SPARSE_DICT' if not data_format is None else data_transpose_format == 'SPARSE_DICT'

            del data_format, data_transpose_format

            TARGET = ldv.list_dict_validater(TARGET, 'TARGET')[1]
            TARGET_TRANSPOSE = ldv.list_dict_validater(TARGET_TRANSPOSE, 'TARGET_TRANSPOSE')[1]
            TARGET_AS_LIST = ldv.list_dict_validater(TARGET_AS_LIST, 'TARGET_AS_LIST')[1]

            # VALIDATION OF NON-MULTICLASS TARGET AND OBJ/OBJ_TRANSPOSE CONGRUENCY IS HANDLED BY ObjectOrienter WHEN NOT BYPASS VALIDATION

        ########################################################################################################################
        # ORIENT DATA AND TARGET OBJECTS #######################################################################################

        # DATA IS ALWAYS [[]=ROW], COULD BE ARRAY OR SD
        # DATA_TRANSPOSE IS ALWAYS [[]=COLUMN] (REMEMBER CRAZY RULES FOR TRANSPOSES, HAVE TO SAY 'ROW' FOR THIS)
        # TARGET IS ARRAY AND ALWAYS [[]=ROW]  (BECAUSE WE HAVE sd.hybrid_matmul NOW)
        # TARGET_TRANSPOSE IS ALWAYS ARRAY AND ALWAYS [[]=ROW] (REMEMBER CRAZY RULES FOR TRANSPOSES)
        # TARGET_AS_LIST ALWAYS AS ARRAY, ALWAYS [[]=ROW]

        data_run_format='AS_GIVEN'
        data_run_orientation='ROW'  # 5/5/23 THIS CANNOT CHANGE BECAUSE OF matmuls AND statsmodels!!!!
        target_run_format='ARRAY' # 4/16/23 CHANGED FROM DEPENDENCE ON is_list/is_dict TO ALWAYS ARRAY
        target_run_orientation='ROW'  # 5/5/23 THIS CANNOT CHANGE BECAUSE OF matmuls AND statsmodels!!!!
        xtx_run_format='ARRAY'    # ALWAYS RETURN AS ARRAY, USE sd.hybrid_matmul IF is_dict IS True
        xtx_inv_run_format='ARRAY'


        OrienterClass = mloo.MLObjectOrienter(
                         DATA=DATA,
                         data_given_orientation=data_given_orientation,
                         data_return_orientation=data_run_orientation,
                         data_return_format=data_run_format,

                         DATA_TRANSPOSE=DATA_TRANSPOSE,
                         data_transpose_given_orientation=data_given_orientation,
                         data_transpose_return_orientation=data_run_orientation,
                         data_transpose_return_format=data_run_format,

                         XTX=XTX,
                         xtx_return_format=xtx_run_format,

                         XTX_INV=XTX_INV,
                         xtx_inv_return_format=xtx_inv_run_format,

                         target_is_multiclass=False,
                         TARGET=TARGET,
                         target_given_orientation=target_given_orientation,
                         target_return_orientation=target_run_orientation,
                         target_return_format=target_run_format,

                         TARGET_TRANSPOSE=TARGET_TRANSPOSE,
                         target_transpose_given_orientation=target_given_orientation,
                         target_transpose_return_orientation=target_run_orientation,
                         target_transpose_return_format=target_run_format,

                         TARGET_AS_LIST=TARGET_AS_LIST,
                         target_as_list_given_orientation=target_given_orientation,
                         target_as_list_return_orientation=target_run_orientation,

                        # 4/16/23 DONT RETURN XTX_INV HERE, DO THAT LATER TO ALLOW FOR SPECIAL HANDLING IF SINGULAR
                         RETURN_OBJECTS=['DATA', 'DATA_TRANSPOSE', 'TARGET', 'TARGET_TRANSPOSE', 'TARGET_AS_LIST', 'XTX'],

                         bypass_validation=bypass_validation,   # CATCHES A MULTICLASS TARGET, CHECKS TRANSPOSES
                         calling_module=self.this_module,
                         calling_fxn=fxn
        )


        self.DATA = OrienterClass.DATA
        self.DATA_TRANSPOSE = OrienterClass.DATA_TRANSPOSE
        self.TARGET = OrienterClass.TARGET
        self.target_run_format = OrienterClass.target_current_format
        self.TARGET_TRANSPOSE = OrienterClass.TARGET_TRANSPOSE
        self.TARGET_AS_LIST = OrienterClass.TARGET_AS_LIST
        self.XTX = OrienterClass.XTX

        self.XTX_INV = XTX_INV   # DONT GET IT FROM ObjectOrienter, JUST TAKE ALONG IF PASSED AS KWARG

        del OrienterClass

        # END ORIENT DATA AND TARGET OBJECTS ###################################################################################
        ########################################################################################################################

        self.rows, self.columns = gs.get_shape('DATA', DATA, data_run_orientation)

        # CURRENTLY ONLY BEING GENERATED FOR CUTOFFS, WHICH ARE NOT BEING USED ###################################################
        # self.sparsity_cutoff = 85
        # self.size_cutoff = 200e6
        # END CURRENTLY ONLY BEING GENERATED FOR CUTOFFS, WHICH ARE NOT BEING USED ################################################

        # MANAGE INTERCEPT OPTIONS ###############################################################################################
        if has_intercept is None:
            for col_idx in range(self.columns):
                if np.min(self.DATA[:, col_idx]) == np.max(self.DATA[:, col_idx]):
                    self.has_intercept = True
                    break
            else: self.has_intercept = False
        else: self.has_intercept = has_intercept

        if intercept_math is None: self.intercept_math = self.has_intercept
        else: self.intercept_math = has_intercept

        # VALIDATE USER ENTRY FOR has_intercept AND intercept_math
        _ = int(self.has_intercept not in [True, False])
        __ = int(self.intercept_math not in [True, False])
        if _ + __ != 0:
            raise Exception(f'INVALID ' +
                           f'{_ * "self.has_intercept "}' +
                           f'{_ * ("(" + str(self.has_intercept) + ") ")}' +
                           f'{__ * "self.intercept_math "}' +
                           f'{__ * ("(" + str(self.intercept_math) + ") ")}' +
                           f'IN RegressionStatistics. CAN ONLY BE True OR False.'
            )
        # END MANAGE INTERCEPT OPTIONS ###############################################################################################

        self.regularization_factor = regularization_factor

        self.safe_matmul = safe_matmul

        self.y_avg = np.average(self.TARGET_AS_LIST.astype(float))

        # FOR F AND RSQ_ADJ ##############################################################################################
        self.df_model = max(1, self.columns - int(self.has_intercept))   # NOT EXACTLY SURE ABT df_model WHEN ONLY ONE COLUMN
        self.df_error = self.rows - self.columns
        self.df_total = self.rows - 1
        # END FOR F AND RSQ_ADJ ##############################################################################################

        # PLACEHOLDERS
        self.XTX_determinant = 0














    def return_fxn(self):
        return self.XTX_determinant, self.COEFFS, self.PREDICTED, self.P_VALUES, self.r, self.R2, self.R2_adj, self.F


    def run(self):

        while True:  # JUST TO ENABLE SKIP-OUT IF XTX NOT INVERTIBLE

            if isinstance(self.XTX_INV, dict): self.XTX_INV = sd.unzip_to_ndarray_float64(self.XTX_INV)[0]

            if not self.XTX_INV is None and self.regularization_factor==0:
                # XTX_INV WAS GIVEN SO XTX WAS OBVIOUSLY INVERTIBLE AND JUST CARRY THRU TO REGRESSION
                pass
            else:  # (XTX_INV WAS NOT GIVEN AND XTX WAS BUILT FROM DATA) OR (XTX_INV WAS GIVEN AND CANT USE BECAUSE OF RIDGE)
                try:  # TEST IF XTX CAN BE INVERTED, IF YES, PROCEED TO CALC, IF NOT, SKIP OUT, RETURN return_on_error()
                    # XTX SHOULD HAVE BEEN BUILT BY ObjectOrienter, MAKE XTX_INV HERE, IGNORE PASSED XTX_INV IF USING RIDGE

                    if self.regularization_factor != 0:  # XTX SHOULD COME OUT OF ObjectOrienter AS ARRAY IF GOING IN HERE
                        self.XTX = np.add(self.XTX, self.regularization_factor * np.identity(len(self.XTX), dtype=np.int8))

                    print(f'            BEAR INVERT XTX...'); t0 = time.time()
                    self.XTX_INV = np.linalg.inv(self.XTX)   # linalg.inv CAN ONLY TAKE ARRAY & RETURN ARRAY, NOT SD
                    print(f'            INVERT XTX Done. time = {self.bear_time_display(t0)} sec')
                except:  # IF UNABLE TO GET INVERSE OF XTX, SKIP ALL CALCULATIONS AND RETURN 'nan' FOR EVERYTHING
                    if np.linalg.LinAlgError:
                        # self.XTX_determinant, self.COEFFS, self.PREDICTED, self.P_VALUES, self.r, self.R2, self.R2_adj, self.F
                        if self.is_list:
                            return 'nan', ['nan' for _ in range(self.columns)], ['nan' for _ in range(self.rows)], \
                                   ['nan' for _ in range(self.columns)], 'nan', 'nan', 'nan', 'nan'

                    else: print(f'\n *** EXCEPTION OTHER THAN LinAlgError WHEN TRYING TO INVERT XTX IN MLRegression *** \n')

            self.XTX_determinant = xd.XTX_determinant(XTX_AS_ARRAY_OR_SPARSEDICT=self.XTX, name=f'XTX_determinant',
                                   module=f'general_data_ops.MLRegression', fxn='run', quit_on_exception=True)[0]
            del self.XTX

            if self.is_list:   # XTX_INV IS ALWAYS ARRAY
                # SAFE / UNSAFE MATMUL FOR NP IS THE SAME
                print(f'            BEAR TEST SAFE/UNSAFE NP_MATMUL...'); t0 = time.time()
                self.COEFFS = np.matmul(
                                        np.matmul(self.XTX_INV.astype(np.float64),
                                                self.DATA_TRANSPOSE.astype(np.float64),
                                                dtype=np.float64
                                        ),
                                        self.TARGET_AS_LIST.astype(np.float64),
                                        dtype=np.float64
                )
                self.PREDICTED = np.matmul(self.DATA.astype(np.float64), self.COEFFS.astype(np.float64), dtype=np.float64).transpose()[0]
                self.COEFFS = self.COEFFS.transpose()[0]
                print(f'            BEAR TEST SAFE/UNSAFE NP MATMUL Done. time = {self.bear_time_display(t0)} sec')

            elif self.is_dict:   # XTX_INV IS ALWAYS ARRAY
                if self.safe_matmul:
                    print(f'            BEAR TEST DICT SAFE SD_MATMUL...'); t0 = time.time()
                    self.COEFFS = sd.hybrid_matmul(
                                                    sd.hybrid_matmul(self.XTX_INV,   # XTX_INV SHOULD ALWAYS BE ARRAY HERE
                                                                      self.DATA_TRANSPOSE,
                                                                      LIST_OR_DICT2_TRANSPOSE=self.DATA,
                                                                      return_as='SPARSE_DICT',
                                                                      return_orientation='ROW'
                                                    ),
                                                    self.TARGET,
                                                    LIST_OR_DICT2_TRANSPOSE=self.TARGET_TRANSPOSE,
                                                    return_as='ARRAY',
                                                    return_orientation='ROW'
                    )
                    self.PREDICTED = sd.hybrid_matmul(self.DATA, self.COEFFS, return_as='ARRAY', return_orientation='COLUMN')
                    self.COEFFS = self.COEFFS.transpose()[0]  # FOR return_fxn
                    print(f'            BEAR TEST SAFE SD_MATMUL Done. time = {self.bear_time_display(t0)} sec')

                elif not self.safe_matmul:
                    print(f'            BEAR TEST DICT UNSAFE SD_MATMUL...'); t0 = time.time()

                    self.COEFFS = sd.core_hybrid_matmul(
                                                        sd.core_hybrid_matmul(self.XTX_INV,   # XTX_INV SHOULD ALWAYS BE ARRAY HERE
                                                                                self.DATA_TRANSPOSE,
                                                                                LIST_OR_DICT2_TRANSPOSE=self.DATA,
                                                                                return_as='SPARSE_DICT',
                                                                                return_orientation='ROW'
                                                        ),
                                                        self.TARGET,
                                                        LIST_OR_DICT2_TRANSPOSE=self.TARGET_TRANSPOSE,
                                                        return_as='ARRAY',
                                                        return_orientation='ROW'
                    )
                    self.PREDICTED = sd.core_hybrid_matmul(self.DATA, self.COEFFS, return_as='ARRAY', return_orientation='COLUMN')
                    self.COEFFS = self.COEFFS.transpose()[0]  # FOR return_fxn
                    print(f'            BEAR TEST UNSAFE SD_MATMUL Done. time = {self.bear_time_display(t0)} sec\n')

            # USE statsmodels TO GET p-VALUES FOR NON-RIDGE CALCULATIONS (ONLY REASON USING statsmodels AT THIS TIME)

            print(f'            BEAR TEST FINAL CALCS....'); t0 = time.time()

            if self.regularization_factor==0:
                if self.is_dict: self.DATA = sd.unzip_to_ndarray_float64(self.DATA)[0] # DONT NEED self.DATA AFTER THIS
                ols_results = sm.OLS(self.TARGET_AS_LIST.astype(float), self.DATA.astype(float),
                                     missing='drop',hasconst=self.has_intercept).fit()

                self.P_VALUES = ols_results.pvalues  # hasconst CAN BE WHATEVER AND p-VALUES WILL BE CORRECT
            else:
                self.P_VALUES = ['-' for _ in range(self.columns)]

                # MAYBE THIS IS WRONG WHEN USING "NO INTERCEPT" DATA AND USING "INTERCEPT MATH"???
                # THINKING THIS SHOULD BE OK, INT MATH / NON-INT MATH SHOULD ONLY BE IMPACTING SSModel (DOES SSModel IMPACT p-VALUES??)

            del self.DATA, self.DATA_TRANSPOSE

            SSReg = np.sum((self.PREDICTED - int(self.intercept_math) * self.y_avg) ** 2)
            SSErr = np.sum((self.TARGET_AS_LIST.transpose() - self.PREDICTED) ** 2)

            if self.has_intercept:
                # CALCULATING r, USE int(self.intercept) BOOLEAN TO CONTROL INTERCEPT / NO-INTERCEPT MATH
                Xi_MINUS_PRED_AVG = np.array(self.PREDICTED - int(self.intercept_math) * np.average(self.PREDICTED), dtype=np.float64)
                Yi_MINUS_Y_AVG = np.array(self.TARGET_AS_LIST.transpose() - int(self.intercept_math) * self.y_avg, dtype=np.float64)
                NUMERATOR = np.sum(Xi_MINUS_PRED_AVG * Yi_MINUS_Y_AVG)
                DENOMINATOR = np.sqrt(np.sum(np.power(Xi_MINUS_PRED_AVG, 2)) * np.sum(np.power(Yi_MINUS_Y_AVG, 2)))
                with np.errstate(all='ignore'):
                    self.r = float(NUMERATOR / DENOMINATOR)

                self.R2 = float(self.r**2)

                # McNemar's formula: R2_adj = 1−(1−R2)(rows−1)/(rows−columns−1), Excel & statsmodels APPEAR TO BE USING PERVERSIONS OF THIS
                try: self.R2_adj = float(1 - (1 - self.R2) * (self.rows - 1) / (self.rows - self.columns))
                except: self.R2_adj = 'NaN'

                # 1-26-22 WHOLE LOT OF FUDGE IN F TO GET AGREEMENT BETWEEN Excel, statsmodels, & THIS FOR INTERCEPT / NON-INTERCEPT
                with np.errstate(all='ignore'):
                    # USE BOOLEAN self.intercept TO TOGGLE CORRECTED / UNCORRECTED SSReg
                    self.F = float((SSReg / self.df_model) / (SSErr / (self.rows - self.df_model - int(self.intercept_math))))

            elif not self.has_intercept:
                # BEAR FIX
                self.r = '-'

                self.R2 = SSReg / (SSReg + SSErr)

                # BEAR FIX
                self.R2_adj = '-'
                # try: print(f'Wherrys adj r2 = {float(1 - (1 - self.R2) * (self.rows - 1) / (self.rows - self.columns))}')
                # except: print(f'Wherrys adj r2 EXCEPTED')
                # try: print(f'McNemars adj r2 = {float(1 - (1 - self.R2) * (self.rows - 1) / (self.rows - self.columns-1))/2}')
                # except: print(f'McNemars adj r2 EXCEPTED')
                # try: print(f'Lords adj r2 = {float(1 - (1 - self.R2) * (self.rows + self.columns + 1) / (self.rows - self.columns - 1))}')
                # except: print(f'Lords adj r2 EXCEPTED')
                # print(f'Steins adj r2 = {}')

                with np.errstate(all='ignore'):
                    # USE BOOLEAN self.intercept TO TOGGLE CORRECTED / UNCORRECTED SSReg
                    self.F = float((SSReg / self.df_model) / (SSErr / (self.rows - self.df_model - int(self.intercept_math))))

            del self.TARGET, self.TARGET_TRANSPOSE, self.TARGET_AS_LIST

            del SSReg, SSErr

            print(f'            BEAR FINAL CALCS Done. time = {self.bear_time_display(t0)} sec\n')

            break

        return self.return_fxn()




# "has_intercept" IS IF DATA HAS/HASNT COLUMN OF CONSTANTS
# "intercept_math" IS TO USE INTERCEPT SSReg (True) OR NON-INTERCEPT SSReg (False)
#  INTERCEPT SSReg = sum(y_predicted - y_avg)**2, NON-INTERCEPT SSReg = sum(y_predicted)**2

        # MLRegression VERIFIED TO GET THE SAME RESULT AS EXCEL (EXCEPT FOR ADJ_RSQ W NO INTERCEPT) FOR
        # (has_intercept=True, intercept_math=True); (has_intercept=False, intercept_math=False);
        # (has_intercept=False, intercept_math=True); (has_intercept=True, intercept_math=False)
        # FOR statsmodel, DATA MUST BE ORIENTED AS [] = ROWS
        # statsmodels COEFFIECIENTS AND p-VALUES ARE ALWAYS CORRECT

'''
NOTES 1/25/22, REVISITED 10/27/22

AFTER CONSIDERABLE RESEARCH AND DAYS OF WORK:

R2 IS ALWAYS EQUAL TO SSReg / SSTotal
F-SCORE IS ALWAYS (SSR / #COLUMNS)) / (SSE / (#ROWS - #COLUMNS - 1))

THERE ARE SEVERAL DIFFERENT FORMULAS FOR ADJUSTED R2:
-----McNemar's FORMULA IS 1 - (1-R2)*(#ROWS-1)/(#ROWS-#COLUMNS-1)
-----THE PERVERTED MCNEMARS USED BY statsmodels AND EXCEL (AT LEAST FOR "WITH INT" FOR EXCEL) IS 1 - (1-R2)*(#ROWS-1)/(#ROWS-#COLUMNS)
-----STATSMODELS GETS IT RIGHT FOR BOTH INTERCEPT AND NO INTERCEPT
-----EXCEL GETS IT RIGHT FOR INTERCEPT, BUT WRONG FOR NO INTERCEPT


THERE ARE TWO DIFFERENT FORMULAS FOR CALCULATING SSReg BASED ON USING INTERCEPT VS NO INTERCEPT
FOR INTERCEPT, SSReg = SUM( (y_predicted - y_actual_avg)^2 )
FOR NO INTERCEPT, SSReg = SUM( (y_predicted)^2 )
SSReg IS HIGHER FOR "NO INTERCEPT" MATH, AND CAUSES RSQ, ADJ_RSQ, AND F TO BE (MUCH) HIGHER THAN "INTERCEPT" MATH,
MAKING FOR A NON-APPLES-TO-APPLES COMPARISON OF "INTERCEPT" AND "NO INTERCEPT" RSQS


COMPARISON OF METHODS AND RESULTS:
XTX
    *** DATA HAS INTERCEPT --- COEFFS RETURNED ARE CORRECT, EXCEL RSQ FUNCTION ON SUBSEQUENT MODEL RETURNS "INTERCEPT STYLE" RSQ
    *** DATA HAS NO INTERCEPT  --- COEFFS RETURNED ARE CORRECT, EXCEL RSQ FUNCTION ON SUBSEQUENT MODEL RETURNS "INTERCEPT STYLE" RSQ
EXCEL
    *** DATA HAS INTERCEPT --- IF MANUALLY APPENDED "1s" TO DATA, NEITHER "INTERCEPT" NOR "NO INTERCEPT" IS CORRECT, EVERYTHING EXCEPT COEFFS IS WRONG
    *** DATA HAS NO INTERCEPT  --- WITH "INTERCEPT STYLE" MATH: COEFFS, p-values, F, RSQ, r, ADJ_RSQ ARE ALL CORRECT FOR AS IF IT HAD AN INTERCEPT APPENDED
                                WITH "NO INTERCEPT STYLE" MATH: COEFFS, p-values, F, RSQ, r ARE CORRECT FOR "NO INTERCEPT STYLE" MATH
                                    ADJ_RSQ IS INCORRECT FOR "NO INTERCEPT STYLE MATH"
NUMPY
    DOES NOT APPEAR TO HAVE ADJ_RSQ OR F CAPABILITY
    corrcoef RETURNS r AND RSQ THAT ARE CORRECT FOR "INTERCEPT STYLE"
    *** DATA HAS INTERCEPT --- corrcoef RETURNS CORRECT r & RSQ FOR "INTERCEPT STYLE"
    *** DATA HAS NO INTERCEPT --- corrcoef RETURNS CORRECT r & RSQ FOR "INTERCEPT STYLE", INCORRECT FOR "NO INTERCEPT STYLE"
STATSMODELS
    DOESNT APPEAR TO HAVE r CAPABILITY
    ALWAYS RETURNS CORRECT COEFFS AND p-values, YES/NO APPEND OF "1s" TO DATA AND hasconst=True/False MUST BE ALIGNED FOR ALL OTHER STATS TO BE CORRECT
    *** DATA HAS INTERCEPT --- MUST TELL sm THAT hasconst=True: COEFFS, p-values, RSQ, ADJ_RSQ, F ARE ALL CORRECT FOR "INTERCEPT STYLE" MATH
                           --- IF hasconst=False: COEFFS, p-values CORRECT --- RSQ, ADJ_RSQ, F ARE ALL INCORRECT
                               RSQ, ADJ_RSQ, F RESULTS DO NOT ALIGN WITH EITHER "INTERCEPT STYLE" OR "NO INTERCEPT STYLE" CORRECT RESULTS
    *** DATA HAS NO INTERCEPT
                           --- MUST TELL sm THAT hasconst=False: COEFFS, p-values, RSQ, ADJ_RSQ, F ARE ALL CORRECT FOR "NO INTERCEPT STYLE" MATH
                           --- IF hasconst=True: COEFFS, p-values CORRECT --- RSQ, ADJ_RSQ, F ARE ALL INCORRECT
                               RSQ, ADJ_RSQ, F RESULTS DO NOT ALIGN WITH EITHER "INTERCEPT STYLE" OR "NO INTERCEPT STYLE" CORRECT RESULTS
                               THIS MEANS UNFORTUNATELY, sm CANT RUN "INTERCEPT STYLE" MATH ON "NO INTERCEPT" DATA.

REVISITED 10/26-27/22, MADE THE DECISION TO DISALLOW USER FROM SELECTING "intercept_math" OR NOT... WILL BE DETERMINED BY WHETHER THE
DATA HAS AN INTERCEPT OR NOT.   RUNNING "INTERCEPT" VERSION FOR DATA THAT HAS INTERCEPT, OTHERWISE RUNNING "NO INTERCEPT" VERSION.
WHERE THIS IS BEING LEFT OFF IS COEFFS, p-VALUES, r, R2, AND F AGREE W/ EXCEL IN BOTH "INTERCEPT" AND "NO INTERCEPT" CASES.  ADJ_R2 ONLY
AGREES W EXCEL FOR "INTERCEPT" CASE, BUT APPEARS TO BE GOOD ENOUGH FOR THE "NO INTERCEPT" CASE.  THE FORMULA FOR EXCEL ADJ_R2 FOR
"NO INTERCEPT" MATH COMES OUT LOWER (AND EVEN MORE NEGATIVE) THAN WHAT IS BEING DONE HERE. HERE, THE PERVERTED MCNEMARA FORMULA IS
BEING USED FOR BOTH "INTERCEPT" AND "NO INTERCEPT".
'''









if __name__ == '__main__':
    from general_sound import winlinsound as wls
    from ML_PACKAGE.MLREGRESSION.misc_test import coeff_calculator as cc, r2_calculator as rc

    this_module = gmn.get_module_name(str(sys.modules[__name__]))

    ####################################################################################################################
    # TEST VIA REFEREE FROM SCRATCH FOR ACCURACY OF COEFFS & RSQ FOR NP AND SD #########################################
    if True: #vui.validate_user_str(f'Run ACCURACY TEST (y/n) > ', 'YN') == 'Y':
        fxn = 'accuracy test'

        GIVEN_TARGET = np.array([1.5 ,3 ,3 ,3 ,4 ,3 ,3.5 ,3 ,4], dtype=np.float64).reshape((-1,1))

        review_taste = np.array([1.5, 3, 3, 3, 4.5, 3.5, 4, 3.5, 4], dtype=np.float64)
        review_palate = np.array([1.5, 3, 3, 2.5, 4, 3, 4, 2, 3.5], dtype=np.float64)
        review_aroma = np.array([2, 2.5, 2.5, 3, 4.5, 3.5, 3.5, 2.5, 3], dtype=np.float64)
        review_appearance = np.array([2.5, 3, 3, 3.5, 4, 3.5, 3.5, 3.5, 3.5], dtype=np.float64)

        BASE_DATA = np.vstack((review_taste, review_palate, review_aroma, review_appearance)).transpose()



        MASTER_NUM_COLUMNS = [1,2,3,4]
        MASTER_BYPASS_VALIDATION = [True, False]
        MASTER_INTCPT = [False, True]
        MASTER_SAFE_MATMUL = [False, True]
        MASTER_DATA_FORMAT = ['ARRAY', 'SPARSE_DICT']
        MASTER_DATA_ORIENT = ['ROW', 'COLUMN']
        MASTER_TARGET_FORMAT = ['ARRAY', 'SPARSE_DICT']
        MASTER_TARGET_ORIENT = ['ROW', 'COLUMN']
        MASTER_RGLZTN_FCTR = [0, 100]

        total_trials = np.product(list(map(len,((MASTER_NUM_COLUMNS, MASTER_BYPASS_VALIDATION, MASTER_INTCPT, MASTER_SAFE_MATMUL,
                         MASTER_DATA_FORMAT, MASTER_DATA_ORIENT, MASTER_TARGET_FORMAT, MASTER_TARGET_ORIENT, MASTER_RGLZTN_FCTR)))))

        ctr = 0
        for num_cols in MASTER_NUM_COLUMNS:
            GIVEN_DATA = BASE_DATA[..., :num_cols]
            for bypass_validation in MASTER_BYPASS_VALIDATION:
                for has_intercept in MASTER_INTCPT:
                    for safe_matmul in MASTER_SAFE_MATMUL:
                        for data_format in MASTER_DATA_FORMAT:
                            for data_orientation in MASTER_DATA_ORIENT:
                                for target_format in MASTER_TARGET_FORMAT:
                                    for target_orientation in MASTER_TARGET_ORIENT:
                                        for rglztn_fctr in MASTER_RGLZTN_FCTR:
                                            ctr += 1
                                            print(f'Running trial {ctr} of {total_trials}...')
                                            print(f'DATA IS {num_cols} COLUMN {data_format} {["+ INTCPT" if has_intercept else "NO INTERCEPT"][0]} AS {data_orientation}')
                                            print(f'rglztn_fctr IS {rglztn_fctr} WITH {"SAFE" if safe_matmul else "UNSAFE"} MATMUL')
                                            print(f'TARGET IS {target_format} AS {target_orientation}')
                                            print()

                                            WIP_DATA = GIVEN_DATA.copy()
                                            WIP_TARGET = GIVEN_TARGET.copy()

                                            if has_intercept: WIP_DATA = np.insert(WIP_DATA, 1, len(WIP_DATA[0]), axis=1)

                                            EXP_COEFFS = cc.exp_coeff_calc(WIP_DATA, WIP_TARGET, rglztn_fctr)      # AS ROW FOR exp_r2
                                            exp_r2 = rc.exp_r2_calc(WIP_DATA, WIP_TARGET, EXP_COEFFS, has_intercept=has_intercept)


                                            if data_orientation=='COLUMN': WIP_DATA = WIP_DATA.transpose()
                                            if data_format=='SPARSE_DICT': WIP_DATA = sd.zip_list_as_py_float(WIP_DATA)

                                            if target_orientation=='COLUMN': WIP_TARGET = WIP_TARGET.transpose()
                                            if target_format=='SPARSE_DICT': WIP_TARGET = sd.zip_list_as_py_float(WIP_TARGET)

                                            DUM, ACT_COEFFS, DUM, DUM, DUM, act_r2, DUM, DUM = \
                                                MLRegression(
                                                                WIP_DATA,
                                                                data_orientation,
                                                                WIP_TARGET,
                                                                target_orientation,
                                                                DATA_TRANSPOSE=None,
                                                                TARGET_TRANSPOSE=None,
                                                                TARGET_AS_LIST=None,
                                                                XTX=None,
                                                                XTX_INV=None,
                                                                has_intercept=has_intercept,
                                                                intercept_math=has_intercept,
                                                                regularization_factor=rglztn_fctr,
                                                                safe_matmul=safe_matmul,
                                                                bypass_validation=bypass_validation
                                                ).run()

                                            del DUM


                                            # SET EXP_COEFFS TO [[]=COLUMN] FOR array_equiv
                                            if not np.array_equiv(np.round(EXP_COEFFS.reshape((1,-1))[0],10),
                                                                  list(map(lambda x: round(x,10), ACT_COEFFS))):
                                                print(f'\033[91m')
                                                print(f'ACT_COEFFS:')
                                                print(ACT_COEFFS)
                                                print()
                                                print(f'EXP_COEFFS:')
                                                print(EXP_COEFFS.reshape((1,-1))[0])
                                                wls.winlinsound(444, 2000)
                                                raise Exception(f'*** ACTUAL AND EXPECTED COEFFS NOT EQUAL ***')

                                            if not round(act_r2,10) == round(exp_r2,10):
                                                wls.winlinsound(444, 2000)
                                                raise Exception(f'\033[91m*** ACT R2 ({act_r2}) DOES NOT EQUAL EXPECTED ({exp_r2}) ***')

        print(f'\033[92m*** MLRegression NP & SD R2 EQUALITY TESTS COMPLETED SUCCESSFULLY ***\033[0m')
        for _ in range(3): wls.winlinsound(888, 500); time.sleep(1)
    # END TEST VIA REFEREE FROM SCRATCH FOR ACCURACY OF COEFFS & RSQ FOR NP AND SD #####################################
    ####################################################################################################################

    ####################################################################################################################
    # TEST FOR EQUALITY OF NP AND SD OUTPUTS ########################################################################

    if True: #vui.validate_user_str(f'Run NP / SD OUTPUT CONGRUENCY TEST (y/n) > ', 'YN') == 'Y':

        # CREATE BASE_DATA & BASE_TARGET AS NP ORIENTED AS COLUMN.
        # ON THE FLY, CREATE DATA AS NP AND DATA AS SD, AND SEPARATELY SEND TO 2 MLRegression FUNCTIONS. COMPARE OUTPUT.
        # ALSO PERMUATE TARGET THRU NP/SD & ROW/COL


        fxn = 'np/sd equality test'

        cols = 5
        rows = 50



        # BASE_DATA = crsn.create_random_sparse_numpy(0, 10, (cols, rows), 20, _dtype=np.int8)

        from MLObjects.TestObjectCreators.SXNL import CreateSXNL as csxnl

        ctr = 0
        while True:

            # KEEP CREATING NEW DATA UNTIL ONE OF THEM IS INVERTIBLE
            ctr += 1
            print(f'\nMAKING ATTEMPT {ctr} TO GET INVERTIBLE DATA...')

            SXNLClass = csxnl.CreateSXNL(rows=rows,
                                         bypass_validation=False,
                                         data_return_format='ARRAY',
                                         data_return_orientation='COLUMN',
                                         DATA_OBJECT=None,
                                         DATA_OBJECT_HEADER=None,
                                         DATA_FULL_SUPOBJ_OR_SINGLE_MDTYPES=None,
                                         data_override_sup_obj=False,
                                         data_given_orientation=None,
                                         data_columns=cols,
                                         DATA_BUILD_FROM_MOD_DTYPES=['STR','INT'],
                                         DATA_NUMBER_OF_CATEGORIES=3,
                                         DATA_MIN_VALUES=0,
                                         DATA_MAX_VALUES=10,
                                         DATA_SPARSITIES=0,
                                         DATA_WORD_COUNT=None,
                                         DATA_POOL_SIZE=None,
                                         target_return_format='ARRAY',
                                         target_return_orientation='COLUMN',
                                         TARGET_OBJECT=None,
                                         TARGET_OBJECT_HEADER=None,
                                         TARGET_FULL_SUPOBJ_OR_SINGLE_MDTYPES=None,
                                         target_type='FLOAT',
                                         target_override_sup_obj=False,
                                         target_given_orientation=None,
                                         target_sparsity=None,
                                         target_build_from_mod_dtype=None,
                                         target_min_value=None,
                                         target_max_value=None,
                                         target_number_of_categories=None,
                                         refvecs_return_format='ARRAY',
                                         refvecs_return_orientation='COLUMN',
                                         REFVECS_OBJECT=None,
                                         REFVECS_OBJECT_HEADER=None,
                                         REFVECS_FULL_SUPOBJ_OR_SINGLE_MDTYPES=None,
                                         REFVECS_BUILD_FROM_MOD_DTYPES=['STR'],
                                         refvecs_override_sup_obj=False,
                                         refvecs_given_orientation=None,
                                         refvecs_columns=3,
                                         REFVECS_NUMBER_OF_CATEGORIES=3,
                                         REFVECS_MIN_VALUES=None,
                                         REFVECS_MAX_VALUES=None,
                                         REFVECS_SPARSITIES=None,
                                         REFVECS_WORD_COUNT=None,
                                         REFVECS_POOL_SIZE=None
                                         )

            SXNLClass.expand_data(expand_as_sparse_dict=False, auto_drop_rightmost_column=True)

            BASE_DATA = SXNLClass.DATA
            BASE_TARGET = SXNLClass.TARGET

            TEST_DATA = np.insert(BASE_DATA.copy(), 0, 1, axis=0)

            TEST_XTX = np.matmul(TEST_DATA.astype(np.float64), TEST_DATA.transpose().astype(np.float64), dtype=np.float64)
            try:
                np.linalg.inv(TEST_XTX)
                del TEST_DATA, TEST_XTX
                print(f'Success.\n')
                break
            except: print(f'Fail.\n')


        del SXNLClass

        while True:
            # KEEP CYCLING TARGET UNTIL GET 1 THAT IS NOT ALL THE SAME NUMBERS
            if np.min(BASE_TARGET) == np.max(BASE_TARGET): BASE_TARGET = np.random.randint(0, 9, (1, rows))
            else: break




        MASTER_data_given_orient = ['ROW', 'COLUMN']
        MASTER_target_given_format = ['ARRAY', 'SPARSE_DICT']
        MASTER_target_given_orient = ['ROW', 'COLUMN']
        MASTER_DATA_TRANSPOSE_IS_GIVEN = [True, False]
        MASTER_XTX_IS_GIVEN = [True, False]
        MASTER_TARGET_TRANSPOSE_IS_GIVEN = [True, False]
        MASTER_TARGET_AS_LIST_IS_GIVEN = [True, False]
        MASTER_BYPASS_VALIDATION = [False, True]
        MASTER_SAFE = [True, False]
        MASTER_INTCPT = [True, False]


        total_trials = np.product(list(map(len, (MASTER_BYPASS_VALIDATION, MASTER_data_given_orient, MASTER_target_given_orient,
            MASTER_DATA_TRANSPOSE_IS_GIVEN, MASTER_XTX_IS_GIVEN, MASTER_TARGET_TRANSPOSE_IS_GIVEN, MASTER_TARGET_AS_LIST_IS_GIVEN,
            MASTER_SAFE, MASTER_INTCPT))))


        ctr = 0
        for bypass_validation in MASTER_BYPASS_VALIDATION:
            for data_given_orient in MASTER_data_given_orient:
                for target_given_format in MASTER_target_given_format:
                    for target_given_orient in MASTER_target_given_orient:
                        for data_transpose_is_given in MASTER_DATA_TRANSPOSE_IS_GIVEN:
                            for xtx_is_given in MASTER_XTX_IS_GIVEN:
                                for target_transpose_is_given in MASTER_TARGET_TRANSPOSE_IS_GIVEN:
                                    for target_as_list_is_given in MASTER_TARGET_AS_LIST_IS_GIVEN:
                                        for safe in MASTER_SAFE:
                                            for has_intercept in MASTER_INTCPT:
                                                ctr += 1
                                                print(f'Running trial {ctr} of {total_trials}...')

                                                GIVEN_DATA = BASE_DATA.copy()   # BASE_DATA AS COLUMN
                                                if has_intercept:
                                                    GIVEN_DATA = np.insert(GIVEN_DATA, len(GIVEN_DATA), 1, axis=0)

                                                TargetClass = mloo.MLObjectOrienter(

                                                target_is_multiclass=False,
                                                TARGET=BASE_TARGET,
                                                target_given_orientation='COLUMN',
                                                target_return_orientation=target_given_orient,
                                                target_return_format=target_given_format,

                                                TARGET_TRANSPOSE=None,
                                                target_transpose_given_orientation=None,
                                                target_transpose_return_orientation=target_given_orient,
                                                target_transpose_return_format=target_given_format,

                                                TARGET_AS_LIST=None,
                                                target_as_list_given_orientation=None,
                                                target_as_list_return_orientation=target_given_orient,

                                                RETURN_OBJECTS=['TARGET', 'TARGET_TRANSPOSE', 'TARGET_AS_LIST'],

                                                bypass_validation=True,  # CATCHES A MULTICLASS TARGET, CHECKS TRANSPOSES
                                                calling_module=this_module,
                                                calling_fxn=fxn
                                                )


                                                NPSD_TARGET = TargetClass.TARGET
                                                NPSD_TARGET_TRANSPOSE = TargetClass.TARGET_TRANSPOSE if not target_transpose_is_given is None else None
                                                NPSD_TARGET_AS_LIST = TargetClass.TARGET_AS_LIST if not target_as_list_is_given is None else None



                                                NPClass = mloo.MLObjectOrienter(
                                                DATA=GIVEN_DATA,
                                                data_given_orientation='COLUMN',
                                                data_return_orientation=data_given_orient,
                                                data_return_format='ARRAY',

                                                DATA_TRANSPOSE=None,
                                                data_transpose_given_orientation=None,
                                                data_transpose_return_orientation=data_given_orient,
                                                data_transpose_return_format='ARRAY',

                                                XTX=None,
                                                xtx_return_format='ARRAY',

                                                XTX_INV=None,
                                                xtx_inv_return_format=None,

                                                RETURN_OBJECTS=['DATA', 'DATA_TRANSPOSE', 'XTX'],

                                                bypass_validation=True,  # CATCHES A MULTICLASS TARGET, CHECKS TRANSPOSES
                                                calling_module=this_module,
                                                calling_fxn=fxn
                                                )

                                                NP_DATA = NPClass.DATA
                                                NP_DATA_TRANSPOSE = NPClass.DATA_TRANSPOSE if not data_transpose_is_given is None else None
                                                NP_XTX = NPClass.XTX if not xtx_is_given is None else None


                                                SDClass = mloo.MLObjectOrienter(
                                                DATA=GIVEN_DATA,
                                                data_given_orientation='COLUMN',
                                                data_return_orientation=data_given_orient,
                                                data_return_format='SPARSE_DICT',

                                                DATA_TRANSPOSE=None,
                                                data_transpose_given_orientation=None,
                                                data_transpose_return_orientation=data_given_orient,
                                                data_transpose_return_format='SPARSE_DICT',

                                                XTX=None,
                                                xtx_return_format='SPARSE_DICT',

                                                XTX_INV=None,
                                                xtx_inv_return_format=None,

                                                RETURN_OBJECTS=['DATA', 'DATA_TRANSPOSE', 'XTX'],

                                                bypass_validation=True,  # CATCHES A MULTICLASS TARGET, CHECKS TRANSPOSES
                                                calling_module=this_module,
                                                calling_fxn=fxn
                                                )

                                                SD_DATA = SDClass.DATA
                                                SD_DATA_TRANSPOSE = SDClass.DATA_TRANSPOSE if not data_transpose_is_given is None else None
                                                SD_XTX = SDClass.XTX if not xtx_is_given is None else None




                                                # self.XTX_determinant, self.COEFFS, self.PREDICTED, self.P_VALUES, self.r, self.R2, self.R2_adj, self.F
                                                NP_XTX_determinant, NP_COEFFS, NP_PREDICTED, NP_P_VALUES, NP_r, NP_R2, NP_R2_adj, NP_F = \
                                                    MLRegression(
                                                                    NP_DATA,
                                                                    data_given_orient,
                                                                    NPSD_TARGET,
                                                                    target_given_orient,
                                                                    DATA_TRANSPOSE=NP_DATA_TRANSPOSE,
                                                                    XTX=NP_XTX,
                                                                    XTX_INV=None,
                                                                    TARGET_TRANSPOSE=NPSD_TARGET_TRANSPOSE,
                                                                    TARGET_AS_LIST=NPSD_TARGET_AS_LIST,
                                                                    has_intercept=has_intercept,
                                                                    intercept_math=has_intercept,
                                                                    regularization_factor=0,
                                                                    safe_matmul=safe,
                                                                    bypass_validation=bypass_validation).run()


                                                SD_XTX_determinant, SD_COEFFS, SD_PREDICTED, SD_P_VALUES, SD_r, SD_R2, SD_R2_adj, SD_F = \
                                                    MLRegression(
                                                                    SD_DATA,
                                                                    data_given_orient,
                                                                    NPSD_TARGET,
                                                                    target_given_orient,
                                                                    DATA_TRANSPOSE=SD_DATA_TRANSPOSE,
                                                                    XTX=SD_XTX,
                                                                    XTX_INV=None,
                                                                    TARGET_TRANSPOSE=NPSD_TARGET_TRANSPOSE,
                                                                    TARGET_AS_LIST=NPSD_TARGET_AS_LIST,
                                                                    has_intercept=has_intercept,
                                                                    intercept_math=not has_intercept,
                                                                    regularization_factor=0,
                                                                    safe_matmul=safe,
                                                                    bypass_validation=bypass_validation).run()

                                                print(f'*'*400)
                                                #############################################################################################################################
                                                # BEAR
                                                # print(f'DATA')
                                                # [print(_) for _ in NP_DATA]
                                                # print()
                                                #############################################################################################################################
                                                #############################################################################################################################
                                                # print(f'TARGET')
                                                # [print(_) for _ in NPSD_TARGET]
                                                # print()
                                                #############################################################################################################################

                                                print(f'HAS INTERCEPT = ', has_intercept)

                                                #############################################################################################################################
                                                print(f'COEFFS')
                                                COEFF_DF = pd.DataFrame(data=np.vstack((NP_COEFFS, SD_COEFFS)).astype(np.float64),
                                                                       index = ['NP', 'SD'])
                                                                       # columns=[*[f'NP{_+1}' for _ in range(cols)], *[f'SD{_+1}' for _ in range(cols)]])
                                                print(COEFF_DF)
                                                print()
                                                #############################################################################################################################
                                                #############################################################################################################################
                                                print(f'PREDICTED')
                                                PREDICTED_DF = pd.DataFrame(data=np.vstack((NP_PREDICTED, SD_PREDICTED)).astype(np.float64),
                                                                        index = ['NP', 'SD'])
                                                print(PREDICTED_DF)
                                                print()
                                                #############################################################################################################################
                                                #############################################################################################################################

                                                print(f'P_VALUES')
                                                P_VALUES_DF = pd.DataFrame(data=np.vstack((NP_P_VALUES, SD_P_VALUES)).astype(np.float64),
                                                                        index=['NP', 'SD'])
                                                print(P_VALUES_DF)
                                                print()
                                                #############################################################################################################################
                                                #############################################################################################################################

                                                DATA = [
                                                    [NP_XTX_determinant, SD_XTX_determinant],
                                                    [NP_r, SD_r],
                                                    [NP_R2, SD_R2],
                                                    [NP_R2_adj, SD_R2_adj],
                                                    [NP_F, SD_F]
                                                ]

                                                STATS_DF = pd.DataFrame(data=np.array(DATA, dtype=object),
                                                                     index = ['XTX_determinant', 'r', 'R2', 'ADJ_R2', 'F'], columns = ['NP', 'SD'])

                                                print(STATS_DF)
                                                print()

                                                if has_intercept and (round(NP_r, 10) != round(SD_r, 10)):
                                                    print(f'\033[91m')
                                                    print(f'\nNP R2 = {NP_r}, SD R2 = {SD_r}')
                                                    wls.winlinsound(444, 1000)
                                                    raise Exception(f'*** NP AND SD ARE GIVING DIFFERENT R VALUES ***')

                                                if round(NP_R2, 10) != round(SD_R2, 10):
                                                    print(f'\033[91m')
                                                    print(f'\nNP R2 = {NP_R2}, SD R2 = {SD_R2}')
                                                    wls.winlinsound(444, 1000)
                                                    raise Exception(f'*** NP AND SD ARE GIVING DIFFERENT R-SQUARED VALUES ***')

                                                if has_intercept and (round(NP_R2_adj, 10) != round(SD_R2_adj, 10)):
                                                    print(f'\033[91m')
                                                    print(f'\nNP ADJ R2 = {NP_R2_adj}, SD ADJ R2 = {SD_R2_adj}')
                                                    wls.winlinsound(444, 1000)
                                                    raise Exception(f'*** NP AND SD ARE GIVING DIFFERENT ADJ R-SQUARED VALUES ***')

                                                if round(NP_F, 10) != round(SD_F, 10):
                                                    print(f'\033[91m')
                                                    print(f'\nNP F = {NP_F}, SD F = {SD_F}')
                                                    wls.winlinsound(444, 1000)
                                                    raise Exception(f'*** NP AND SD ARE GIVING DIFFERENT F SCORES ***')

                                                if not np.allclose(NP_P_VALUES, SD_P_VALUES, rtol=1e-6, atol=1e-6):
                                                    print(f'\033[91m')
                                                    print(f'\nNP P VALUES = ')
                                                    print(NP_P_VALUES)
                                                    print()
                                                    print(f'\nSD P VALUES = ')
                                                    print(SD_P_VALUES)
                                                    wls.winlinsound(444, 1000)
                                                    raise Exception(f'*** NP AND SD ARE GIVING DIFFERENT P VALUES  ***')

                                                if not np.allclose(NP_COEFFS, SD_COEFFS, rtol=1e-6, atol=1e-6):
                                                    print(f'\033[91m')
                                                    print(f'\nNP COEFFS = ')
                                                    print(NP_COEFFS)
                                                    print()
                                                    print(f'\nSD COEFFS = ')
                                                    print(SD_COEFFS)
                                                    wls.winlinsound(444, 1000)
                                                    raise Exception(f'*** NP AND SD ARE GIVING DIFFERENT COEFFS  ***')


        print(f'\033[92m*** MLRegression NP & SD OUTPUT EQUALITY TESTS COMPLETED SUCCESSFULLY ***\033[0m')
        for _ in range(3): wls.winlinsound(888, 500); time.sleep(1)

    # END TEST FOR EQUALITY OF NP AND SD OUTPUTS #######################################################################
    ####################################################################################################################






































