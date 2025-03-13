import numpy as n, time
from copy import deepcopy
import statsmodels.api as sm
import sparse_dict as sd
from debug import IdentifyObjectAndPrint as ioap

'''
NOTES 1-25-22

AFTER CONSIDERABLE RESEARCH AND DAYS OF WORK:

R2 IS ALWAYS EQUAL TO SSReg / SSTotal
F-SCORE IS ALWAYS (SSR / #COLUMNS)) / (SSE / (#ROWS - #COLUMNS - 1))

THERE ARE SEVERAL DIFFERENT FORMULAS FOR ADJUSTED R2:
-----STATSMODELS USES McNemar's (OR MAYBE A PERVERSION OF IT) AND GETS IT RIGHT FOR BOTH INTERCEPT AND NO INTERCEPT
-----EXCEL USES McNemar'S (OR MAYBE A PERVERSION OF IT) AND GETS IT RIGHT FOR INTERCEPT, BUT WRONG FOR NO INTERCEPT
-----McNemar's FORMULA IS 1 - (1-R2)*(#ROWS-1)/(#ROWS-#COLUMNS-1)


THERE ARE TWO DIFFERENT FORMULAS FOR CALCULATING SSReg BASED ON USING INTERCEPT VS NO INTERCEPT
FOR INTERCEPT, SSReg = SUM((y_model - y_actual_avg)^2_
FOR NO INTERCEPT, SSReg = SUM((y_model)^2)
SSReg IS HIGHER FOR "NO INTERCEPT" MATH, AND CAUSES RSQ, ADJ_RSQ, AND F TO BE (MUCH) HIGHER THAN "INTERCEPT" MATH,
MAKING FOR A NON-APPLES-TO-APPLES COMPARISON OF "INTERCEPT" AND "NO INTERCEPT" RSQS
TO RECONCILE THIS, MAKING THE EXECUTIVE DECISION TO USE "INTERCEPT MATH" OVER "NO INTERCEPT MATH" ON ALL DATA SETS FOR
APPLES-TO-APPLES COMPARISON OF "INTERCEPT" VS "NO INTERCEPT" DATA.  UNFORTUNATELY, STATSMODELS CANT GET THE MATH CORRECT
WHEN APPLYING THEIR "INTERCEPT STYLE" MATH TO "NO INTERCEPT" DATA, SO MUST CREATE FUNCTIONS TO DO THIS.


COMPARISON OF METHODS AND RESULTS:
XTX
    *** DATA HAS INTERCEPT --- COEFFS RETURNED ARE CORRECT, EXCEL RSQ FUNCTION ON SUBSEQUENT MODEL RETURNS "INTERCEPT STYLE" RSQ
    *** DATA HAS NO INTERCEPT  --- COEFFS RETURNED ARE CORRECT, EXCEL RSQ FUNCTION ON SUBSEQUENT MODEL RETURNS "INTERCEPT STYLE" RSQ
EXCEL
    *** DATA HAS INTERCEPT --- IF MANUALLY APPENDED "1s" TO DATA, NEITHER "INTERCEPT" NOR "NO INTERCEPT" IS CORRECT, EVERYTHING EXCEPT COEFFS IS WRONG
    *** DATA HAS NO INTERCEPT  --- WITH "INTERCEPT STYLE" MATH: COEFFS, p-values, F, RSQ, r, ADJ_RSQ ARE ALL CORRECT FOR "INTERCEPT STYLE"
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
                           --- MUST TELL sm THAT hasconst=False: COEFFS, p-values, ADJ_RSQ, F ARE ALL CORRECT FOR "NO INTERCEPT STYLE" MATH
                           --- IF hasconst=True: COEFFS, p-values CORRECT --- RSQ, ADJ_RSQ, F ARE ALL INCORRECT
                               RSQ, ADJ_RSQ, F RESULTS DO NOT ALIGN WITH EITHER "INTERCEPT STYLE" OR "NO INTERCEPT STYLE" CORRECT RESULTS
                               UNFORTUNATELY, sm CANT RUN "INTERCEPT STYLE" MATH ON "NO INTERCEPT" DATA.  SINCE THE DECISION
                               TO ALWAYS USE "INTERCEPT STYLE" MATH ON BOTH "INTERCEPT" AND "NO INTERCEPT" DATA, THIS NECESSITATES
                               CREATING HOMEMADE CLASSES TO DO THIS.
'''


# "has_intercept" IS IF DATA HAS/HASNT COLUMN OF 1s
# "intercept_math" IS TO USE INTERCEPT (CORRECTED) SSReg OR NON-INTERCEPT SSReg MATH
#  INTERCEPT SSReg = sum(y_predicted - y_avg)**2, NON-INTERCEPT SSReg = sum(y_predicted)**2
class MLRegression:
    def __init__(self, name, DATA=None, TARGET=None, DATA_TRANSPOSE=None, TARGET_TRANSPOSE=None, TARGET_AS_LIST=None,
                 orientation='COLUMN', has_intercept=True, intercept_math=True, safe_matmul=True, transpose_check=False):
        
        self.bear_time_display = lambda t0: round((time.time() - t0), 2)
        
        # DATA MUST COME IN AS [[], ... ORIENTATION IS A kwarg], TARGET CAN COME IN AS [[] = COLUMN OR ROW]
        # LSRegression VERIFIED FOR has_intercept=True, intercept_math=True; has_intercept=False, intercept_math=True; has_intercept=False, intercept_math=False
        # ***** NOT VERIFIED FOR has_intercept=True, intercept_math=False *****
        # FOR statsmodel, DATA MUST BE ORIENTED AS [] = ROWS
        # statsmodels COEFFIECIENTS AND p-VALUES ARE ALWAYS CORRECT
        # VALIDATE USER ENTRY FOR orientation AND ORIENT self.DATA

        self.DATA = DATA  # THESE ARE HERE JUST TO ENABLE is_list() AND is_dict() BELOW, CANNOT SAY n.array OR WILL WRECK dict()
        self.DATA_TRANSPOSE = DATA_TRANSPOSE

        if self.is_list():
            self.sparsity = sd.list_sparsity(self.DATA)
            self.shape = self.DATA.shape
        elif self.is_dict():
            self.sparsity = sd.sparsity(self.DATA)
            self.shape = (sd.outer_len(self.DATA), sd.inner_len_quick(self.DATA))
        # self.sparsity_cutoff = 85
        # self.size_cutoff = 200e6

        orientation = orientation.upper()

        # CONFIGURE DATA ###########################################################################################################
        if orientation == 'COLUMN':
            if self.is_list():
                if DATA is not None and DATA_TRANSPOSE is not None:
                    self.DATA, self.DATA_TRANSPOSE = n.array(DATA_TRANSPOSE, dtype=n.float64), n.array(DATA, dtype=n.float64)
                elif DATA is not None and DATA_TRANSPOSE is None:
                    self.DATA, self.DATA_TRANSPOSE = n.array(DATA, dtype=n.float64).transpose(), n.array(DATA, dtype=n.float64)
                elif DATA is None and DATA_TRANSPOSE is not None:
                    self.DATA, self.DATA_TRANSPOSE = n.array(DATA_TRANSPOSE, dtype=n.float64).transpose(), n.array(DATA_TRANSPOSE, dtype=n.float64)
            elif self.is_dict():
                if DATA is not None and DATA_TRANSPOSE is not None:
                    self.DATA, self.DATA_TRANSPOSE = DATA_TRANSPOSE, DATA
                elif DATA is not None and DATA_TRANSPOSE is None:
                    self.DATA, self.DATA_TRANSPOSE = sd.sparse_transpose(DATA), DATA
                elif DATA is None and DATA_TRANSPOSE is not None:
                    self.DATA, self.DATA_TRANSPOSE = sd.sparse_transpose(DATA_TRANSPOSE), DATA_TRANSPOSE
        elif orientation == 'ROW':
            if self.is_list():
                if DATA is not None and DATA_TRANSPOSE is not None:
                    self.DATA, self.DATA_TRANSPOSE = n.array(DATA, dtype=n.float64), n.array(DATA_TRANSPOSE, dtype=n.float64)
                elif DATA is not None and DATA_TRANSPOSE is None:
                    self.DATA, self.DATA_TRANSPOSE = n.array(DATA, dtype=n.float64), n.array(DATA, dtype=n.float64).transpose()
                elif DATA is None and DATA_TRANSPOSE is not None:
                    self.DATA, self.DATA_TRANSPOSE = n.array(DATA_TRANSPOSE, dtype=n.float64), n.array(DATA_TRANSPOSE, dtype=n.float64).transpose()
            elif self.is_dict():
                if DATA is not None and DATA_TRANSPOSE is not None:
                    self.DATA, self.DATA_TRANSPOSE = DATA, DATA_TRANSPOSE
                elif DATA is not None and DATA_TRANSPOSE is None:
                    self.DATA, self.DATA_TRANSPOSE = DATA, sd.sparse_transpose(DATA)
                elif DATA is None and DATA_TRANSPOSE is not None:
                    self.DATA, self.DATA_TRANSPOSE = DATA_TRANSPOSE, sd.sparse_transpose(DATA_TRANSPOSE)
        else: raise ValueError(f'INVALID orientation "{orientation}" IN MLRegression __init__()')

        # END CONFIGURE DATA ##########################################################################################

        # CONFIGURE TARGET_AS_LIST ##########################################################################################
        if TARGET_AS_LIST is not None:
            if isinstance(TARGET_AS_LIST, dict):
                self.TARGET_AS_LIST = sd.unzip_to_ndarray_float64(TARGET_AS_LIST)[0]
            elif isinstance(TARGET_AS_LIST, (n.ndarray, list, tuple)):
                self.TARGET_AS_LIST = n.array(TARGET_AS_LIST, dtype=n.float64)
            else:
                raise TypeError(f'\n*** INVALID TARGET_AS_LIST IN MLRegression()***\n')

        elif TARGET_AS_LIST is None:
            if isinstance(TARGET, dict):
                self.TARGET_AS_LIST = sd.unzip_to_ndarray_float64(TARGET)[0]
            elif isinstance(TARGET, (n.ndarray, list, tuple)):
                self.TARGET_AS_LIST = n.array(TARGET, dtype=n.float64)

        if len(self.TARGET_AS_LIST) == 1: self.TARGET_AS_LIST = self.TARGET_AS_LIST.transpose()
        # elif len(TARGET_AS_LIST) > 1: pass
        # END CREATE TARGET_AS_LIST ##########################################################################################

        # CONFIGURE TARGET ######################################################################################################
        if self.is_dict():
            if isinstance(TARGET, (list, tuple, n.ndarray)):
                # 9-16-22 IF DATA IS DICT & TARGET IS LIST, sd.zip TARGET HERE, TO REMOVE EMBEDDED sd.zips FROM FORMULAS BELOW
                self.TARGET = n.array(TARGET, dtype=n.float64)
                if len(self.TARGET) == 1: self.TARGET = self.TARGET.transpose()
                # elif len(self.TARGET) > 1: pass
                self.TARGET = sd.zip_list_as_py_float(self.TARGET)

            elif isinstance(TARGET, dict):
                self.TARGET = TARGET
                if len(self.TARGET) == 1: self.TARGET = sd.sparse_transpose(self.TARGET)    # TARGET COMING IN AS 1 DICT
                # elif len(self.TARGET) > 1: pass                           # TARGET COMING IN AS MANY DICTS (TRANSPOSED)

        elif self.is_list() and isinstance(TARGET, (n.ndarray, list, tuple)):  # DATA IS LIST & TARGET IS LIST
            # ASSUMES IF TARGET IS DICT, DATA COULD NOT BE LIST
            self.TARGET = n.array(TARGET, dtype=n.float64)
            if len(self.TARGET) == 1: self.TARGET = self.TARGET.transpose()
            # elif len(self.TARGET) > 1: pass

            if not n.array_equiv(self.TARGET, self.TARGET_AS_LIST):
                raise AssertionError(f'MLRegression TARGET (returned as list) IS NOT EQUAL to TARGET AS LIST')
        # END CONFIGURE TARGET ##############################################################################################

        # CONFIGURE TARGET_TRANSPOSE ########################################################################################
        if TARGET_TRANSPOSE is not None:
            if self.is_dict():
                if isinstance(TARGET_TRANSPOSE, dict):
                    self.TARGET_TRANSPOSE = TARGET_TRANSPOSE

                elif isinstance(TARGET_TRANSPOSE, (n.ndarray, list, tuple)):
                    self.TARGET_TRANSPOSE = sd.zip_list_as_py_float(TARGET_TRANSPOSE)

            elif self.is_list():
                if isinstance(TARGET_TRANSPOSE, dict):
                    self.TARGET_TRANSPOSE = sd.unzip_to_ndarray_float64(TARGET_TRANSPOSE)[0]

                elif isinstance(TARGET_TRANSPOSE, (n.ndarray, list, tuple)):
                    self.TARGET_TRANSPOSE = n.array(TARGET_TRANSPOSE, dtype=n.float64)

        elif TARGET_TRANSPOSE is None:
            if self.is_dict(): self.TARGET_TRANSPOSE = sd.sparse_transpose(self.TARGET.copy())
            elif self.is_list(): self.TARGET_TRANSPOSE = self.TARGET.copy().transpose()

        if transpose_check:
            _equiv = True
            if self.is_dict():
                if not sd.core_sparse_equiv(sd.core_sparse_transpose(self.TARGET), self.TARGET_TRANSPOSE): _equiv = False
            elif self.is_list():
                if not n.array_equiv(self.TARGET.transpose(), self.TARGET_TRANSPOSE): _equiv = False

            if not _equiv:
                raise AssertionError(f'transpose(TARGET) IS NOT EQUAL TO TARGET TRANSPOSE.')

        # END CONFIGURE TARGET_TRANSPOSE ########################################################################################

        self.name = name
        self.has_intercept = has_intercept
        self.intercept_math = intercept_math
        self.safe_matmul = safe_matmul

        if self.is_list():
            self.columns = self.DATA.shape[0]
            self.rows = self.DATA.shape[1]
        elif self.is_dict():
            self.columns = sd.inner_len(self.DATA)
            self.rows = sd.outer_len(self.DATA)
        else:
            # 10/12/22 ERROR BELOW SAYING MLRegression HAS NO ATTRIBUTE CALLED "self.columns".  SHOULD HAVE BEEN SET ABOVE.
            raise TypeError(f'\nDATA type IS {type(self.DATA)}, MUST BE DICT OR NDARRAY.')

        self.df_model = max(1, self.columns - int(self.has_intercept))   # NOT EXACTLY SURE ABT df_model WHEN ONLY ONE COLUMN
        self.df_error = self.rows - self.columns
        self.df_total = self.rows - 1

        self.y_avg = n.average(self.TARGET_AS_LIST.astype(float))

        # PLACEHOLDERS
        self.XTX_determinant = 0
        self.XTX_INV = []


    def is_list(self):
        try: _ = isinstance(self.DATA, (list, tuple, n.ndarray))
        except:
            try: _ = isinstance(self.DATA_TRANSPOSE, (list, tuple, n.ndarray))
            except: raise Exception(f'\n*** MLRegression() MUST HAVE AT LEAST ONE DATA OBJECT AS ARGUMENT ***\n')
        return _


    def is_dict(self):
        try: _ = isinstance(self.DATA, dict)
        except:
            try: _ = isinstance(self.DATA_TRANSPOSE, dict)
            except: raise Exception(f'\n*** MLRegression() MUST HAVE AT LEAST ONE DATA OBJECT AS ARGUMENT ***\n')
        return _


    def return_fxn(self):
        return self.XTX_determinant, self.COEFFS, self.PREDICTED, self.SSReg, self.SSErr, self.P_VALUES, \
                    self.r, self.R2, self.R2_adj, self.F


    def return_on_error(self):
        # self.XTX_determinant, self.COEFFS, self.PREDICTED, self.SSReg, self.SSErr, self.P_VALUES, \
        # self.r, self.R2, self.R2_adj, self.F

        if self.is_list():
            return 'nan', ['nan' for _ in range(len(self.DATA[0]))], 'nan', 'nan', 'nan', ['nan' for _ in range(len(self.DATA[0]))], \
                    'nan', 'nan', 'nan', 'nan'
        elif self.is_dict():
            return 'nan', ['nan' for _ in range(sd.inner_len_quick(self.DATA))], 'nan', 'nan', 'nan', \
                   ['nan' for _ in range(sd.inner_len_quick(self.DATA))],  'nan', 'nan', 'nan', 'nan'


    def run(self):

        while True:   # THIS IS HERE JUST TO VERIFY TARGET IS ONLY ONE VECTOR, ELSE BYPASS
            # if len(self.TARGET_AS_LIST) > 1:  # 10/6/22 PASS ON THIS NOW THAT ABLE TO INGEST TARGET AS [ [] = ROWS ]
                # print(f'\n*** TARGET ENTERING MLRegression HAS MORE THAN ONE VECTOR.  CANNOT PERFORM MULTIPLE LINEAR REGRESSION. ***')
                # return self.return_on_error()   # EXITS BY return STATEMENT
            # else:
            if True:
            # IF inv(XTX) CANT BE CALCULATED, RETURN nan FOR EVERYTHING
                while True:  # JUST TO ENABLE SKIP-OUT IF XTX NOT INVERTIBLE
                    try:  # TEST IF XTX CAN BE INVERTED, IF YES, PROCEED TO CALC, IF NOT, SKIP OUT
                        if self.is_list():
                            self.XTX_INV = n.linalg.inv(n.matmul(self.DATA_TRANSPOSE.astype(float), self.DATA.astype(float)))
                        elif self.is_dict():
                            # 9-14-2022 FOR SOME REASON, unzip_to_ndarray DOES NOT WORK, BUT unzip_to_list DOES(???)
                            # unzip_to_nd_array WORKS IF NDARRAY_ INSTANTIATED IN unzip AS FLOAT, BUT NOT IF INSTANTIATED AS OBJECT (Y?)

                            self.XTX_INV = n.linalg.inv(
                                                        sd.core_symmetric_matmul(self.DATA_TRANSPOSE, self.DATA,
                                                        DICT2_TRANSPOSE=self.DATA_TRANSPOSE, return_as='ARRAY')
                                                        )   # linalg.inv CAN ONLY TAKE ARRAY & RETURN ARRAY

                    except:  # IF UNABLE TO GET INVERSE OF XTX, SKIP ALL CALCULATIONS AND RETURN 'nan' FOR EVERYTHING
                        if n.linalg.LinAlgError: return self.return_on_error()  # EXITS BY return STATEMENT
                        else: print(f'\n *** EXCEPTION OTHER THAN LinAlgError IN MLRegression *** \n')

                    with n.errstate(all='ignore'):
                        sign_, logdet_ = n.linalg.slogdet(self.XTX_INV)
                        determ_ = sign_ * n.exp(logdet_)
                        self.XTX_determinant = determ_.astype(float)

                    if self.safe_matmul:
                        if self.is_list():
                            print(f'            BEAR TEST LIST SAFE NP_MATMUL...'); t0 = time.time()
                            self.COEFFS = n.matmul(
                                                    n.matmul(self.XTX_INV.astype(float),
                                                            self.DATA_TRANSPOSE.astype(float),
                                                            dtype=object
                                                    ),
                                                    self.TARGET_AS_LIST.astype(float),
                                                    dtype=object
                            )
                            self.PREDICTED = n.matmul(self.DATA.astype(float), self.COEFFS.astype(float)).transpose()[0]
                            self.COEFFS = self.COEFFS.transpose()[0]
                            print(f'            LIST SAFE MATMUL DOne. time = {self.bear_time_display(t0)} sec')
                        elif self.is_dict():
                            print(f'            BEAR TEST DICT SAFE SD_MATMUL...'); t0 = time.time()
                            self.XTX_INV = sd.zip_list_as_py_float(self.XTX_INV)
                            self.COEFFS = sd.matmul(
                                                    sd.matmul(self.XTX_INV, self.DATA_TRANSPOSE, DICT2_TRANSPOSE=self.DATA),
                                                    self.TARGET, DICT2_TRANSPOSE=self.TARGET_TRANSPOSE
                            )
                            self.PREDICTED = sd.unzip_to_ndarray_float64(sd.matmul(self.DATA, self.COEFFS))[0].transpose()[0]
                            self.COEFFS = sd.unzip_to_ndarray_float64(self.COEFFS)[0].transpose()[0]  # FOR return_fxn
                            print(f'            BEAR SAFE SD_MATMUL Done. time = {self.bear_time_display(t0)} sec')

                    elif not self.safe_matmul:
                        if self.is_list():
                            print(f'            BEAR TEST LIST UNSAFE NP_MATMUL...'); t0 = time.time()
                            # THIS IS ACTUALLY SAFE n.matmul, BUT THIS MUST BE HERE TO ALLOW FOR LISTS IN UNSAFE MODE
                            self.COEFFS = n.matmul(
                                                    n.matmul(self.XTX_INV.astype(float),
                                                             self.DATA_TRANSPOSE.astype(float),
                                                             dtype=object
                                                             ),
                                                    self.TARGET_AS_LIST.astype(float),
                                                    dtype=object
                            )

                            self.PREDICTED = n.matmul(self.DATA.astype(float), self.COEFFS.astype(float)).transpose()[0]
                            self.COEFFS = self.COEFFS.transpose()[0]
                            print(f'            BEAR UNSAFE NP_MATMUL Done. time = {self.bear_time_display(t0)} sec')
                        elif self.is_dict():
                            print(f'            BEAR TEST DICT UNSAFE SD_MATMUL...'); t0 = time.time()
                            self.XTX_INV = sd.zip_list_as_py_float(self.XTX_INV)
                            self.COEFFS = sd.core_matmul(
                                                sd.core_matmul(self.XTX_INV, self.DATA_TRANSPOSE, DICT2_TRANSPOSE=self.DATA),
                                                self.TARGET, DICT2_TRANSPOSE=self.TARGET_TRANSPOSE
                            )

                            self.PREDICTED = sd.unzip_to_ndarray_float64(sd.core_matmul(self.DATA, self.COEFFS))[0].transpose()[0]
                            self.COEFFS = sd.unzip_to_ndarray_float64(self.COEFFS)[0].transpose()[0]  # FOR return_fxn
                            print(f'            BEAR UNSAFE SD_MATMUL Done. time = {self.bear_time_display(t0)} sec\n')

                    self.y_pred_avg = n.mean(self.PREDICTED)

                    # VALIDATE USER ENTRY FOR has_intercept AND intercept_math
                    _ = int(self.has_intercept not in [True, False])
                    __ = int(self.intercept_math not in [True, False])
                    if _ + __ != 0:
                        raise ValueError(f'INVALID ' +
                                       f'{_ * "self.has_intercept "}' +
                                       f'{_ * ("("+str(self.has_intercept)+") ")}' +
                                       f'{__ * "self.intercept_math "}' +
                                       f'{__ * ("("+str(self.intercept_math)+") ")}' +
                                       f'IN RegressionStatistics. CAN ONLY BE True OR False.'
                        )

                    # USE BOOLEAN self.intercept TO TOGGLE CORRECTED / UNCORRECTED SSReg
                    self.SSReg = n.sum((self.PREDICTED - int(self.intercept_math) * self.y_avg) ** 2)
                    self.SSErr = n.sum((self.TARGET_AS_LIST.transpose() - self.PREDICTED) ** 2)

                    # USE statsmodels TO GET p-VALUES (ONLY REASON USING statsmodels AT THIS TIME)
                    if self.is_dict(): self.DATA = sd.unzip_to_ndarray_float64(self.DATA)[0] # DONT NEED self.DATA AFTER THIS
                    ols_results = sm.OLS(self.TARGET_AS_LIST.astype(float), self.DATA.astype(float), missing='drop',hasconst=False).fit()
                    self.P_VALUES = ols_results.pvalues  # hasconst CAN BE WHATEVER AND p-VALUES WILL BE CORRECT
                    # MAYBE THIS IS WRONG WHEN USING "NO INTERCEPT" DATA AND USING "INTERCEPT MATH"???
                    # THINKING THIS SHOULD BE OK, INT MATH / NON-INT MATH SHOULD ONLY BE IMPACTING SSModel (DOES SSModel IMPACT p-VALUES??)

                    print(f'            BEAR TEST FINAL CALCS....'); t0 = time.time()
                    # SCRATCH FOR CALCULATING r, USE int(self.intercept) BOOLEAN TO CONTROL INTERCEPT / NO-INTERCEPT MATH
                    Xi_MINUS_Y_PRED_AVG = n.array(self.PREDICTED - int(self.intercept_math) * self.y_pred_avg, dtype=float)
                    Yi_MINUS_Y_AVG = n.array(self.TARGET_AS_LIST.transpose() - int(self.intercept_math) * self.y_avg, dtype=float)
                    NUMERATOR = n.sum(Xi_MINUS_Y_PRED_AVG[_] * Yi_MINUS_Y_AVG[_])
                    DENOMINATOR = n.sqrt(n.sum(n.power(Xi_MINUS_Y_PRED_AVG, 2)) * n.sum(n.power(Yi_MINUS_Y_AVG, 2)))
                    with n.errstate(all='ignore'):
                        self.r = NUMERATOR / DENOMINATOR

                    if not self.has_intercept and self.intercept_math:
                        self.R2 = self.r**2
                    else:
                        self.R2 = self.SSReg / (self.SSReg + self.SSErr)
                        # 1-26-22 WITH NO INTERCEPT BUT USING INTERCEPT MATH, CANNOT RECONCILE THE DIFFERENCE BETWEEN R2
                        # CALCULATED W NUMPY AND CALCULATED W SSR / (SSR+SSE).  NUMPY IS COMING OUT HIGHER THAN SSR / (SSR+SSE) BUT
                        # BOTH OF THESE METHODS ARE BELOW R2 WHEN USING INTERCEPT IN DATA AND INTERCEPT MATH, WHICH IS EXPECTED & GOOD.
                        # THE INTERESTING THING IS THAT r FOR NO INTERCEPT & INTERCEPT MATH AGREES WITH NUMPY r, BUT R2s DONT
                        # AGREE.  MAKING THE EXECUTIVE DECISION TO USE NUMPY R2 FOR 2 REASONS -- 1) THE r CALCULATED BY THIS
                        # MODULE CAN BE SQUARED TO GET IT (AS ABOVE) AND 2) LATER REGRESSIONS WOULD PROBABLY BE DONE USING THE
                        # NUMPY METHOD, AND THIS AVOIDS FUTURE CONFLICT, WHILE SEEMINGLY MAINTAINING THE ORIGINAL INTENT OF THIS MODULE,
                        # TO GET A MORE SENSIBLE VALUE FOR R2 WHEN NOT USING AN INTERCEPT THAN IS GIVEN WHEN USING "NO INTERCEPT" MATH

                    # McNemar's formula: R2_adj = 1−(1−R2)(rows−1)/(rows−columns−1), Excel & statsmodels APPEAR TO BE USING PERVERSIONS OF THIS
                    self.R2_adj = 1 - (1 - self.R2) * (self.rows - int(self.intercept_math)) / (self.rows - self.columns)
                    # 1-26-22 WHOLE LOT OF FUDGE IN F TO GET AGREEMENT BETWEEN Excel, statsmodels, & THIS FOR INTERCEPT / NON-INTERCEPT
                    with n.errstate(all='ignore'):
                        self.F = (self.SSReg / self.df_model) / (self.SSErr / (self.rows - self.df_model - int(self.intercept_math)))
                    print(f'            BEAR FINAL CALCS Done. time = {self.bear_time_display(t0)} sec\n')
                    break

                return self.return_fxn()












if __name__ == '__main__':
    pass





















































