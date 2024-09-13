import numpy as n, pandas as p

p.set_option("display.max_rows", None, "display.max_columns", None)
from copy import deepcopy
from ML_PACKAGE.MLREGRESSION import MLRegression as mlr
from debug import IdentifyObjectAndPrint as ioap
from linear_algebra import EigenvalueDecomposition as ed


class SVDCoreRunCode:
    # DATA MUST COME IN AS [[], [], .... ] TARGET MUST COME IN AS [[]]
    def __init__(self, DATA, DATA_HEADER, TARGET_VECTOR, TARGET_VECTOR_HEADER, svd_max_columns, orientation='column'):


        while True:  # RELIC FROM GMLR / MI, ESCAPE NOT REALLY NEEDED



            print(f'\nRUNNING SINGULAR VALUE DECOMPOSITION...\n')

            if orientation.upper() == 'COLUMN': self.DATA = n.array(DATA, dtype=object)
            elif orientation.upper() == 'ROW': self.DATA = n.array(DATA, dtype=object).transpose()
            raise ValueError(f'\n*** INVALID orientation "{orientation}" IN SVDCoreRunCode. ***\n')

            self.DATA_HEADER = n.array(DATA_HEADER, dtype=object)

            self.TARGET_VECTOR = n.array(TARGET_VECTOR, dtype=float).tolist()       # TARGET VECTOR MUST COME IN AS [[]]
            self.TARGET_VECTOR_HEADER = TARGET_VECTOR_HEADER

            self.max_columns = svd_max_columns

            # CREATE A HOLDER TO TRACK THE COLUMNS IN X_FINAL
            self.AVAILABLE_COLUMNS = n.array([*range(len(self.DATA))], dtype=int)    # A RELIC FROM GMLRCoreRun, DONT KNOW IF ITS USEFUL

            # CREATE HOLDERS FOR EASY SUBSTITUTION DURING AGGLOMERATIVE MLR CALCULATIONS
            self.R_LIST = ['' for _ in range(self.max_columns)]
            self.R2_LIST = ['' for _ in range(self.max_columns)]
            self.R2_ADJ_LIST = ['' for _ in range(self.max_columns)]
            self.F_LIST = ['' for _ in range(self.max_columns)]

            print(f'BEAR self.DATA')
            ioap.IdentifyObjectAndPrint(self.DATA, 'self.DATA', __name__, 20, 20).run_print_as_df(
                df_columns=[str(_) for _ in range(len(self.DATA))], orientation='column')

            # A MUST BE [ [] = COLUMNS ]
            AAT = n.matmul(self.DATA.transpose().astype(float), self.DATA.astype(float), dtype=object)
            ATA = n.matmul(self.DATA.astype(float), self.DATA.transpose().astype(float), dtype=object)

            ##################################################################################################################################
            # EigenvalueDecomposition RETURNS DESCENDING EIGENVALUE MATRIX AND EIGENVECTORS AS [] = ROWS, BUT COLUMN = UNITARY EIGENVECTORS
            print(f'\nCalculating eigenvectors for AAT (U)...')
            U_EIGVAL_MATRIX, U = ed.EigenvalueDecomposition(AAT.astype(float)).symmetric()
            U_INV = deepcopy(U).transpose()    # U IS ORTHONORMAL SO TRANSPOSE = INVERSE

            print(f'BEAR TEST')
            ioap.IdentifyObjectAndPrint(U, 'U', __name__, 20, 20).run_print_as_df(
                df_columns=[str(_) for _ in range(len(U[0]))], orientation='row')

            print(f'BEAR TEST')
            ioap.IdentifyObjectAndPrint(U_INV, 'U_INV', __name__, 20, 20).run_print_as_df(
                df_columns=[str(_) for _ in range(len(U_INV[0]))], orientation='row')
            
            print(f'Done.')
            ##################################################################################################################################

            ##################################################################################################################################
            # EigenvalueDecomposition RETURNS DESCENDING EIGENVALUE MATRIX AND EIGENVECTORS AS [] = ROWS, BUT COLUMN = UNITARY EIGENVECTORS
            print(f'\nCalculating eigenvectors for ATA (V)...')
            V_EIGVAL_MATRIX, V = ed.EigenvalueDecomposition(ATA.astype(float)).symmetric()
            V_INV = deepcopy(V).transpose()     # U IS ORTHONORMAL SO TRANSPOSE = INVERSE

            print(f'BEAR TEST')
            ioap.IdentifyObjectAndPrint(V, 'V', __name__, 20, 20).run_print_as_df(
                df_columns=[str(_) for _ in range(len(V[0]))], orientation='row')

            print(f'BEAR TEST')
            ioap.IdentifyObjectAndPrint(V_INV, 'V_INV', __name__, 20, 20).run_print_as_df(
                df_columns=[str(_) for _ in range(len(V_INV[0]))], orientation='row')

            print(f'Done.')
            ##################################################################################################################################

            ##################################################################################################################################
            # SINGULAR VALUES = SQRT(EIGENVALUES OF ATA)
            # EASIER WAY TO GET SINGULAR VALUES, KEEP TO CHECK AGAINST VALUES FROM SVD BELOW
            SQRT_EIGVALS_ATA = n.sqrt([V_EIGVAL_MATRIX[_][_] for _ in range(len(V_EIGVAL_MATRIX))])
            print(f'sqrt(EIGENVALUES OF V) = {SQRT_EIGVALS_ATA}')
            ##################################################################################################################################

            ##################################################################################################################################
            print(f'BEAR TEST REASSEMBLAGE OF AAT FROM U @ /\_U @ U_T')
            # BEAR 5-9-22 FIGURE OUT Y THIS ISNT GIVING AAT
            ioap.IdentifyObjectAndPrint(AAT, 'AAT', __name__, 20, 20).run_print_as_df(
                df_columns=[str(_) for _ in range(len(AAT))], orientation='column')
            TEST_AAT = U@(U_EIGVAL_MATRIX@U_INV).transpose()
            ioap.IdentifyObjectAndPrint(TEST_AAT, 'TEST_AAT', __name__, 20, 20).run_print_as_df(
                df_columns=[str(_) for _ in range(len(TEST_AAT))], orientation='column')

            print(f'BEAR TEST REASSEMBLAGE OF ATA FROM V @ /\_V @ V_T')
            # BEAR 5-9-22 FIGURE OUT Y THIS ISNT GIVING ATA
            ioap.IdentifyObjectAndPrint(ATA, 'ATA', __name__, 20, 20).run_print_as_df(
                df_columns=[str(_) for _ in range(len(ATA))], orientation='column')
            TEST_ATA = V@(V_EIGVAL_MATRIX@V_INV).transpose()
            ioap.IdentifyObjectAndPrint(TEST_ATA, 'TEST_ATA', __name__, 20, 20).run_print_as_df(
                df_columns=[str(_) for _ in range(len(TEST_ATA))], orientation='column')
            ##################################################################################################################################


            # A = (U)(SIGMA)(V_T) .........  SIGMA = (U_I)(A)(V) .....  U AND V ARE ORTHONORMAL, TRANSPOSE = INVERSE
            with n.errstate(all='ignore'):
                # STEP1 = A@V
                STEP1 = n.matmul(self.DATA.transpose().astype(float), V.astype(float), dtype=object)
                print(f'BEAR TEST')

                # SIGMA_MATRIX = U_INV@STEP1, AKA U_INV@(A@V)
                SIGMA_MATRIX = n.matmul(U_INV.astype(float), STEP1.astype(float), dtype=object)
                # ATA AND AAT SHOULD ALWAYS BE SYMMETRIC POSITIVE SEMI-DEFINITE
                # OPERATING UNDER THE BELIEF THAT SING VALUES ARE ALWAYS >= 0, THEREFORE APPLY ABS FUNCTION
                # BECAUSE FOR SOME REASON eigh RETURNS EIGVALS AND EIGVECS THAT CAUSE SOME NEGATIVE SING VALUES
                # (THE INTERNET SEEMS TO INDICATE TAKING ABS IS THE THING TO DO)
                SIGMA_MATRIX_CHECK = deepcopy(SIGMA_MATRIX)  # KEEP A COPY OF OLD FOR CHECK OF RECONSTRUCTING DATA
                SIGMA_MATRIX = n.abs(SIGMA_MATRIX)   # FOR REPORTING
                # GET RID OF ROUND OFF ERROR ON ZEROS IN EIGENVALUES, ARBITRARILY CHOOSING THRESHOLD OF 1e-10
                SIGMA_MATRIX = SIGMA_MATRIX * (SIGMA_MATRIX >= 1e-10)

                ioap.IdentifyObjectAndPrint(SIGMA_MATRIX, 'SIGMA_MATRIX', __name__, 20, 20).run_print_as_df(
                    df_columns=[str(_) for _ in range(len(SIGMA_MATRIX[0]))], orientation='row')

                # SIGMA_MATRIX COMES OUT OF ABOVE MATMULS AS [ [] = ROWS ]

            # TAKE THE DIAGONAL TO GET ARRAY OF SIGMAS
            SIGMAS = n.array([SIGMA_MATRIX[_][_] for _ in range(len(SIGMA_MATRIX[0]))], dtype=float)
            ioap.IdentifyObjectAndPrint(SIGMAS, 'SIGMAS', __name__, 20, 20).run_print_as_df(
                df_columns=['SIGMAS'], orientation='column')

            # CHECK EQUIVALENCY OF SIGMAS AND sqrt(reverse(sort(EIGVALS_ATA)))
            if n.array_equiv([round(_, 8) for _ in SIGMAS], [round(_, 8) for _ in SQRT_EIGVALS_ATA]):
                print(f'\n*** SIGMAS ARE EQUAL TO SQRT OF ATAs EIGENVALUES ***\n')
            else:
                print(f'\n*** FATAL ERROR.  SIGMAS ARE NOT EQUAL TO SQRT OF ATAs EIGENVALUES ***\n')

            ##################################################################################################################################
            ##################################################################################################################################
            #################################################################################################################################
            print(f'BEAR TEST REASSEMBLAGE')
            PART1 = n.matmul(SIGMA_MATRIX_CHECK.astype(float), V_INV.astype(float), dtype=object)
            REASSEMBLED_A = n.matmul(U.astype(float), PART1.astype(float), dtype=object)
            ioap.IdentifyObjectAndPrint(REASSEMBLED_A, 'REASSEMBLED_A', __name__, 20, 20).run_print_as_df(
                df_columns=DATA_HEADER[0], orientation='row')
            ##################################################################################################################################
            ##################################################################################################################################
            ##################################################################################################################################


            print(f'\n*** SVD SCORE CALCULATIONS COMPLETE. ***')
            print(f'\n*** PROCEEDING TO SORT, SELECT, AND DISPLAY ***\n')

            SORT_KEY = n.fromiter(reversed(n.argsort(SIGMAS)), dtype=int)
            SORTED_HEADER = n.array(deepcopy(self.DATA_HEADER[0]), dtype=str)[SORT_KEY]

            self.SVD_SCORES = SIGMAS[SORT_KEY]     # [:self.max_columns]
            self.WINNING_COLUMNS = deepcopy(SORT_KEY)     #[:self.max_columns]


            # AGGLOMERATE ROWS ONE AT A TIME, GET DETERM, RSQ, RSQ-ADJ, F
            DATA_WIP = []
            DATA_HEADER_WIP = []
            for win_rank in range(len(self.WINNING_COLUMNS)):
                win_idx = self.WINNING_COLUMNS[win_rank]
                DATA_WIP.append(deepcopy(self.DATA[win_idx]))
                DATA_HEADER_WIP.append(self.DATA_HEADER[0][win_idx])
                # 4-18-22 CALCULATE MLR RESULTS TO GIVE SOME KIND OF ASSESSMENT OF WHAT SVD IS ACCOMPLISHING
                # self.COEFFS, self.P_VALUES ARE OVERWRITTEN EACH CYCLE, UNTIL THE LAST, YIELDING FINAL COEFFS AND P_VALUES
                # FOR FINAL ASSEMBLAGE OF COLUMNS; self.R_LIST, self.R2_LIST, self.R2_ADJ_LIST, self.F_LIST ARE APPENDED
                # ON EACH CYCLE AND REPORT THE STEP-WISE CHANGES DURING AGGLOMERATION

                # 4/15/23 BEAR TEST THIS --- ARGS/KWARGS, MLRegression CHANGED
                DUM, self.COEFFS, DUM, self.P_VALUES, self.R_LIST[win_rank], self.R2_LIST[win_rank], \
                self.R2_ADJ_LIST[win_rank], self.F_LIST[win_rank] = \
                    mlr.MLRegression(DATA=DATA_WIP, DATA_TRANSPOSE=None,
                                     data_given_orientation='COLUMN',
                                     TARGET=self.TARGET_VECTOR, target_given_orientation='COLUMN',
                                     TARGET_TRANSPOSE=None, TARGET_AS_LIST=self.TARGET_VECTOR, has_intercept=False,
                                     intercept_math=True, safe_matmul=True, transpose_check=False
                    ).run()


            # CREATE "TRAIN_RESULTS" OBJECT, TO BE RETURNED TO SVDRun FOR PRINT TO SCREEN & DUMP TO FILE
            self.TRAIN_RESULTS = [['COLUMN'],['SVD SCORE'],['COEFFS'],['p VALUE'],[f'CUM R'],[f'CUM RSQ'],[f'CUM ADJ RSQ'],
                                  [f'CUM F']]


            train_appender = lambda idx2, object: self.TRAIN_RESULTS[idx2].append([float(f'{_:6g}') if isinstance(_, (int,float)) else object for _ in [object]][0])

            for idx in range(len(DATA_HEADER_WIP)):
                OBJECTS = ['', self.SVD_SCORES[idx], self.COEFFS[idx], self.P_VALUES[idx], self.R_LIST[idx], self.R2_LIST[idx],
                           self.R2_ADJ_LIST[idx], self.F_LIST[idx]]

                # '' IN OBJECTS FOR CONGRUENT INDEXING, KEEP INSIDE for OTHERWISE EXPLODY FOR idx REFERENCED B4 ASSIGNMENT
                self.TRAIN_RESULTS[0].append(DATA_HEADER_WIP[idx])
                for col_idx in range(1,8):
                    train_appender(col_idx, OBJECTS[col_idx])


            # PRINT WINNING COLUMNS
            print(f'\nWINNING COLUMNS FOR SINGULAR VALUE DECOMPOSITION:')
            wd = 12
            pd = lambda object, width: str([f'{object:7f}' if isinstance(_, (int,float)) else object for _ in [object]][0])[:width-3].ljust(width)
            print(f'{"".ljust(6)}{pd("COLUMN", 40)}{pd("SVD SCORE", wd)}{pd("COEFF", wd)}{pd("p VALUE", wd)}{pd(f"CUM R", wd)}'
                  f'{pd(f"CUM R2", wd)}{pd(f"CUM ADJ R2", wd)}{pd(f"CUM F", wd)}')

            for idx in range(len(DATA_HEADER_WIP)):
                p_value = [f'{float(_):7f}' if isinstance(_, (int,float)) else 'ERR' for _ in [self.P_VALUES[idx]]][0]  # TURN INTO A VARIABLE HERE TO BE ABLE TO PUT INTO pd lambda INSIDE ANOTHER f''
                print(f'{str(idx).ljust(6)}' +
                      f'{pd(DATA_HEADER_WIP[idx], 40)}' +
                      f'{pd(self.SVD_SCORES[idx], wd)}' +
                      f'{pd(self.COEFFS[idx], wd)}' +
                      f'{pd(p_value, wd)}' +
                      f'{pd(self.R_LIST[idx], wd)}' +
                      f'{pd(self.R2_LIST[idx], wd)}' +
                      f'{pd(self.R2_ADJ_LIST[idx], wd)}' +
                      f'{pd(self.F_LIST[idx], wd)}')

            break


    def return_fxn(self):
        return self.WINNING_COLUMNS, self.COEFFS, self.TRAIN_RESULTS


    def run(self):
        return self.return_fxn()





if __name__ == '__main__':
    from data_validation import validate_user_input as vui
    from read_write_file.generate_full_filename import base_path_select as bps, filename_enter as fe

    header_dum = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    rows = 8
    columns = 5
    # # INT
    DATA = [[n.random.randint(0,10) for _ in range(rows)] for __ in range(columns)]
    DATA_HEADER = [[header_dum[_] for _ in range(columns)]]
    # BIN
    # DATA = [[n._random_.randint(0,2) for _ in range(rows)] for __ in range(columns)]
    # DATA_HEADER = [[header_dum[_] for _ in range(columns)]]
    TARGET = [[n.random.randint(0,2) for _ in range(rows)]]
    TARGET_HEADER = [['TARGET']]
    INDEX = [*range(1,rows+1), 'SCORE']

    svd_max_columns = 10

    WINNING_COLUMNS, COEFFS, TRAIN_RESULTS = SVDCoreRunCode(DATA, DATA_HEADER, TARGET, TARGET_HEADER, svd_max_columns,
                                  orientation='column').run()

    print(f'\nTRAIN RESULTS:')
    print(p.DataFrame(data=n.array([COLUMN[1:] for COLUMN in TRAIN_RESULTS], dtype=object).transpose(),
                      columns=[COLUMN[0] for COLUMN in TRAIN_RESULTS]))

    DATA_DICT = {}
    for idx in range(len(TARGET)):
        DATA_DICT[TARGET_HEADER[0][idx]] = TARGET[idx]

    for idx in WINNING_COLUMNS:
        DATA_DICT[DATA_HEADER[0][idx]] = DATA[idx]

    DF = p.DataFrame(DATA_DICT)
    print()
    print(DF)

    if vui.validate_user_str(f'Dump DATA to file? (y/n) > ', 'YN') == 'Y':
        base_path = bps.base_path_select()
        file_name = fe.filename_wo_extension()
        print(f'\nSaving file to {base_path+file_name+".xlsx"}....')
        p.DataFrame.to_excel(DF,
                     excel_writer=base_path + file_name + '.xlsx',
                     float_format='%.2f',
                     startrow=1,
                     startcol=1,
                     merge_cells=False
                     )
        print('Done.')





