import numpy as n, pandas as p
from copy import deepcopy
from debug import IdentifyObjectAndPrint as ioap



class EigenvalueDecomposition:
    '''
    Use numpy eigh (symmetric matrix) and eig (general matrix) to get unitary eigenvectors and eigenvalues.
    Returns eigenvalues sorted descending in a matrix and unitary eigenvectors following the same sort pattern.
    eigh and eig return unitary eigenvector matrices that are oriented as [] = rows, with the eigenvectors
    in the column, i.e., eigenvector for lambda1 occupies the [0] position of all the lists in the array.
    '''

    def __init__(self, A):
        self.A = n.array(A, dtype=object)
        self.A_WIP = deepcopy(A)
        self.EIGVALS = ''
        self.EIGVECS = ''


    def general_core_fxn(self):
        return n.linalg.eig


    def symmetric_core_fxn(self):
        return n.linalg.eigh


    def general_eigval_treatment(self):
        # OPERATING UNDER THE BELIEF THAT EIGVALS OF GENERAL MATRICES CAN BE <0, THEREFORE NO TREATMENT
        return lambda _: _


    def symmetric_eigval_treatment(self):
        # OPERATING UNDER THE BELIEF THAT EIGVALS OF SYMMETRIC MATRICES ARE ALWAYS >= 0, THEREFORE APPLY ABS FUNCTION
        # TO EIGVALS RETURNED FROM eigh, BECAUSE FOR SOME REASON eigh RETURNS SOME NEGATIVE (THE INTERNET SEEMS TO INDICATE
        # TAKING ABS IS THE THING TO DO)
        # return n.abs
        return lambda _: _


    def return_fxn(self):
        return self.EIGVAL_MATRIX, self.EIGVECS


    def core_run_code(self, core_fxn, eigval_treatment):
        # GET EIGVALS AND EIGVECS #########################################################################################################
        # eig RETURNS EIGVALS IN NO PARTICULAR ORDER, WANT DESCENDING, SO DO argsort
        # eigh RETURNS EIGVALS IN ASCENDING, WANT DESCENDING, SO REVERSE EIGVALS AND X_T WITH argsort
        # X IS UNITARY EIGENVECTOR MATRIX WITH EIGENVECTORS IN COLUMNS
        EIGVALS, X = core_fxn(self.A_WIP.astype(float))
        EIGVALS = eigval_treatment(EIGVALS)    # MAKE SURE ALL ARE POSITIVE IF SYMMETRIC, PASS OTHERWISE
        # GET RID OF ROUND OFF ERROR ON ZEROS IN EIGENVALUES, ARBITRARILY CHOOSING THRESHOLD OF 1e-10
        EIGVALS = EIGVALS * (n.abs(EIGVALS >= 1e-10))
        MASTER_SORT_KEY =  n.fromiter(reversed(n.argsort(EIGVALS)), dtype=int)
        EIGVALS_SORTED = EIGVALS[MASTER_SORT_KEY]

        EIGVAL_MATRIX = n.array([[EIGVALS_SORTED[_] if _ == __ else 0 for _ in range(len(EIGVALS))] for __ in range(len(EIGVALS))],
                                dtype=object)

        X_SORTED = deepcopy(X).transpose()[MASTER_SORT_KEY]  # TRANSPOSE FOR EASE OF SORTING COLUMNS TO MATCH WITH EIGVAL SORT
        X_SORTED = X_SORTED.transpose()  # TRANSPOSE BACK TO COLUMNS (NOT []) = EIGVECS

        self.EIGVAL_MATRIX = EIGVAL_MATRIX
        self.EIGVECS = X_SORTED


    def general(self):
        self.core_run_code(self.general_core_fxn(), self.general_eigval_treatment())
        return self.return_fxn()


    def symmetric(self):
        self.core_run_code(self.symmetric_core_fxn(), self.symmetric_eigval_treatment())
        return self.return_fxn()










if __name__ == '__main__':
    # GENERATE & PRINT DUMMY MATRIX ########################################################################################################
    # A = n.random.randint(0,10,[5,5])
    # A = n.array([[9,-2,-4],[4,3,-4],[2,-2,3]], dtype=object)
    A = n.array([[9,0,2],[0,3,0],[2,0,9]], dtype=object)
    ioap.IdentifyObjectAndPrint(A, 'A', __name__, 20,20).run_print_as_df(df_columns=[str(_) for _ in range(len(A[0]))], orientation='row')
    # END GENERATE & PRINT DUMMY MATRIX ########################################################################################################

    TEST = EigenvalueDecomposition(A)

    TEST_EIGVAL_MATRIX, TEST_EIGVECS = TEST.symmetric()
    ioap.IdentifyObjectAndPrint(TEST_EIGVAL_MATRIX, 'TEST_EIGVAL_MATRIX', __name__, 20, 20).run_print_as_df(
        df_columns=[str(_) for _ in range(len(TEST_EIGVECS[0]))], orientation='row')
    ioap.IdentifyObjectAndPrint(TEST_EIGVECS, 'TEST_EIGVECS', __name__, 20, 20).run_print_as_df(
        df_columns=[str(_) for _ in range(len(TEST_EIGVECS[0]))], orientation='row')

    TEST_A = TEST_EIGVECS@(TEST_EIGVAL_MATRIX@n.linalg.inv(TEST_EIGVECS))
    ioap.IdentifyObjectAndPrint(TEST_A, 'TEST_A', __name__, 20, 20).run_print_as_df(
        df_columns=[str(_) for _ in range(len(TEST_A[0]))], orientation='row')






