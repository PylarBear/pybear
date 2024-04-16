import numpy as n
from ML_PACKAGE.MLREGRESSION import MLRegression as mlr
from linear_algebra import XTX_determinant as xtxd
import sys, inspect
from debug import get_module_name as gmn



class RowColumnIteratorTemplate:

    # WORK ON GETTING THE HEADER LAYOUTS FOR column_drop, row_drop, whole_data_object_stats FOR
    # Expand and Augment AND STANDARDIZE TO USE THIS MODULE FOR BOTH.


    def __init__(self, DATA_OBJECT, name, DATA_OBJECT_HEADER, TARGET_OBJECT,
    top_line_description, # ESSENTIALLY TITLE OF TABLE
                          # in Expand:
                          # whole_data_object_stats = f'\nRESULTS FOR ENTIRE {name} OBJECT:', f'ORIGINAL COLUMN', max_len)
                          # column_drop_iterator = (f'\nEFFECTS OF REMOVING A COLUMN:', 'CATEGORY', max_len)
                          # row_drop_iterator = stats_header_print(f'\nEFFECTS OF REMOVING A ROW:', 'ROW INDEX', max_len)

                          # in Augment everything is hard-coded
                          # column_drop_iterator prints 2 objects
                 print(
        #f'{"CATEGORY".ljust(min(max(8, max_len + 2), 50))}{"FREQ".ljust(10)}{"XTX DETERM".ljust(20)}{"minv(XTX) MIN ELEM".ljust(20)}{"minv(XTX) MAX ELEM".ljust(20)}')
                          # '''print(f'\nEFFECTS OF REMOVING A COLUMN ON DETERMINANT:')'''

        first_column_header,  # in Expand this is header of first column,
                            # whole_data_object_stats = f'\nRESULTS FOR ENTIRE {name} OBJECT:', f'ORIGINAL COLUMN', max_len)
                            # column_iterator = (f'\nEFFECTS OF REMOVING A COLUMN:', 'CATEGORY', max_len)
        first_column_text,
        max_len,
        datatype,
        freq_text,
        max_len,
        calc_allow='Y',
        append_ones='N'):

        self.this_module = gmn.get_module_name(str(sys.modules[__name__]))

        fxn = inspect.stack()[0][3]


            if calc_allow == 'N':
                self.determ, self.min_elem, self.max_elem, self.COEFFICIENTS, self.P_VALUES, self.r, self.R2, \
                self.R2_adj, self.F = tuple(['' for _ in range(9)])
            elif calc_allow == 'Y':
                with n.errstate(all='ignore'):
                    xtxd.XTX_determinant(DATA_AS_ARRAY_OR_SPARSEDICT=DATA_OBJECT, name=name, module=this_module,
                                         fxn=fxn,
                                         print_to_screen=False, return_on_exception='nan')
                    # COULD ALSO GET THIS AS AN ATTRIBUTE OF MLRegression CLASS

                    # *** MLRegression WAS HERE ****


    def iterator_function(self):
        # OVERWRITTEN IN CHILD
        pass

    def core_ML_function(self):
        # RETURNED FROM MLRegression
        # XTX_determinant, self.COEFFS, PREDICTED, P_VALUES, r, R2, R2_adj, F
        RESULTS = mlr.MLRegression(DATA=DATA_OBJECT,
                                   DATA_TRANSPOSE=None,
                                   data_given_orientation='COLUMN', # 4/15/23 BEAR FIX THIS ARGS/KWARGS, MLRegression CHANGED
                                   TARGET=TARGET_OBJECT,
                                   TARGET_TRANSPOSE=None,
                                   TARGET_AS_LIST=None,
                                   target_given_orientation='COLUMN', # 4/15/23 BEAR FIX THIS ARGS/KWARGS, MLRegression CHANGED
                                   has_intercept='INTERCEPT' in DATA_OBJECT_HEADER[0],
                                   intercept_math=True,
                                   safe_matmul=True,
                                   transpose_check=False
                                   ).run()


        # self.COEFFICIENTS, self.P_VALUES, self.r, self.R2, self.R2_adj, self.F = RESULTS[1], *RESULTS[3:]





    def calc_statistics(self, DATA_OBJECT, name, DATA_OBJECT_HEADER, TARGET_OBJECT, calc_allow='Y')
    def stats_header_print(self, top_line_description, first_column_header, max_len)
    def stats_print(self, first_column_text, datatype, freq_text, max_len)
    def whole_data_object_stats(self, OBJECT, name, HEADER, append_ones='N')
    def row_drop_iterator(self, OBJECT, name, HEADER, TARGET_OBJECT)

    def return_fxn(self):
        pass














if __name__ == '__main__':
    pass









