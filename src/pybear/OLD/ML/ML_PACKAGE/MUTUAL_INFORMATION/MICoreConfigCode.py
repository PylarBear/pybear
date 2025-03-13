import numpy as np
from data_validation import validate_user_input as vui
from general_data_ops import get_shape as gs


class MICoreConfigCode:
    def __init__(self, mi_config, DATA, data_run_orientation, batch_method, batch_size,
                 mi_int_or_bin_only, mi_max_columns, mi_bypass_agg):

        self.mi_config = mi_config  # THIS HAS TO STAY mi_config, IT WILL CONFLICT WITH THE FUNCTION config()
        self.DATA = DATA

        self.data_run_orientation = data_run_orientation
        self.batch_method = batch_method
        self.batch_size = batch_size
        self.int_or_bin_only = mi_int_or_bin_only
        self.max_columns = mi_max_columns
        self.bypass_agg = mi_bypass_agg

        self.data_shape = gs.get_shape('DATA', self.DATA, self.data_run_orientation)


    def return_fxn(self):
        del self.data_shape
        return self.batch_method, self.batch_size, self.int_or_bin_only, self.max_columns, self.bypass_agg


    def config(self):


        ####################################################################################################################
        ####################################################################################################################
        if self.mi_config in 'BE':  # batch method (b)',
            self.batch_method = vui.validate_user_str(f'\nSelect batch method (b)atch (m)inibatch > ', 'MB')

            if self.batch_method == 'B': self.batch_size = self.data_shape[0]
        ####################################################################################################################
        ####################################################################################################################

        ####################################################################################################################
        ####################################################################################################################
        if self.mi_config in 'CE':  # batch size (c)',
            if self.batch_method == 'B':
                print(f'\n*** CANNOT SET batch size WHEN BATCH METHOD IS BATCH(b) ***\n')
            elif self.batch_method == 'M':
                print(f'\nEnter batch size as integer for absolute batch size, enter as decimal < 1 for percentage of full batch ')
                self.batch_size = vui.validate_user_int(
                    f'Select batch size (DATA has {int(self.data_shape[0])} examples) > ', min=1, max=int(self.data_shape[0]))
                # self.batch_size IS str AT THIS POINT
                if float(self.batch_size) == int(self.batch_size): self.batch_size = int(self.batch_size)
                else: self.batch_size = float(self.batch_size)   # IS TRANSLATED FROM % TO # ROWS IN MICoreRunCode
        ####################################################################################################################
        ####################################################################################################################

        ####################################################################################################################
        ####################################################################################################################
        if self.mi_config in 'DE':  # toggle int_or_bin_only(d)
            self.int_or_bin_only = {'Y': True, 'N': False}[
                                        vui.validate_user_str(f"\nUse 'INT' or 'BIN' columns only? (y/n) > ", 'YN')]
        ####################################################################################################################
        ####################################################################################################################

        ####################################################################################################################
        ####################################################################################################################
        if self.mi_config in 'FE':  # max columns(f)
            self.max_columns = vui.validate_user_int(
                f'\nEnter max columns (DATA has {self.data_shape[1]} columns) > ', min=1, max=self.data_shape[1])
        ####################################################################################################################
        ####################################################################################################################

        ####################################################################################################################
        ####################################################################################################################
        if self.mi_config in 'GE':  # bypass agglomerative MLR(g)
            self.bypass_agg = {'Y': True, 'N': False}[
                                        vui.validate_user_str(f"\nBypass agglomerative MLR? (y/n) > ", 'YN')]
        ####################################################################################################################
        ####################################################################################################################

        return self.return_fxn()




if __name__ == '__main__':

    # TEST MODULE  --- MODULE AND TEST VERIFIED GOOD 5/13/23

    _rows = 6
    _cols = 4

    standard_config = 'BYPASS'
    mi_config = 'BYPASS'
    DATA = np.arange(_rows*_cols, dtype=np.int32).reshape((_cols, _rows))
    data_run_orientation = 'COLUMN'
    batch_method = 'B'
    batch_size = int(1e12)
    mi_int_or_bin_only = True
    mi_max_columns = 10
    mi_bypass_agg = False


    for mi_config in 'EBCDFG':
        base_desc = f'\nPASS {mi_config} '
        config_info = ''

        if mi_config=='E': config_info = 'SHOULD PROMPT FOR BATCH TYPE, BATCH SIZE, INT OR BIN ONLY, MAX COLUMNS, AND BYPASS AGGLOMERATIVE MLR'
        if mi_config=='B': config_info = 'SHOULD PROMPT FOR BATCH TYPE'
        if mi_config=='C': config_info = 'SHOULD PROMPT FOR BATCH SIZE'; batch_method = 'M'
        if mi_config=='D': config_info = 'SHOULD PROMPT FOR INT OR BIN ONLY'
        if mi_config=='F': config_info = 'SHOULD PROMPT FOR MAX COLUMNS'
        if mi_config=='G': config_info = 'SHOULD PROMPT FOR BYPASS AGGLOMERATIVE MLR'

        print(f'*'*120)
        __ = input(base_desc + config_info + f'\nHIT ENTER TO CONTINUE > ')

        batch_method, batch_size, int_or_bin_only, max_columns, bypass_agg = \
            MICoreConfigCode(mi_config, DATA, data_run_orientation, batch_method, batch_size, mi_int_or_bin_only,
                             mi_max_columns, mi_bypass_agg).config()
        print(f'*'*120)



















