import numpy as n, pandas as p
from copy import deepcopy
from data_validation import validate_user_input as vui
from general_list_ops import list_select as ls


class SVDCoreConfigCode:
    def __init__(self, svd_config, DATA, DATA_HEADER, svd_max_columns):


        self.svd_config = svd_config  # THIS HAS TO STAY svd_config, IT WILL CONFLICT WITH THE FUNCTION config()
        self.DATA = DATA
        self.DATA_HEADER = DATA_HEADER

        self.max_columns = svd_max_columns





    def return_fxn(self):
        return self.max_columns


    def config(self):
        ####################################################################################################################
        ####################################################################################################################
        if self.svd_config in 'BE':  # max columns(b)

            while True:
                self.max_columns = vui.validate_user_int(f'\nEnter max columns > ', min=1)
                if vui.validate_user_str('Accept max columns? (y/n) > ', 'YN') == 'Y':
                    break
        ####################################################################################################################
        ####################################################################################################################

        ####################################################################################################################
        ####################################################################################################################
        '''
        if self.svd_config in 'CE':  # toggle int_or_bin_only(c)

            while True:
                self.int_or_bin_only = {'Y': True, 'N': False}[
                                            vui.validate_user_str(f"\nUse 'INT' or 'BIN' columns only? (y/n) > ", 'YN')]

                if vui.validate_user_str(f'Accept selection? (y/n) > ', 'YN') == 'Y':
                    break
        '''
        ####################################################################################################################
        ####################################################################################################################

        return self.return_fxn()




if __name__ == '__main__':

    standard_config = 'BYPASS'
    svd_config = 'BYPASS'
    DATA = [
        [1, 2, 3, 4, 5, 6],
        [1, 2, 3, 4, 5, 6],
        [1, 2, 3, 4, 5, 6],
        [1, 2, 3, 4, 5, 6]
    ]
    DATA_HEADER = [['A','B','C','D']]
    rglztn_fctr = 0.5
    conv_kill = 1000
    pct_change = 0.1
    conv_end_method = 'KILL'
    svd_int_or_bin_only = True
    svd_max_columns = 10

    print(type(SVDCoreConfigCode))

    max_columns = \
        SVDCoreConfigCode(svd_config, DATA, DATA_HEADER, svd_max_columns).config()




















