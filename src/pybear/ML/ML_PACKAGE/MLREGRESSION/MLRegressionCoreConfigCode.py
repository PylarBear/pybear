from data_validation import validate_user_input as vui
import sparse_dict as sd
from general_data_ops import get_shape as gs
from general_list_ops import list_select as ls


class MLRegressionCoreConfigCode:
    def __init__(self, mlr_config, DATA, DATA_HEADER, data_run_orientation, rglztn_type, rglztn_fctr, batch_method, batch_size):

        self.mlr_config = mlr_config
        self.DATA = DATA
        self.DATA_HEADER = DATA_HEADER
        self.data_run_orientation = data_run_orientation
        self.rglztn_type = rglztn_type
        self.rglztn_fctr = rglztn_fctr
        self.batch_method = batch_method
        self.batch_size = batch_size


    def return_fxn(self):
        return self.rglztn_type, self.rglztn_fctr, self.batch_method, self.batch_size


    def config(self):

        # 11/15/22 OPTIONS COMING IN ARE 'RWBHC'

        ####################################################################################################################
        ###################################################################################################################
        if self.mlr_config in 'BCE':    # 'regularization type (b)'
            print(f'\n*** RIDGE IS CURRENTLY THE ONLY REGULARIZATION AVAILABLE FOR MLRegression ***\n')
            if self.mlr_config in 'BE':
                self.rglztn_type = dict({'R':'RIDGE', 'N':'NONE'})[
                                        vui.validate_user_str(f'Select reqularization type RIDGE(r), None(n) > ', 'RN')]
                if self.rglztn_type == 'NONE': self.rglztn_fctr = 0
            if self.mlr_config in 'CE' and not self.rglztn_type=='NONE':  # 'regularization factor (c)'
                self.rglztn_fctr = vui.validate_user_float(f'Enter regularization factor > ', min=0, max=1e12)
                self.rglztn_type = 'RIDGE'
        ################################################################################################################
        #################################################################################################################

        ####################################################################################################################
        ###################################################################################################################
        if self.mlr_config in 'DE':    # 'batch method (d)'
            self.batch_method = vui.validate_user_str(f'Select batch method (b)atch (m)inibatch > ', 'BM')
            if self.batch_method == 'B': self.batch_size = gs.get_shape('DATA', self.DATA, self.data_run_orientation)[0]
        ################################################################################################################
        #################################################################################################################

        ####################################################################################################################
        ###################################################################################################################
        if self.mlr_config in 'FE':    # 'batch size (f)'
            if self.batch_method == 'B': pass
            elif self.batch_method == 'M':
                _len = gs.get_shape('DATA', self.DATA, self.data_run_orientation)[0]
                self.batch_size = vui.validate_user_int(f'Select batch size (data has {int(_len)} examples) > ', min=1, max=int(_len))
                del _len
        ################################################################################################################
        #################################################################################################################


        return self.return_fxn()


























