import numpy as n, pandas as p, warnings
from read_write_file.filetype_package_pair_readers import excel_csv_pandas_reader as ecpr
from data_validation import validate_user_input as vui
from ML_PACKAGE.standard_configs import ReadConfigTemplate as rct
from read_write_file.generate_full_filename import base_path_select as bps


'''
    FOR FILE READ CONFIG, NEED TO SPECIFY
    1) FILEPATH
    2) PACKAGE
    3) IF EXCEL, SHEET NAME, COLUMNS TO READ FOR DF
'''

# THE ONLY REASON THIS IS A CLASS AND NOT A FUNCTION IS THAT IN THE CASE OF POWERBALL, ORIGINAL TARGET VECTOR
# CONSTRUCTION MACHINERY CREATES TARGET VECTOR FROM FULL DATA_DF PULL, NOT A SEPARATE RAW TARGET SOURCE,
# SO NOW THAT WE'RE DOING THIS IN SEPARATE PULLS, IT'S MORE CONVIENENT TO TURN POWERBALL_DATA_DF INTO A CLASS
# AND JUST CREATE A CHILD POWERBALL_RAW_TARGET_DF CLASS TO DO THE SAME PULL AGAIN, JUST IN A DIFFERENT PLACE UNDER
# A DIFFERENT NAME


#CALLED BY standard_configs.standard_configs.file_read_standard_configs()
class PowerballDataReadConfigRun(rct.DataReadConfigTemplate):
    def __init__(self, standard_config, data_read_method):
        self.standard_config = standard_config
        self.method = data_read_method
        super().__init__(standard_config, data_read_method)

    def read_methods(self):
        return ['STANDARD',
                'SORTED',
                'STANDARD - DATE']

    # config(self) inherited

    # no_config(self) inherited

    def build_data_df(self, filename, filetype, SHEET, use_col_select):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            number_of_rows = None

        header=0
        self.DATA_DF = ecpr.excel_csv_pandas_reader(filename, filetype, SHEET, header, use_col_select, number_of_rows,
                                              suppress_print='Y')
        return self.DATA_DF

    def powerball_yyyy_format(self):
        ych = 'YYYY'  # ych = year_column_header
        if vui.validate_user_str(f'\nUse YY(a) or YYYY(b) year format > ', 'AB') == 'A':
            for item_idx in range(len(self.DATA_DF[ych])):
                self.DATA_DF[ych][item_idx] = int(str(self.DATA_DF[ych][item_idx])[-2:])

            COLUMNS = [_ for _ in self.DATA_DF]
            COLUMNS[COLUMNS.index('YYYY')] = 'YY'
            self.DATA_DF.columns = COLUMNS

        else:
            for item_idx in range(len(self.DATA_DF[ych])):
                self.DATA_DF[ych][item_idx] = int('20' + str(self.DATA_DF[item_idx])[-2:])
        return self.DATA_DF

    def run(self):

        basepath = bps.base_path_select()
        filename = "POWERBALL.xlsm"
        filetype = 'EXCEL'

        # ****************************************SET FILE PATH NAME SHEET PARAMETERS*************************************
        print(f'\nCurrent path/file is {basepath+filename}')

        SHEET = 'Sheet1'

        if self.config() == 'NONE':
            self.DATA_DF = self.no_config()

        elif self.config() == 'STANDARD':

            use_col_select = [5, 6, 7, 8, 9, 10]
            self.DATA_DF = self.build_data_df(basepath+filename, filetype, SHEET, use_col_select)

        elif self.config() == 'SORTED':

            use_col_select = [5, 6, 7, 8, 9, 10]
            DATA_DF = self.build_data_df(basepath+filename, filetype, SHEET, use_col_select)

            HEADER = [header for header in self.DATA_DF]

            NUMPY = n.array(self.DATA_DF, dtype=int)

            SORTED_DRAW = []
            for DRAW in NUMPY:
                SORTED_DRAW.append(sorted(DRAW[:5]))
                SORTED_DRAW[-1].append(DRAW[-1])

            self.DATA_DF = p.DataFrame(columns=HEADER, data=SORTED_DRAW, dtype=int)


        elif self.config() == 'STANDARD - DATE':

            use_col_select = [1, 2, 3, 5, 6, 7, 8, 9, 10]
            self.DATA_DF = self.build_data_df(basepath+filename, filetype, SHEET, use_col_select)

            self.DATA_DF = self.powerball_yyyy_format()

        return self.DATA_DF





