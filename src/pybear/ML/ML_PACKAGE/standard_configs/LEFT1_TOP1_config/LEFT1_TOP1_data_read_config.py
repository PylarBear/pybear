import numpy as n, pandas as p, warnings
from read_write_file import filetype as ft
from read_write_file.filetype_package_pair_readers import excel_csv_pandas_reader as ecpr
from read_write_file.generate_full_filename import generate_full_filename as gff
from data_validation import validate_user_input as vui
from general_list_ops import list_select as ls
from ML_PACKAGE.standard_configs import ReadConfigTemplate as rct

'''
    FOR FILE READ CONFIG, NEED TO SPECIFY
    1) FILEPATH
    2) PACKAGE
    3) IF EXCEL, SHEET NAME, COLUMNS TO READ FOR DF
'''


# CALLED BY standard_configs.standard_configs.file_read_standard_configs()
class Left1Top1DataReadConfig(rct.DataReadConfigTemplate):
    def __init__(self, standard_config, data_read_method):
        super().__init__(standard_config, data_read_method)
        self.standard_config = standard_config
        self.method = data_read_method

    def read_methods(self):
        return ['STANDARD']

    def build_data_df(self, filename, filetype, SHEET):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            use_col_select = None
            number_of_rows = None

        header=0
        self.DATA_DF = ecpr.excel_csv_pandas_reader(filename, filetype, SHEET, header, use_col_select, number_of_rows,
                                               suppress_print='Y')
        return self.DATA_DF

    # load_config_template(self) inherited

    # config(self) inherited

    # no_config(self) inherited

    def run(self):

        if self.config() == 'NONE':
            self.OBJECT_DF = self.no_config()

        if self.config() == 'STANDARD':

            filename = gff.generate_full_filename()
            filetype = ft.filetype(filename)

            # ****************************************SET FILE PATH NAME SHEET PARAMETERS*************************************
            print(f'\nCurrent path/file is {filename}')

            SHEET = ls.list_single_select(p.ExcelFile(filename).sheet_names, \
                        'Select sheet', 'value')[0]

            self.DATA_DF = self.build_data_df(filename, filetype, SHEET)

            # DON'T DROP THE LEFTMOST COLUMN HERE, KEEP IT ON SO THAT IT CAN BE EXTRACTED AND
            # DATA_DF IS MODIFIED IN THE RawTargetReadConfig STEP.

        return self.DATA_DF





