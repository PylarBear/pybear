import pandas as p, warnings
from read_write_file.filetype_package_pair_readers import excel_csv_pandas_reader as ecpr
from ML_PACKAGE.standard_configs import ReadConfigTemplate as rct
from data_validation import validate_user_input as vui
from read_write_file.generate_full_filename import base_path_select as bps


'''
    FOR FILE READ CONFIG, NEED TO SPECIFY
    1) FILEPATH
    2) PACKAGE
    3) IF EXCEL, SHEET NAME, COLUMNS TO READ FOR DF
'''

class AADataReadConfig(rct.DataReadConfigTemplate):
    def __init__(self, standard_config, data_read_method):
        self.standard_config = standard_config
        self.method = data_read_method


    def read_methods(self):
        return ['STANDARD']

    # load_config_template(self) inherited

    # config(self) inherited

    # no_config(self) inherited

    def run(self):

        if self.config() == 'NONE':
            self.OBJECT_DF = self.no_config()

        elif self.config() == 'STANDARD':
            while True:
                basepath = bps.base_path_select()
                filename = 'APPLICATION ANALYSIS - NN.xlsx'
                full_path = basepath + filename
                if vui.validate_user_str(f'\nCurrent path/file is {full_path}   Accept? (y/n) > ', 'YN') == 'Y':
                    break

            filetype = 'EXCEL'

            # ****************************************SET FILE PATH NAME SHEET PARAMETERS*************************************

            print(f'\nReading {filename} from disk...')

            SHEET = 'Sheet2'
            header = 0
            number_of_rows = None
            use_col_select = [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # DETERMINE NUMBER OF VALID ROWS IN APPS, EXCLUDE ALL THE GIBBERISH BELOW THE DATA IN THE FILE
                self.DATA_DF = ecpr.excel_csv_pandas_reader(full_path, filetype, SHEET, header, use_col_select,
                                                            number_of_rows, suppress_print='Y')

            first_value = self.DATA_DF['ROW_ID'][0]
            for idx in range(1,len(self.DATA_DF)):
                second_value = self.DATA_DF['ROW_ID'][idx]
                if second_value != first_value + 1:
                    self.DATA_DF = self.DATA_DF.iloc[:idx,:]
                    break
                else: first_value = second_value
            del first_value, second_value, header, number_of_rows, use_col_select, SHEET, filename, full_path

            # SHEET = 'NAICS'
            # NAICS_DF = ecpr.excel_csv_pandas_reader(path_file, filetype, SHEET, use_col_select=None,
            #                                         number_of_rows=None, suppress_print='Y')
            #
            # SHEET = 'REGION'
            # REGION_DF = ecpr.excel_csv_pandas_reader(path_file, filetype, SHEET, use_col_select=None,
            #                                         number_of_rows=None, suppress_print='Y')
            #
            # SHEET = 'MARKETCAP'
            # MARKETCAP_DF = ecpr.excel_csv_pandas_reader(path_file, filetype, SHEET, use_col_select=None,
            #                                         number_of_rows=None, suppress_print='Y')

            print(f'Done.')

            return self.DATA_DF#, NAICS_DF, REGION_DF, MARKETCAP_DF





