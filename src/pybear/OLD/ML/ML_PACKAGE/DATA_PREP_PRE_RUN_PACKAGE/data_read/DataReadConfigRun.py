import sys
from data_validation import validate_user_input as vui
from read_write_file import package_select as ps
import read_write_file.filetype as ft
from read_write_file.filetype_package_pair_configs import filetype_package_pair_configs as fppc
from read_write_file.filetype_package_pair_readers import filetype_package_pair_readers as fppr
from read_write_file.generate_full_filename import generate_full_filename as gff
from ML_PACKAGE.GENERIC_PRINT import print_post_run_options as ppro

'''accept config / continue(a) select path/filename(f) select package(p) reconfigure all(z) 
filetype/package pair config(b) choose standard config(s)
'''

user_manual_or_std = 'Z'

#CALLED BY class_DataTargetReferenceTestReadBuildDF.DataBuild
class DataReadConfigRun:
    def __init__(self, user_manual_or_std, standard_config, data_read_method, DATA_DF):
        self.user_manual_or_std = user_manual_or_std
        self.standard_config = standard_config
        self.method = data_read_method
        self.DATA_DF = DATA_DF











    def standard_config_source(self):
        from ML_PACKAGE.standard_configs import standard_configs as sc
        return sc.data_read_standard_configs(self.standard_config, self.method)
        # DONT CHANGE THIS STANDARDIZED W RawTargetReadConfigRun & RVReadConfigRun

    def config_run(self):
        while True:

            while True:
                if self.user_manual_or_std == 'S':
                    self.DATA_DF = self.standard_config_source()
                    final_calc_bypass = 'Y'
                    if self.method != '':
                        self.user_manual_or_std = 'A'
                else:
                    final_calc_bypass = 'N'

                if self.user_manual_or_std in 'FZ':
                    while True:
                        filename = gff.generate_full_filename()
                        filetype = ft.filetype(filename)
                        if filetype != 'NONE':
                            break

                if self.user_manual_or_std in 'PFZ':
                    user_package = ps.package_select(filename, filetype, '')

                if self.user_manual_or_std in 'BPFZ':
                    user_package, delimiter, object_type, filename, filetype, SHEET, header, use_cols_select, \
                    number_of_rows, csv_openpyxl_stuff, excel_openpyxl_stuff = \
                        fppc.filetype_package_pair_configs(user_package, filename, filetype)


                if self.user_manual_or_std == 'A':
                    break

                READ_FILE_MENU_OPTIONS = \
                    ['accept config / continue(a)', 'filetype/package pair config(b)', 'config path/filename(f)',
                              'config package(p)', 'choose standard config(s)', 'reconfigure all(z)']

                ppro.TopLevelMenuPrint(READ_FILE_MENU_OPTIONS, 'ABFPSZ')

                self.user_manual_or_std = vui.validate_user_str(' > ', 'ABFPSZ')

            if final_calc_bypass == 'N':
                try:
                    self.DATA_DF = fppr.filetype_package_pair_readers(user_package, delimiter, object_type, filename, filetype,
                                SHEET, header, use_cols_select, number_of_rows, csv_openpyxl_stuff, excel_openpyxl_stuff)
                    break

                except:
                    print(f'Error encountered trying to read {filename}')
                    if vui.validate_user_str('Modify filename(m) or abort(a)? > ', 'MA') == 'M':
                        self.user_manual_or_std = 'F'
                    else:
                        sys.exit(f'User terminated.')
            else:
                break

        return self.DATA_DF

    def final_output(self):
        # DATA, RV, AND RAW TARGET ALL SHARE THE SAME CLASS TargetReferenceDataReadBuild() AND SINCE
        # RV AND RT NEED 2 OUTPUTS (ONE FOR THEIR OBJECT, PLUS ONE TO RETURN DATA_DF IF COLUMN-DROPS WERE
        # MADE TO IT) THE CLASS IS MADE TO ACCOMODATE TWO OUTPUTS, SO MUST HAVE 2 OUTPUTS HERE, HENCE DUMMY
        return 'DUM', self.config_run()















