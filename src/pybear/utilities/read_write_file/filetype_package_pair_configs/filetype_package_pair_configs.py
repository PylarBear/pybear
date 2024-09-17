# Authors:
#
#       Bill Sousa
#
# License: BSD 3 clause




from ..filetype_package_pair_configs import (
    txt_python_config as tp,
    csv_openpyxl_config as co,
    excel_openpyxl_config as eo,
    excel_csv_pandas_config as ecp
)



#CALLED BY ML_PACKAGE.DATA_PREP_PRE_RUN_PACKAGE.data_read.data_read_config_run
def filetype_package_pair_configs(user_package, filename, filetype):

    delimiter, object_type, SHEET, use_cols_select, header, number_of_rows, csv_openpyxl_stuff, excel_openpyxl_stuff = \
        '','','','','','','',''

    if filetype == 'TXT' and user_package == 'PYTHON':
        delimiter, object_type = tp.txt_python_config()

    elif filetype == 'CSV' and user_package == 'PANDAS':
        filename, filetype, SHEET, header, use_cols_select, number_of_rows = ecp.excel_csv_pandas_config(filename, filetype)

    elif filetype == 'EXCEL' and user_package == 'PANDAS':
        filename, filetype, SHEET, header, use_cols_select, number_of_rows = ecp.excel_csv_pandas_config(filename, filetype)

    elif filetype == 'CSV' and user_package == 'OPENPYXL':
        csv_openpyxl_stuff = co.csv_openpyxl_config()

    elif filetype == 'EXCEL' and user_package == 'OPENPYXL':
        excel_openpyxl_stuff = eo.excel_openpyxl_config()

    return user_package, delimiter, object_type, filename, filetype, SHEET, header, use_cols_select, \
           number_of_rows, csv_openpyxl_stuff, excel_openpyxl_stuff



















