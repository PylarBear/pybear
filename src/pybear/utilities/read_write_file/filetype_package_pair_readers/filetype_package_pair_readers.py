# Authors:
#
#       Bill Sousa
#
# License: BSD 3 clause


import pandas as pd
from ..filetype_package_pair_readers import (
    txt_python_reader as tpr,
    csv_openpyxl_reader as cor,
    excel_openpyxl_reader as eor,
    excel_csv_pandas_reader as ecpr
)


#CALLED BY read_write_file.read_file_config_run
def filetype_package_pair_readers(
    user_package,
    delimiter,
    object_type,
    filename,
    filetype,
    SHEET,
    header,
    use_cols_select,
    number_of_rows,
    csv_openpyxl_stuff,
    excel_openpyxl_stuff
):

    print(f'\nReading full contents of {filename} from disk...')

    if filetype == 'TXT' and user_package == 'PYTHON':
        OBJECT = tpr.txt_python_reader(filename, delimiter, object_type)

    elif filetype == 'CSV' and user_package == 'PANDAS':
        OBJECT = ecpr.excel_csv_pandas_reader(filename, filetype, SHEET, header, use_cols_select, number_of_rows, suppress_print='Y')

    elif filetype == 'EXCEL' and user_package == 'PANDAS':
        OBJECT = ecpr.excel_csv_pandas_reader(filename, filetype, SHEET, header, use_cols_select, number_of_rows, suppress_print='Y')

    elif filetype == 'CSV' and user_package == 'OPENPYXL':
        OBJECT = cor.csv_openpyxl_reader()

    elif filetype == 'EXCEL' and user_package == 'OPENPYXL':
        OBJECT = eor.excel_openpyxl_reader()

    else:
        print(f"INCORRECT FILE TYPE GOING INTO excel_csv_pandas_reader().  FILE TYPE IS {filetype}.  SHOULD BE 'EXCEL' OR 'CSV'.")
        return pd.DataFrame()

    print(f'Done.')

    return OBJECT













