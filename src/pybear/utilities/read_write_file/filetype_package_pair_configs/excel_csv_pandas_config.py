# Authors:
#
#       Bill Sousa
#
# License: BSD 3 clause


import sys
import pandas as pd
from .. import excel_sheet_select as ess
from data_validation import validate_user_input as vui
from general_list_ops import list_select as ls
from ..generate_full_filename import generate_full_filename as gff


def excel_csv_pandas_config(filename, filetype):

    print(f'\nCurrent path/file is {filename}')

    print(f'\nCONFIGURING DATAFRAME CONSTRUCTION')

    while True:
        try:
            if filetype == 'EXCEL':
                print(f'\nSELECT SHEET')
                SHEET = ess.excel_sheet_select(filename)
                DF_PING = pd.DataFrame(pd.read_excel(filename, header=None, index_col=None, nrows=10, sheet_name=SHEET))

            elif filetype == 'CSV':
                SHEET = ''
                DF_PING = pd.DataFrame(pd.read_csv(filename, header=None, index_col=None, nrows=10))

            print(f'\nFile ping OK.\n')
            break

        except:
            print(f'Error encountered trying to read {filename}')
            if vui.validate_user_str('Modify filename(y) or abort(n)? (y/n) > ', 'YN') == 'Y':
                filename = gff.generate_full_filename()
                continue
            else:
                sys.exit(f'User terminated.')


    while True:

        print(f'FIRST 10 ROWS:')
        print(DF_PING)
        print(f'\nSELECT HEADER')
        user_option_select = ls.list_single_select(['None','Top row (idx=0)', 'Top 2 rows', 'Other'],
                                                   'Select option', 'idx')[0]
        if user_option_select == 0:
            header = None
        elif user_option_select == 1:
            header = 0
        elif user_option_select == 2:
            header = [0,1]
        elif user_option_select == 3:
            header_start_row = vui.validate_user_int(f'Enter header start row (0 idx) > ', min=0, max=10)
            header_end_row = vui.validate_user_int(f'Enter header end row (0 idx) > ', min=header_start_row, max=10)
            header = [_ for _ in range(header_start_row, header_end_row+1)]
        else:
            raise Exception(f'\nTHERE IS AN ESCAPE IN excel_csv_pandas_config HEADER SELECTION.')

        print('\nHEADER SELECTION:')
        print(header)
        if vui.validate_user_str(f'Accept header selection? (y/n) > ', 'YN') == 'Y':
            break

    print(f'\nPREVIEW OF DF LOOKS LIKE:')
    if filetype == 'EXCEL':
        print(pd.DataFrame(pd.read_excel(filename, header=header, index_col=None, nrows=10, sheet_name=SHEET)))

    elif filetype == 'CSV':
        print(pd.DataFrame(pd.read_csv(filename, header=header, index_col=None, nrows=10)))

    #
    # print(f'SELECT COLUMNS TO INCLUDE > ')
    use_cols_select = None #ls.DF_column_select2(DF_PING, SHEET, '\nSelect columns to use', 'value')
    number_of_rows = None

    return filename, filetype, SHEET, header, use_cols_select, number_of_rows






