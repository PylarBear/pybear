# Authors:
#
#       Bill Sousa
#
# License: BSD 3 clause


import pandas as pd
from ....data_validation import validate_user_input as vui



#CALLED BY read_write_file.filetype_package_pair_readers.filetype_package_pair_readers
def excel_csv_pandas_reader(
        filename, filetype, SHEET, header, use_cols_select, number_of_rows,
        suppress_print='N'
):

    if filetype.upper() == 'EXCEL':
        EXCEL_CSV_DF = pd.DataFrame(
            pd.read_excel(filename, header=header, sheet_name=SHEET,
            index_col=None, nrows=number_of_rows,
            usecols=use_cols_select)
        )

    elif filetype.upper() == 'CSV':

        EXCEL_CSV_DF = pd.DataFrame(
            pd.read_csv(filename, header=header,
            index_col=None, nrows=number_of_rows,
            usecols=use_cols_select)
        )

    else:
        print(f"INCORRECT FILE TYPE GOING INTO excel_csv_pandas_reader().  FILE TYPE IS {filetype}.  SHOULD BE 'EXCEL' OR 'CSV'.")
        return pd.DataFrame()

    if 'LIST' in str(type(header)).upper():  # IF header IS LIST THEN LEN(header) IS > 1, OTHERWISE IT IS A SINGLE INT
        if vui.validate_user_str(f'Merge header rows? (y/n) > ', 'YN') == 'Y':

            import textwrap

            # PIZZA 24_04_09_16_01_00 THIS WAS ONCE ITS OWN MODULE BUT WAS ONLY CALLED HERE
            # THIS CAN BE MELDED INTO THE CURRENT LANDSCAPE AS NEEDED
            def pandas_merge_header_wrap(DF_OBJECT, wrap_width=15):
                COLUMNS = [_ for _ in DF_OBJECT]

                if 'tuple' in str(type(COLUMNS[0])):
                    NEW_COLUMNS = []
                    for idx in range(len(COLUMNS)):
                        dum_str = ''
                        for thing in COLUMNS[idx]:
                            dum_str += thing + ' '
                        dum_str.strip()
                        NEW_COLUMNS.append(textwrap.wrap(dum_str, width=wrap_width)[0])
                else:
                    NEW_COLUMNS = [textwrap.wrap(_, width=wrap_width)[0] for _ in COLUMNS]

                DF_OBJECT.columns = NEW_COLUMNS
                DF_OBJECT.style.set_table_styles([dict(selector="th", props=[('max-width', '100px')])])

                return DF_OBJECT

            pandas_merge_header_wrap(EXCEL_CSV_DF, wrap_width=15)

            del pandas_merge_header_wrap

    if suppress_print == 'N':
        print(f'DataFrame looks like:')
        print(EXCEL_CSV_DF.head(10))

    return EXCEL_CSV_DF



if __name__ == '__main__':
    from ..generate_full_filename import base_path_select as bps

    basepath = bps.base_path_select()
    filename = basepath + r'rGDP.csv'

    SHEET = 'rGDP'
    filetype = 'EXCEL'

    excel_csv_pandas_reader(
        filename,
        filetype,
        SHEET,
        header,
        use_cols_select,
        number_of_rows,
        suppress_print='N'
    )