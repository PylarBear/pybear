from ML_PACKAGE.openpyxl_shorthand import openpyxl_write as ow


def mlregression_setup_dump(wb, rglztn_type, rglztn_fctr, batch_method, batch_size, intcpt_col_idx):


    def custom_write(sheet, row, column, value, horiz, vert, bold):
        ow.openpyxl_write(wb, sheet, row, column, value, horiz=horiz, vert=vert, bold=bold)


    wb.create_sheet('MLRegression SETUP')

    custom_write('MLRegression SETUP', 1, 1, 'MLRegression SETUP', 'left', 'center', True)

    row = 3

    custom_write('MLRegression SETUP', row, 2, 'COST FUNCTION', 'left', 'center', True)
    custom_write('MLRegression SETUP', row, 3, f'Total Sum of Squares (always)', 'left', 'center', False)
    row += 1

    custom_write('MLRegression SETUP', row, 2, 'REGULARIZATION TYPE', 'left', 'center', True)
    custom_write('MLRegression SETUP', row, 3, rglztn_type, 'left', 'center', False)
    row += 1

    custom_write('MLRegression SETUP', row, 2, 'REGULARIZATION FACTOR', 'left', 'center', True)
    custom_write('MLRegression SETUP', row, 3, rglztn_fctr, 'left', 'center', False)
    row += 1

    custom_write('MLRegression SETUP', row, 2, 'BATCH METHOD', 'left', 'center', True)
    custom_write('MLRegression SETUP', row, 3, batch_method, 'left', 'center', False)
    row += 1

    custom_write('MLRegression SETUP', row, 2, 'BATCH SIZE', 'left', 'center', True)
    custom_write('MLRegression SETUP', row, 3, batch_size, 'left', 'center', False)
    row += 1

    custom_write('MLRegression SETUP', row, 2, 'INTERCEPT INDEX', 'left', 'center', True)
    custom_write('MLRegression SETUP', row, 3, intcpt_col_idx, 'left', 'center', False)

    return wb










