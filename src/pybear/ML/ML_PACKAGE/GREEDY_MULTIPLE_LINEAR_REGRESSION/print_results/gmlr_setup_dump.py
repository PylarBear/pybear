from ML_PACKAGE.openpyxl_shorthand import openpyxl_write as ow


def gmlr_setup_dump(wb, conv_kill, pct_change, conv_end_method, rglztn_type, rglztn_fctr, batch_method, batch_size,
                    type, method, float_only, max_columns, intcpt_col_idx):

    def custom_write(sheet, row, column, value, horiz, vert, bold):
        ow.openpyxl_write(wb, sheet, row, column, value, horiz=horiz, vert=vert, bold=bold)

    wb.create_sheet('GMLR SETUP')

    custom_write('GMLR SETUP', 1, 1, 'GMLR SETUP', 'left', 'center', True)

    row = 3

    custom_write('GMLR SETUP', row, 2, 'CONVERGENCE KILL', 'left', 'center', True)
    custom_write('GMLR SETUP', row, 3, conv_kill, 'left', 'center', False)
    row += 1

    custom_write('GMLR SETUP', row, 2, 'PCT CHANGE', 'left', 'center', True)
    custom_write('GMLR SETUP', row, 3, pct_change, 'left', 'center', False)
    row += 1

    custom_write('GMLR SETUP', row, 2, 'CONVERGENCE END METHOD', 'left', 'center', True)
    custom_write('GMLR SETUP', row, 3, conv_end_method, 'left', 'center', False)
    row += 1

    custom_write('GMLR SETUP', row, 2, 'REGULARIZATION TYPE', 'left', 'center', True)
    custom_write('GMLR SETUP', row, 3, rglztn_type, 'left', 'center', False)
    row += 1

    custom_write('GMLR SETUP', row, 2, 'REGULARIZATION FACTOR', 'left', 'center', True)
    custom_write('GMLR SETUP', row, 3, rglztn_fctr, 'left', 'center', False)
    row += 1

    custom_write('GMLR SETUP', row, 2, 'BATCH METHOD', 'left', 'center', True)
    custom_write('GMLR SETUP', row, 3, batch_method, 'left', 'center', False)
    row += 1

    custom_write('GMLR SETUP', row, 2, 'BATCH SIZE', 'left', 'center', True)
    custom_write('GMLR SETUP', row, 3, batch_size, 'left', 'center', False)
    row += 1

    custom_write('GMLR SETUP', row, 2, 'TYPE', 'left', 'center', True)
    custom_write('GMLR SETUP', row, 3, type, 'left', 'center', False)
    row += 1

    custom_write('GMLR SETUP', row, 2, 'GMLR METHOD', 'left', 'center', True)
    custom_write('GMLR SETUP', row, 3, method, 'left', 'center', False)
    row += 1

    custom_write('GMLR SETUP', row, 2, 'FLOAT ONLY', 'left', 'center', True)
    custom_write('GMLR SETUP', row, 3, float_only, 'left', 'center', False)
    row += 1

    custom_write('GMLR SETUP', row, 2, 'MAX COLUMNS', 'left', 'center', True)
    custom_write('GMLR SETUP', row, 3, max_columns, 'left', 'center', False)
    row += 1

    custom_write('GMLR SETUP', row, 2, 'INTERCEPT INDEX', 'left', 'center', True)
    custom_write('GMLR SETUP', row, 3, intcpt_col_idx, 'left', 'center', False)

    return wb










