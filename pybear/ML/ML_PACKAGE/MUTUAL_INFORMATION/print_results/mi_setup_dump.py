from ML_PACKAGE.openpyxl_shorthand import openpyxl_write as ow


def mi_setup_dump(wb, batch_method, batch_size, mi_int_or_bin_only, mi_max_columns, intcpt_col_idx):


    def custom_write(sheet, row, column, value, horiz, vert, bold):
        ow.openpyxl_write(wb, sheet, row, column, value, horiz=horiz, vert=vert, bold=bold)


    wb.create_sheet('MI SETUP')

    custom_write('MI SETUP', 1, 1, 'MI SETUP', 'left', 'center', True)

    row = 3

    custom_write('MI SETUP', row, 2, 'BATCH METHOD', 'left', 'center', True)
    custom_write('MI SETUP', row, 3, batch_method, 'left', 'center', False)
    row += 1

    custom_write('MI SETUP', row, 2, 'BATCH SIZE', 'left', 'center', True)
    custom_write('MI SETUP', row, 3, batch_size, 'left', 'center', False)
    row += 1

    custom_write('MI SETUP', row, 2, 'INT OR BIN ONLY', 'left', 'center', True)
    custom_write('MI SETUP', row, 3, mi_int_or_bin_only, 'left', 'center', False)
    row += 1

    custom_write('MI SETUP', row, 2, 'MAX COLUMNS', 'left', 'center', True)
    custom_write('MI SETUP', row, 3, mi_max_columns, 'left', 'center', False)
    row += 1

    custom_write('MI SETUP', row, 2, 'INTERCEPT COLUMN INDEX', 'left', 'center', True)
    custom_write('MI SETUP', row, 3, intcpt_col_idx, 'left', 'center', False)

    return wb










