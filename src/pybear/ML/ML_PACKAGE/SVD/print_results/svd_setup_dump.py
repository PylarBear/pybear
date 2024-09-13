from ML_PACKAGE.openpyxl_shorthand import openpyxl_write as ow


def svd_setup_dump(wb, svd_max_columns):


    def custom_write(sheet, row, column, value, horiz, vert, bold):
        ow.openpyxl_write(wb, sheet, row, column, value, horiz=horiz, vert=vert, bold=bold)


    wb.create_sheet('SVD SETUP')

    custom_write('SVD SETUP', 1, 1, 'SVD SETUP', 'left', 'center', True)

    row = 3

    # custom_write('SVD SETUP', row, 2, 'INT OF BIN ONLY', 'left', 'center', True)
    # custom_write('SVD SETUP', row, 3, svd_int_or_bin_only, 'left', 'center', False)
    # row += 1

    custom_write('SVD SETUP', row, 2, 'MAX COLUMNS', 'left', 'center', True)
    custom_write('SVD SETUP', row, 3, svd_max_columns, 'left', 'center', False)

    return wb




