from ML_PACKAGE.openpyxl_shorthand import openpyxl_write as ow


def svd_train_results_dump(wb, TRAIN_RESULTS):


    def custom_write(sheet, row, column, value, horiz, vert, bold):
        ow.openpyxl_write(wb, sheet, row, column, value, horiz=horiz, vert=vert, bold=bold)


    wb.create_sheet('SVD TRAIN RESULTS')

    row_counter = 1

    custom_write('SVD TRAIN RESULTS', row_counter, 1, 'SVD TRAIN RESULTS', 'left', 'center', True)

    row_counter += 2

    # CREATE HEADER ROW
    for col_idx in range(len(TRAIN_RESULTS)):
        custom_write('SVD TRAIN RESULTS', row_counter, 2 + col_idx, TRAIN_RESULTS[col_idx][0], 'center', 'center', True)

    row_counter += 1

    # FILL IN DATA
    for row_idx in range(1, len(TRAIN_RESULTS[0][1:])+1):
        col_counter = 2
        for col_idx in range(len(TRAIN_RESULTS)):
            if col_idx == 0: align = 'left'
            else: align = 'center'

            custom_write('SVD TRAIN RESULTS', row_counter, col_counter, TRAIN_RESULTS[col_idx][row_idx], align, 'center', False)

            col_counter += 1
        row_counter += 1

    return wb
