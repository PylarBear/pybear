import numpy as np

from ML_PACKAGE.openpyxl_shorthand import openpyxl_write as ow


def general_dev_results_dump(wb, DEV_ERROR, RGLZTN_FACTORS):

    def custom_write(sheet, row, column, value, horiz, vert, bold):
        ow.openpyxl_write(wb, sheet, row, column, value, horiz=horiz, vert=vert, bold=bold)

        # EACH [] IN DEV_ERROR HOLDS ERRORS FOR ONE rglztn_fctr, AND THERE IS A [] FOR EACH OF THE PARTITIONS

    # wb.active.title = 'BLANK'

    wb.create_sheet('DEV RESULTS')

    custom_write('DEV RESULTS', 1, 1, 'DEV RESULTS', 'left', 'center', True)

    row_counter = 3

    custom_write('DEV RESULTS', row_counter, 2, 'REGULARIZATION', 'center', 'center', True)
    custom_write('DEV RESULTS', row_counter, (len(DEV_ERROR[0])+1) // 2 + 3, 'DEV SET', 'center', 'center', True)
    row_counter += 1
    custom_write('DEV RESULTS', row_counter, 2, 'FACTOR', 'center', 'center', True)
    col_counter = 4
    for dev_set_idx in range(1, len(DEV_ERROR[0]) + 1):
        custom_write('DEV RESULTS', row_counter, col_counter, dev_set_idx, 'center', 'center', True)
        col_counter += 1
    custom_write('DEV RESULTS', row_counter, col_counter, 'AVERAGE', 'center', 'center', True)

    row_counter += 1

    for idx in range(len(DEV_ERROR)):

        custom_write('DEV RESULTS', row_counter, 2, RGLZTN_FACTORS[idx], 'center', 'center', True)

        col_idx = 4
        for inner_idx in range(len(DEV_ERROR[0])):
            custom_write('DEV RESULTS', row_counter, col_idx, DEV_ERROR[idx][inner_idx], 'center', 'center', False)
            col_idx += 1

        custom_write('DEV RESULTS', row_counter, col_idx, np.average(DEV_ERROR[idx]), 'center', 'center', False)

        row_counter += 1

    return wb




