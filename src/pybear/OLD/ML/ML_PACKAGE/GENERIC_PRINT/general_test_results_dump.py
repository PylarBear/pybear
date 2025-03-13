from ML_PACKAGE.openpyxl_shorthand import openpyxl_write as ow


def general_test_results_dump(wb, TEST_RESULTS_DF_OBJECT, DISPLAY_COLUMNS, display_criteria, display_rows):

    def custom_write(sheet, row, column, value, horiz, vert, bold):
        ow.openpyxl_write(wb, sheet, row, column, value, horiz=horiz, vert=vert, bold=bold)

    wb.create_sheet('TEST RESULTS')

    custom_write('TEST RESULTS', 1, 1, 'TEST RESULTS', 'left', 'center', True)

    'HBTA'

    row_counter = 3
    column_counter = 2

    if display_criteria == 'A':
        description = 'ALL RESULTS'
        start_row = 0
        end_row = len(TEST_RESULTS_DF_OBJECT)
    elif display_criteria in 'HB':
        description = f'TOP {display_rows}'
        start_row = 0
        end_row = display_rows
    elif display_criteria == 'T':
        description = f'BOTTOM {display_rows}'
        start_row = len(TEST_RESULTS_DF_OBJECT) - display_rows
        end_row = len(TEST_RESULTS_DF_OBJECT)

    custom_write('TEST RESULTS', row_counter, 1, description, 'left', 'center', True)

    # row_counter += 1


    # WRITE HEADER
    column_counter = 1
    for col_idx in range(len(DISPLAY_COLUMNS)):
        column_counter += 1
        custom_write('TEST RESULTS', row_counter, column_counter, [*TEST_RESULTS_DF_OBJECT.keys()][DISPLAY_COLUMNS[col_idx]],
                     'center', 'center', True)
    column_counter = 2

    for row_idx in range(start_row, end_row):
        row_counter += 1
        column_counter = 1  # GETS 1 ADDED FIRST THING BELOW
        for col_idx in range(len(DISPLAY_COLUMNS)):
            column_counter += 1

            custom_write('TEST RESULTS', row_counter, column_counter,
                         TEST_RESULTS_DF_OBJECT[[*TEST_RESULTS_DF_OBJECT.keys()][DISPLAY_COLUMNS[col_idx]]][row_idx],
                         'center', 'center', False)

    column_counter = 2

    if display_criteria == 'B':
        row_counter += 2
        custom_write('TEST RESULTS', row_counter, 1, f'BOTTOM {display_rows}', 'left', 'center', True)
        start_row = len(TEST_RESULTS_DF_OBJECT) - display_rows
        end_row = len(TEST_RESULTS_DF_OBJECT)

        for row_idx in range(start_row, end_row):
            row_counter += 1
            column_counter = 1  # GETS 1 ADDED FIRST THING BELOW
            for column in DISPLAY_COLUMNS:
                column_counter += 1
                custom_write('TEST RESULTS', row_counter, column_counter,
                             TEST_RESULTS_DF_OBJECT[[*TEST_RESULTS_DF_OBJECT.keys()][column]][row_idx],
                             'center', 'center', False)

    return wb
















