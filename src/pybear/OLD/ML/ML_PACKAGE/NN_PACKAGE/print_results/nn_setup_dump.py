from ML_PACKAGE.openpyxl_shorthand import openpyxl_write as ow
from ML_PACKAGE.NN_PACKAGE.gd_run import error_calc as ec


def nn_setup_dump(wb, NEURONS, nodes, activation_constant, aon_base_path, aon_filename, cost_fxn, SELECT_LINK_FXN,
            batch_method, BATCH_SIZE, gd_method, conv_method, lr_method, LEARNING_RATE, momentum_weight, rglztn_type,
            rglztn_fctr, conv_kill, pct_change, conv_end_method, gd_iterations, non_neg_coeffs, allow_summary_print,
            summary_print_interval, iteration):


    def custom_write(sheet, row, column, value, horiz, vert, bold):
        ow.openpyxl_write(wb, sheet, row, column, value, horiz=horiz, vert=vert, bold=bold)


    wb.create_sheet('NN SETUP')

    custom_write('NN SETUP', 1, 1, 'NN SETUP', 'left', 'center', True)

    row_counter = 2

    lg = FIRST_TWO_ROWS = [
                ['', 'LINK', '', 'DESCENT', 'LEARNING', 'ACTIVATION'],
                ['NODE', 'FXN', 'NEURONS', 'METHOD', 'RATE (FIRST 5)', 'CONSTANT']
            ]

    for row_idx in range(len(FIRST_TWO_ROWS)):
        row_counter += 1
        col_idx = 2
        for word_idx in range(len(FIRST_TWO_ROWS[row_idx])):
            custom_write('NN SETUP', row_counter, col_idx, FIRST_TWO_ROWS[row_idx][word_idx], 'center', 'center', True)
            if col_idx == 5 or col_idx == 8: col_idx += 3
            else: col_idx += 1

    row_counter += 1

    LR_DUM = [[f'{_:.5g}' for _ in __] for __ in LEARNING_RATE[:5]]  # LEARNING_RATE MODIFIER FOR DISPLAY
    for idx in range(nodes):
        custom_write('NN SETUP', row_counter, 2, idx + 1, 'center', 'center', False)
        custom_write('NN SETUP', row_counter, 3, SELECT_LINK_FXN[idx], 'center', 'center', False)
        custom_write('NN SETUP', row_counter, 4, NEURONS[idx], 'center', 'center', False)
        custom_write('NN SETUP', row_counter, 5, dict({'G': 'GRADIENT', 'C': 'COORDINATE'})[gd_method], 'center', 'center', False)
        col_idx = 5
        for lr in LR_DUM[idx][:5]:
            col_idx += 1
            custom_write('NN SETUP', row_counter, col_idx, lr, 'center', 'center', False)
        col_idx += 1
        custom_write('NN SETUP', row_counter, col_idx, activation_constant, 'center', 'center', False)
        row_counter += 1

    row_counter += 2

    NN_PARAMS_PRINT1 = dict({
        f'cost_fxn': ec.cost_functions()[cost_fxn.lower()],  # 5/2/23 BEAR COP OUT SOLUTION, DICT IS L-CASE, cost_fxn IS U_CASE, COULDNT FIND WHERE IT IS BECOMING U-CASE
        f'gd_iterations': gd_iterations,
        f'gd_method': dict({'G': 'GRADIENT', 'C': 'COORDINATE'})[gd_method],
        f'conv_method': dict({'G': 'GRADIENT', 'R': 'RMS', 'A':'ADAM', 'N': 'NEWTON'})[conv_method],
        f'lr_method': dict({'C': 'CONSTANT', 'S': 'CAUCHY'})[lr_method],
        f'batch_method': dict({'S': 'STOCHASTIC', 'B': 'BATCH', 'M': 'MINI-BATCH'})[batch_method],
        f'regularization_type': rglztn_type,
        f'regularization_fctr': rglztn_fctr,
        f'momentum_weight': momentum_weight
    })

    NN_PARAMS_PRINT2 = dict({
        f'conv_kill': conv_kill,
        f'pct_change': pct_change,
        f'conv_end_method': conv_end_method,
        f'allow_summary_print': allow_summary_print,
        f'summary_print_interval': summary_print_interval,
        f'aon_base_path': aon_base_path,
        f'aon_filename': aon_filename,
        f'non_neg_coeffs': non_neg_coeffs,
        f'iteration': iteration
    })

    row_counter2 = row_counter
    for _ in NN_PARAMS_PRINT1:
        custom_write('NN SETUP', row_counter2, 2, _, 'left', 'center', True)
        custom_write('NN SETUP', row_counter2, 5, NN_PARAMS_PRINT1[_], 'left', 'center', False)
        row_counter2 += 1

    for _ in NN_PARAMS_PRINT2:
        custom_write('NN SETUP', row_counter, 8, _, 'left', 'center', True)
        custom_write('NN SETUP', row_counter, 11, NN_PARAMS_PRINT2[_], 'left', 'center', False)
        row_counter +=1

    row_counter += 2

    custom_write('NN SETUP', row_counter, 2, f'BATCH_SIZE[:10]', 'left', 'center', True)
    col_idx = 4
    for size in BATCH_SIZE[:10]:
        custom_write('NN SETUP', row_counter, col_idx, size, 'center', 'center', False)
        col_idx += 1


    return wb









