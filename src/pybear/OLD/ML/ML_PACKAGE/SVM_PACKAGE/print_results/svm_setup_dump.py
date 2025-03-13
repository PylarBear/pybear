from ML_PACKAGE.openpyxl_shorthand import openpyxl_write as ow
from ML_PACKAGE.SVM_PACKAGE import svm_error_calc as sec


def svm_setup_dump(wb, margin_type, C, cost_fxn, kernel_fxn, constant, exponent, sigma, alpha_seed, alpha_selection_alg,
    max_passes, tol, SMO_a2_selection_method):


    def custom_write(sheet, row, column, value, horiz, vert, bold):
        ow.openpyxl_write(wb, sheet, row, column, value, horiz=horiz, vert=vert, bold=bold)


    wb.create_sheet('SVM SETUP')

    custom_write('SVM SETUP', 1, 1, 'SVM SETUP', 'left', 'center', True)

    row_counter = 2

    SVM_PARAMS_PRINT1 = dict({
        f'margin type': margin_type,
        f'C': str(C),
        f'cost fxn': sec.cost_functions()[cost_fxn],
        f'kernel fxn': kernel_fxn,
        f'polynomial constant': constant,
        f'polynomial exponent': exponent,
        f'gaussian sigma': sigma
    })

    SVM_PARAMS_PRINT2 = dict({
        f'alpha seed': alpha_seed,
        f'alpha selection alg': alpha_selection_alg,
        f'max passes': max_passes,
        f'tol': tol,
        f'SMO_a2_selection_method': SMO_a2_selection_method
    })

    row_counter2 = row_counter
    for _ in SVM_PARAMS_PRINT1:
        custom_write('SVM SETUP', row_counter2, 2, _, 'left', 'center', True)
        custom_write('SVM SETUP', row_counter2, 5, SVM_PARAMS_PRINT1[_], 'left', 'center', False)
        row_counter2 += 1

    for _ in SVM_PARAMS_PRINT2:
        custom_write('SVM SETUP', row_counter, 8, _, 'left', 'center', True)
        custom_write('SVM SETUP', row_counter, 11, SVM_PARAMS_PRINT2[_], 'left', 'center', False)
        row_counter +=1


    return wb









