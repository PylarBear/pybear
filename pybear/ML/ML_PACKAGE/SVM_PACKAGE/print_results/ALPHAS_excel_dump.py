from general_list_ops import list_select as ls
import openpyxl as xl
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font
from read_write_file.generate_full_filename import base_path_select as bps, filename_enter as fe


#CALLED BY SVMRun
def ALPHAS_excel_dump(ALPHAS):

    # CREATE FILE & SETUP SHEET
    base_path = bps.base_path_select()

    file_name = fe.filename_wo_extension()

    full_path = base_path + file_name + '.xlsx'

    wb = Workbook()

    wb.active.title = 'NODE 0'
    for row_idx in range(len(ALPHAS)):
        wb['ALPHAS'].cell(row_idx+1, 1).value = ALPHAS[row_idx]


    wb.save(full_path)

    print(f'\nALPHAS saved to {base_path+file_name} successfully.\n')













