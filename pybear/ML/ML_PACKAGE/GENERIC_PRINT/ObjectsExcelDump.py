from openpyxl import Workbook
from ML_PACKAGE.openpyxl_shorthand import openpyxl_write as ow
from data_validation import validate_user_input as vui
from read_write_file.generate_full_filename import base_path_select as bps, filename_enter as fe


class ObjectsExcelDump:

    def __init__(self, DATA_OBJECTS_TUPLE, object_type):
        self.DATA_OBJECTS_TUPLE = DATA_OBJECTS_TUPLE
        self.SHEET_NAMES = ['DATA', '', 'TARGET', '', 'REFERENCE', '', 'TEST', '']  # NOT ALL SHEETS MAY NOT BE IN DATA_OBJECTS_TUPLE
        self.object_type = object_type   # "BASE" OR "WORKING"
        if self.object_type.upper() not in ['BASE','WORKING']:
            raise TypeError(f'\n*** INVALID object_type "{self.object_type}" in ObjectsExcelDump. ***\n')
        self.OBJ_IDXS = [idx for idx in range(2 * ((len(self.DATA_OBJECTS_TUPLE)-1) // 2)) if idx % 2 == 0]
        self.wb = Workbook()


    def custom_write(self, sheet, row, column, value, horiz, vert, bold):
        ow.openpyxl_write(self.wb, sheet, row, column, value, horiz=horiz, vert=vert, bold=bold)


    def dump(self):
        # ITERATE THRU OBJECTS (NOT HEADERS)
        for obj_idx in self.OBJ_IDXS:
            sheet_name = f'{self.object_type} {self.SHEET_NAMES[obj_idx]}'
            if obj_idx == 0:
                self.wb.active.title = sheet_name
            else:
                self.wb.create_sheet(f'{sheet_name}')

            # PUT HEADER AT TOP OF SHEET
            for col_idx in range(len(self.DATA_OBJECTS_TUPLE[obj_idx + 1][0])):
                self.custom_write(sheet_name, 1, col_idx+1, self.DATA_OBJECTS_TUPLE[obj_idx + 1][0][col_idx], 'center',
                                  'center', True)

            # FILL IN THE REST OF THE DATA
            for col_idx in range(len(self.DATA_OBJECTS_TUPLE[obj_idx])):
                for row_idx in range(len(self.DATA_OBJECTS_TUPLE[obj_idx][col_idx])):
                    self.custom_write(sheet_name, row_idx+2, col_idx+1, self.DATA_OBJECTS_TUPLE[obj_idx][col_idx][row_idx],
                                      'center', 'center', False)


        # SAVE TO USER PATH
        while True:
            base_path = bps.base_path_select()
            file_name = fe.filename_wo_extension()
            full_path = base_path + file_name + '.xlsx'

            try:
                self.wb.save(full_path)
                print(f'\n*** FILE DUMP of {self.object_type} DATA OBJECTS to {full_path} SUCCESSFUL! ***\n')
                break
            except:
                print(f'\n*** UNABLE TO SAVE FILE TO PATH {full_path}! ***\n')
                if vui.validate_user_str(f'Try again(t), abort(a) > ', 'TA') == 'T': continue
                else: print(f'\n*** OBJECT DUMP TO EXCEL NOT COMPLETED! ***\n'); break









