import numpy as np, pandas as pd
from read_write_file.generate_full_filename import base_path_select as bps, filename_enter as fe




###################################################################################################################
# FUNCTION FOR HANDLING DUMP OF FULL OR GOOD RESULTS TO EXCEL #####################################################

def train_results_excel_dump(RESULTS_DF, sheet_name):

    base_path = bps.base_path_select()

    file_name = fe.filename_wo_extension()

    full_path = base_path + file_name + ".xlsx"
    print(f'\nSaving file to {full_path}....')

    with pd.ExcelWriter(full_path) as writer:
        # index must be True, NotImplementedError: Writing to Excel with MultiIndex columns and no index ('index'=False) is not yet implemented.
        RESULTS_DF.style.set_properties(**{'text-align': 'center'}).to_excel(
            excel_writer=writer, sheet_name=sheet_name, float_format='%.4f', startrow=1, startcol=1, merge_cells=False,
            index=True, na_rep='NaN'
        )

    del RESULTS_DF, file_name, full_path, base_path, sheet_name
    print('Done.')

# END FUNCTION FOR HANDLING DUMP OF FULL OR GOOD RESULTS TO EXCEL #####################################################
#######################################################################################################################






















