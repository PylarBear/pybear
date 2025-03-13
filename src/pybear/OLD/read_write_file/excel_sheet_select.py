# Authors:
#
#       Bill Sousa
#
# License: BSD 3 clause


import pandas as pd
from ...data_validation import validate_user_input as vui
from ..general_list_ops import list_select as ls
from .generate_full_filename import generate_full_filename as gff


#CALLED BY excel_csv_pandas
def excel_sheet_select(filename):

    xl = pd.ExcelFile(filename)
    xl.sheet_names

    SHEET = ls.list_single_select(xl.sheet_names,'Select sheet','value')[0]

    return SHEET

# import xlrd
# xls = xlrd.open_workbook(
#     r"C:/users/Bill/Documents/Work Stuff/Resume/APPLICATION ANALYSIS - NN.xlsx",
#     on_demand=True)
# print(xls.sheet_names())

