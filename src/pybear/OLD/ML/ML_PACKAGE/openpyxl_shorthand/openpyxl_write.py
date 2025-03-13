from openpyxl import Workbook
from openpyxl.styles import Alignment, Font


def openpyxl_write(WORKBOOK, sheet, row, column, value, horiz='center', vert='center', bold=False):

    if not isinstance(WORKBOOK, Workbook):
        raise ValueError(f"'WORKBOOK' must be an instance of openpyxl.Workbook")

    if not isinstance(bold, bool):
        raise ValueError(f"'bold' must be boolean")

    __ = WORKBOOK[str(sheet)].cell(row, column)

    __.value = value

    __.alignment = Alignment(horizontal=horiz, vertical=vert)

    if bold:
        __.font = Font(bold='True')






