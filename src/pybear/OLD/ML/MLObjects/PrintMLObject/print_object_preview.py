from debug import IdentifyObjectAndPrint as ioap


def print_object_preview(data, name, rows, columns, start_row, start_col, orientation='column', header=''):
    ioap.IdentifyObjectAndPrint(data, name, __name__, rows, columns, start_row, start_col
                                ).run_print_as_df(df_columns=header, orientation=orientation)













