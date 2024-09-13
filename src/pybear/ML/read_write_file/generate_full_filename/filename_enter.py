from data_validation import validate_user_input as vui
from general_list_ops import list_select as ls



def filename_wo_extension():
    file_name = vui.user_entry(f'Enter filename without extension')
    return file_name


def filename_w_extension():
    # BEAR FINISH 6/9/23
    file_name = vui.user_entry(f'Enter filename with extension')
    return file_name


def filename_menu():

    METHODS = [
                "Manual entry with extension",
                "Manual entry without extension"
    ]

    method = ls.list_single_select(METHODS, '\nSELECT ENTRY TYPE', 'idx')[0]

    if method==0: filename = filename_w_extension()
    elif method==1: filename = filename_wo_extension()

    return filename






if __name__ == '__main__':
    print(f'Should prompt for no extension > ')
    __ = filename_wo_extension()
    print(f'\nRETURNED "{__}"')


    print(f'Should prompt for extension > ')
    __ = filename_w_extension()
    print(f'\nRETURNED "{__}"')







