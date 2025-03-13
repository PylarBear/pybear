# Authors:
#
#       Bill Sousa
#
# License: BSD 3 clause


from ...general_list_ops import list_select as ls
from ....data_validation import validate_user_input as vui



#CALLED BY read_write_file.generate_full_filename.generate_full_filename


def file_ext():

    EXT = [
        '.txt',
        '.csv',
        '.xls',
        '.xlsx',
        '.xlsm',
        'MANUAL ENTRY',
        'NO EXTENSION'
    ]

    print('\nSELECT EXTENSION')

    user_select = ls.list_single_select(EXT, 'Select number of extension', 'value')[0]

    if user_select == 'MANUAL ENTRY':
        ext = f'.' + vui.user_entry('User selected manual entry. Enter extension without period > ')
    elif user_select == 'NO EXTENSION':
        ext = f''
    else:
        ext = user_select

    return ext

