# Authors:
#
#       Bill Sousa
#
# License: BSD 3 clause




from general_list_ops import list_select as ls
from data_validation import validate_user_input as vui
from .. import paths



def base_path_select():

    print('\nSELECT BASEPATH')
    base_path = ls.list_single_select(paths.paths(), 'Select base path', 'value')[0]

    if base_path == 'MANUAL ENTRY':
        while True:
            _ = input(f'Enter path without filename > ')
            if vui.validate_user_str(f'User entered {_}, accept? (y/n) > ', 'YN') == 'Y':
                base_path = _
                break

    return base_path




if __name__ == '__main__':
    __ = base_path_select()
    print(f'\nOutput: \n"{__}"')



