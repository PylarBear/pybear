# Authors:
#
#       Bill Sousa
#
# License: BSD 3 clause



import os
from ....data_validation import validate_user_input as vui
from ..generate_full_filename import (
    base_path_select as bps,
    filename_enter as ef,
    file_ext as fe
)


def generate_full_filename():

    while True:
        base_path = bps.base_path_select()
        filename = ef.filename_menu()

        if not '.' in filename: ext = f'{fe.file_ext()}'
        else: ext = ''

        full_filename = os.path.join(base_path, filename) + ext

        if vui.validate_user_str(f'Specified path is {full_filename}   Accept? (y/n) > ', 'YN') == 'Y':
            break

    return full_filename







if __name__ == '__main__':
    _ = generate_full_filename()
    print(f'\nOutput: \n{_}')


