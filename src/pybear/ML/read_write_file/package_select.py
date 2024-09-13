from general_list_ops import list_select as ls
from data_validation import validate_user_input as vui

#CALLED by read_write_file.read_file_config_run
def package_select(filename, filetype, user_package):

    print(f'\nSELECTING PACKAGE TO READ {filename}')

    filetype = filetype.upper()
    user_package = user_package.upper()

    PS_DICT = {
        'EXCEL': ['PANDAS','OPENPYXL','OTHER'],
        'CSV': ['PANDAS','OPENPYXL','OTHER'],
        'TXT': ['PYTHON','OTHER']
    }
    READ_WITH_MASTER = ['PANDAS','OPENPYXL','PYTHON','OTHER']

    if filetype in ['EXCEL', 'XLS', 'XLSM', 'XLSX']:
        filetype = 'EXCEL'
        READ_WITH = PS_DICT['EXCEL']
    elif filetype in ['CSV']:
        filetype = 'CSV'
        READ_WITH = PS_DICT['CSV']
    elif filetype in ['TXT']:
        filetype = 'TXT'
        READ_WITH = PS_DICT['TXT']
    else:
        filetype = 'UNRECOGNIZED'
        READ_WITH = READ_WITH_MASTER

    override = 'Y'
    if user_package == '':
        print(f'\nUser package is not selected.')
    elif user_package in READ_WITH:
        override = vui.validate_user_str(f'\nFile type and selected package match.  Override? (y/n) > ', 'YN')
        READ_WITH = READ_WITH_MASTER
    elif user_package not in READ_WITH and user_package in READ_WITH_MASTER:
        print(f'\nUser entered non-standard package {user_package} for file type {filetype}.')
        override = vui.validate_user_str(f'Change package? (y/n) > ', 'YN')
    elif user_package not in READ_WITH_MASTER:
        print(f'\nCurrent package is unrecognized or unavailable.')

    while True:
        if override == 'Y':
            print(f'\nFiletype is {filetype}\nUser entered package: {user_package}\nRecommended packages: {", ".join(READ_WITH)}')
            user_package = ls.list_single_select(READ_WITH_MASTER, f'Select read package', 'value')[0]
            break

        if user_package == 'OTHER':
            print(f'\n*** "OTHER" READ FUNCTIONALITY NOT AVAILABLE YET ***\n')
            # user_package = vui.user_entry(f'User selected {user_package}.  Enter package')
            continue

    return user_package.upper()
