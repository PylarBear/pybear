# Authors:
#
#       Bill Sousa
#
# License: BSD 3 clause



#CALLED BY read_write_file.read_file_config_run,
# standard_configs.LEFT1_TOP1_config.LEFT1_TOP1_file_read_config.LEFT1_TOP1_file_read_config()
def filetype(full_filename):

    if '.' not in full_filename:
        print(f'\n*** FILE NAME MUST HAVE AN EXTENSION ***\n')
        filetype = 'NONE'
    else:
        filetype = full_filename[full_filename.index('.') + 1:].upper()

        if filetype in ['XLS', 'XLSM', 'XLSX']:
            filetype = 'EXCEL'

        elif filetype in ['CSV', 'TXT']:
            pass

        else:
            filetype = 'UNRECOGNIZED'
            print(f'FILE HAS AN UNRECOGNIZED EXTENSION - {filetype}')

    return filetype














