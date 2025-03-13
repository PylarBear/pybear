# Authors:
#
#       Bill Sousa
#
# License: BSD 3 clause



from ....data_validation import validate_user_input as vui


def txt_python_config():
    delimiter = vui.user_entry('Use a delimiter (n) or enter delimiter > ')

    object_type = vui.validate_user_str(
        'Return a python list-of-lists(l), numpy array(n) or dataframe(d) > ',
        'LND'
    )

    if object_type == 'L':
        object_type = 'LIST_OF_LISTS'
    elif object_type == 'N':
        object_type = 'NUMPY_ARRAY'
    elif object_type == 'D':
        object_type = 'DATAFRAME'

    return delimiter, object_type
















