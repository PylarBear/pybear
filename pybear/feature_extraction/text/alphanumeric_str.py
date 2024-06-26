# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#





def alphabet_str_lower():
    return ' abcdefghijklmnopqrstuvwxyz'


def alphabet_str_upper():
    return alphabet_str_lower().upper()


def alphabet_str():
    return alphabet_str_lower() + alphabet_str_upper()


def number_str():
    return '0123456789'


def alphanumeric_str():
    return alphabet_str() + number_str()





