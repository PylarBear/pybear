# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

from typing import Optional



# DEFINE FUNCTION FOR MANUALLY ENTERING NUMBERS INTO LISTS DURING RUN-TIME


def manual_num_seq_list_fill(
    name: str,
    VAR_LIST,
    size,
    min:Optional[float]=float('-inf'),
    max:Optional[float]=float('inf')
):

    original_len = len(VAR_LIST)
    __ = size - original_len

    iteration = 0
    while iteration < __:
        if iteration > 0:
            # IF USER ENTERED THIS ON LAST GO-AROUND, LOOP UNTIL LIST IS FULL
            if var == 'F':
                VAR_LIST.append(VAR_LIST[-1])
                iteration += 1
                continue
            option_text = f'Fill remainder with last(f), end(e), or e'
        else: option_text = f'E'

        while True:
            try:
                var = input(f'{option_text}nter {name} as float or integer for position {iteration+1+original_len} (of {size}) > ').upper()
                if float(var) >= min and float(var) <= max:
                    VAR_LIST.append(float(var))
                    break
                else: print(f'Must be > {min} and < {max} or')

            except:
                if var in 'EF':
                    break
                else:
                    print(f'Must be > {min} and < {max}, '
                        f'"f" after first iteration, or "e"'
                )

        iteration += 1
        if var == 'F': iteration -= 1
        if var == 'E': break

    return VAR_LIST



if __name__ == '__main__':
    name = 'TATER'
    VAR_LIST = []
    size = 5

    manual_num_seq_list_fill(name, VAR_LIST, size, min=float('-inf'), max=float('inf'))

    print(VAR_LIST)