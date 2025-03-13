import numpy as n
from ...data_validation import validate_user_input as vui


def decay_fxn_seq_list_fill(length_number, LIST_TO_FILL):

    while True:

        constant = vui.validate_user_float('Enter starting learning rate > ')
        divisor = vui.validate_user_float('Enter divisor > ', min=1e-30)

        LIST_TO_FILL = n.fromiter((
            constant / divisor ** x for x in range(length_number))
            , dtype=n.float64
        )

        print(f'First [:{min(10, len(LIST_TO_FILL))}] of decay seq look like \n{LIST_TO_FILL[:10]}')

        if vui.validate_user_str('Accept decay sequence list? (y/n) > ', 'YN') == 'Y':
            break

    return LIST_TO_FILL

