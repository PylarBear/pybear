
import pytest

pytest.skip(reason=f'24_09_07_11_44_00 need rewrite', allow_module_level=True)


import numpy as np
from general_text import alphanumeric_str as ans



def test_header(_len):

    __ = ans.alphabet_str_upper().replace(' ', '')
    n = len(__)

    # FIND HOW MANY CHARS ARE NEEDED TO FULFILL THE LEN
    chars = 0
    for _exp in range(6):
        if _len in range(n**_exp): chars = _exp; break

    return np.fromiter((f'{__[(idx//n**5)%n]}{__[(idx//n**4)%n]}{__[(idx//n**3)%n]}{__[(idx//n**2)%n]}{__[(idx//n)%n]}{__[idx%n]}'[-chars:]
                       for idx in range(_len)), dtype=f'<U{chars}').reshape((1,-1)).astype('<U10000')
    




if __name__ == '__main__':
    print(test_header(1000))





