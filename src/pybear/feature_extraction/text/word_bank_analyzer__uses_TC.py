# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#




import numpy as np
from ..text import TextCleaner as tc



def wordbank_analyzer():

    filename = r'C:\Users\Bill\AppData\Local\Programs\Python\Python39\PyCharmProjects\general_text\word_bank'

    raw_text = np.fromiter(open(filename, mode='r', encoding="utf8"), dtype='<U1000')
    raw_text.reshape((1,-1))
    print(f'RAW TEXT (dtype={type(raw_text)}):')
    print(raw_text)

    Scrubber = tc.TextCleaner(raw_text, auto_delete=False, update_lexicon=True)
    Scrubber.menu()











