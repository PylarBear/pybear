# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import numpy as np
# pizza
# from feature_extraction.text._TextCleaner.TextCleaner import TextCleaner as tc



def variable_finder():

    filename = (r'C:\Users\Bill\AppData\Local\Programs\Python\Python39\PyCharmProjects\
                text\_variable_finder\variable_finder_code_sample')

    raw_text = np.fromiter(open(filename, mode='r', encoding="utf8"), dtype='<U1000')
    raw_text = raw_text.reshape((1,-1))[0]

    for line_idx in range(len(raw_text)):
        for idx, char in enumerate(raw_text[line_idx]):
            hit_len = False
            if char == '#': raw_text[line_idx] = raw_text[line_idx][:idx]; break

        for thing in ['print(', 'int', 'float', 'str', 'bool', 'pass', 'len',
                      'for ', 'while ', 'in ', '{', '}', '(', ')', '[', ']',
                      ',', ':']:
            raw_text[line_idx] = raw_text[line_idx].replace(thing, ' ')



    Scrubber = tc.TextCleaner(raw_text, update_lexicon=True)
    Scrubber.menu()







































