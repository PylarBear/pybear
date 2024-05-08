import sys, inspect, warnings
from general_sound import winlinsound as wls
import numpy as np
from debug import time_memory_tester_2 as tmt2, time_memory_tester_3 as tmt3
from general_text import alphanumeric_str as ans


# TEST FOR TextCleaner.remove_characters()

''' ITERATE OVER np.uniques GOTTEN FROM np.fromiter OVER CHARS IS HANDS DOWN WINNER
np_unique1                                        average, sdev: time = 0.695 sec, 0.078; mem = 0.000, 0.000
np_unique2                                        average, sdev: time = 2.586 sec, 0.140; mem = 0.000, 0.000
plug_n_chug                                       average, sdev: time = 2.443 sec, 0.206; mem = 0.000, 0.000
'''


# THIS MODULE TESTS SPEED OF TWO DIFFERENT METHODS TO CLEAN CHARS OUT OF A LIST OF STRINGS [str1, str2,... ]
# METHODS 1&2: GET UNIQUES FROM EACH STR AND DO np.char.replace OVER UNIQUES TO REMOVE DISALLOWED CHARS FROM EACH STR
# METHOD 3: PLUG-N-CHUG VIA for LOOP OVER EACH STR AND BUILD A SUBSTITUTE STR W ONLY ALLOWED CHARS

def np_unique1(LIST_OF_STRS, allowed_chars_as_str=ans.alphanumeric_str(), disallowed_chars_as_str=None):

    for row_idx in range(len(LIST_OF_STRS)):
        UNIQUES = "".join(np.unique(np.fromiter((_ for _ in str(LIST_OF_STRS[row_idx])), dtype='<U1')))
        for char in UNIQUES:
            if (not allowed_chars_as_str is None and char not in allowed_chars_as_str) or \
                (not disallowed_chars_as_str is None and char in disallowed_chars_as_str):
                LIST_OF_STRS[row_idx] = np.char.replace(LIST_OF_STRS[row_idx], char, '')
    del UNIQUES
    
    return LIST_OF_STRS


def np_unique2(LIST_OF_STRS, allowed_chars_as_str=ans.alphanumeric_str(), disallowed_chars_as_str=None):
    for row_idx in range(len(LIST_OF_STRS)):
        UNIQUES = "".join(np.unique(np.hstack(list(map(list, str(LIST_OF_STRS[row_idx]))))))
        for char in UNIQUES:
            if (not allowed_chars_as_str is None and char not in allowed_chars_as_str) or \
                    (not disallowed_chars_as_str is None and char in disallowed_chars_as_str):
                LIST_OF_STRS[row_idx] = np.char.replace(LIST_OF_STRS[row_idx], char, '')
    del UNIQUES

    return LIST_OF_STRS



def plug_n_chug(LIST_OF_STRS, allowed_chars_as_str=ans.alphanumeric_str(), disallowed_chars_as_str=None):

    for row_idx in range(len(LIST_OF_STRS)):
        holder_str = f''
        for char_idx in range(len(str(LIST_OF_STRS[row_idx]))):
            __ = str(LIST_OF_STRS[row_idx])[char_idx]
            if not allowed_chars_as_str is None and __ in allowed_chars_as_str: holder_str += __
            elif not disallowed_chars_as_str is None and __ not in disallowed_chars_as_str: holder_str += __
        LIST_OF_STRS[row_idx] = holder_str
    del holder_str, __

    return LIST_OF_STRS








if __name__ == '__main__':

    if 1==2:
        # MODULE FOR TESTING ACCURACY OF FUNCTIONS

        DUM_TXT = [f'''python - List of all unique characters in a string? - Stack Overflowhttps://stackoverflow.com › questions''',
                f'''› list-of-all-uniqu... Apr 25, 2017 — Now of course I have two solutions in my mind. One is using a list''',
                f'''that will map the characters with their ASCII codes. So whenever I encounter a letter it will ... 9 answers.''',
                f'''Top answer: The simplest solution is probably: In [10]: ''.join(set('aaabcabccd')) Out[10]: 'a']''']

        TEST1 = np_unique1(DUM_TXT)
        TEST2 = np_unique2(DUM_TXT)
        TEST3 = plug_n_chug(DUM_TXT)

        if np.array_equiv(TEST1, TEST2) and np.array_equiv(TEST1, TEST3):
            print('\033[92mACCURACY TEST PASSED.\033[0m')
        else:
            _print = lambda x: print(f'\033[91m{x}\033[0m')
            _print(f'ACCURACY TEST FAILED')
            _print(f'np_unique1')
            [print(_) for _ in TEST1]
            print()
            _print(f'np_unique1')
            [print(_) for _ in TEST2]
            print()
            _print(f'plug_n_chug')
            [print(_) for _ in TEST3]


    if 1==1:

        DUM_TXT = [
                    '''Contrary to popular belief, Lorem Ipsum is not simply _random_ text.''',
                    '''It has roots in a piece of classical Latin literature from 45 BC, making it over 2000 years old.''',
                    '''Richard McClintock, a Latin professor at Hampden-Sydney College in Virginia, looked up one of the 
                        more obscure Latin words, consectetur, from a Lorem Ipsum passage, and going through the cites of
                        the word in classical literature, discovered the undoubtable source.''',
                    '''Lorem Ipsum comes from sections 1.10.32 and 1.10.33 of "de Finibus Bonorum et Malorum" (The Extremes 
                        of Good and Evil) by Cicero, written in 45 BC. This book is a treatise on the theory of ethics, very 
                        popular during the Renaissance.''',
                    '''The first line of Lorem Ipsum, "Lorem ipsum dolor sit amet..", comes from a line in section 1.10.32.'''
                    ]

        DUM_TXT = np.hstack((np.fromiter((DUM_TXT for _ in range(1000)), dtype=object)))

        function1 = np_unique1
        function2 = np_unique2
        function3 = plug_n_chug
        tmt3.time_memory_tester_3(function1,
                                 function2,
                                 function3,
                                 [DUM_TXT],
                                 [DUM_TXT],
                                 [DUM_TXT],
                                 FUNCTION1_KWARGS_IN_DICT={},
                                 FUNCTION2_KWARGS_IN_DICT={},
                                 FUNCTION3_KWARGS_IN_DICT={},
                                 function1_name='np_unique1',
                                 function2_name='np_unique2',
                                 function3_name='plug_n_chug',
                                 number_of_trials=30,
                                 rest_time=1)


        wls.winlinsound(444, 500)


