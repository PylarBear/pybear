import numpy as np
from general_text import TextCleaner as tc


# ALSO SEE general_text.TextCleaner FOR SIMILAR CODE, IF EVER CONSOLIDATING

print(f'*** Reading from file ***')
# filename = r'C:\Users\Bill\AppData\Local\Programs\Python\Python39\PyCharmProjects\general_text\notepad.txt'
filename = r'C:\Users\Bill\Desktop\new_notepad.txt'

print(f'*** Building array ***')
raw_text = np.fromiter(open(filename, mode='r', encoding="utf8"), dtype='<U1000')
raw_text = raw_text.reshape((1,-1))[0]

print(f'*** Loading text to TextCleaner ***')
JustifierClass = tc.TextCleaner(raw_text, update_lexicon=False)

# CONVERT TO LIST OF LISTS
print(f'*** Converting to lists of lists ***')
JustifierClass.as_list_of_lists()

TXT = JustifierClass.CLEANED_TEXT

seed = f''
max_line_len = 120
NEW_TXT = np.empty((1,0), dtype=f'<U{max_line_len}')[0]
print(f'*** Justifying... ***')
for row_idx in range(len(TXT)):
    for word_idx in range(len(TXT[row_idx])):
        new_word = TXT[row_idx][word_idx]
        if len(seed) + len(new_word) <= max_line_len:
            seed += new_word + ' '
        elif len(seed) + len(new_word) > max_line_len:
            NEW_TXT = np.insert(NEW_TXT, len(NEW_TXT), seed.strip(), axis=0)
            seed = new_word + ' '
if len(seed) > 0: NEW_TXT = np.insert(NEW_TXT, len(NEW_TXT), seed.strip(), axis=0)

del TXT, seed, max_line_len, new_word

print(f'\n*** Done. ***')

# SET CLASS OBJECT TO THE NEW TXT OBJECT TO USE CLASS METHODS
JustifierClass.CLEANED_TEXT = NEW_TXT; del NEW_TXT
JustifierClass.is_list_of_lists = False
JustifierClass.dump_to_txt()

del JustifierClass

print(f'*** Done ***')