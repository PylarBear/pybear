from general_text import TextCleaner as tc
import numpy as np


TEST_TEXT = ['AND FRODO WOULD NOT GIVE THE RING TO HIM',
             'THE PIZZA WENT OVER THE MOUNTAIN TO SEE WHAT HE COULD SEE',
             'AND YOUR LITTLE DOG TOO AND',
             'AND AND AND AND AND']

TestClass = tc.TextCleaner(TEST_TEXT)

TestClass.remove_stops()

[print(f'***{_}***') for _ in TestClass.CLEANED_TEXT]







