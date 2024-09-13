# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

import os
import numpy as np

from pybear.feature_extraction.text._TextCleaner.TextCleaner import TextCleaner


abspath = os.path.abspath(__file__)


# Traverse three directory levels up to reach the desired directory
project_root = os.path.dirname(
                    os.path.dirname(
                        os.path.dirname(
                            os.path.dirname(
                                abspath
                            )
                        )
                    )
)

# Construct the path to the desired file
path = os.path.join(project_root, 'notepad.txt')

# Read the file and create the numpy array
try:
    with open(path, mode='r', encoding="utf8") as file:
        raw_text = np.fromiter(file, dtype='<U1000')
except FileNotFoundError:
    raise FileNotFoundError(f"File not found: {path}")
    # raw_text = np.array([])  # Or handle the error as needed
except Exception as e:
    raise Exception(f"reading notepad file excepted with --- {e}")


raw_text.reshape((1, -1))

Scrubber = TextCleaner(raw_text, update_lexicon=True)
Scrubber.menu()


























