# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


def as_list_of_strs(self):
    """
    Convert CLEANED_TEXT object to a single vector of strings.




    """



    if not self.is_list_of_lists:
        pass
    elif self.is_list_of_lists:
        self.CLEANED_TEXT = np.fromiter(map(' '.join, self.CLEANED_TEXT),
                                        dtype=object)
        self.is_list_of_lists = False