# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#







#
# def justify(
#     chars:Optional[Union[numbers.Integral, None]] = None
# ) ->
#     """
#     Fit text as strings or as lists to user-specified number of
#     characters per row.
#
#     Parameters
#     ----------
#     chars:
#         int - number of characters per row
#
#
#     """
#
#     # ALSO SEE text.notepad_justifier FOR SIMILAR CODE, IF EVER CONSOLIDATING
#
#     if not chars is None:
#         arg_kwarg_validater(
#             chars,
#             'characters',
#             list(range(30, 50001)),
#             'TC',
#             'justify'
#         )
#     elif chars is None:
#         # DONT PUT THIS IN akv(return_if_none=)... PROMPTS USER FOR
#         # sINPUT BEFORE ENDING args/kwargs TO akv
#         chars = vui.validate_user_int(
#             f'\nEnter number of characters per line (min=30, max=50000) > ', min=30, max=50000)
#
#     # CONVERT TO LIST OF LISTS
#     converted = False
#     if not self.is_list_of_lists:
#         self.as_list_of_lists()
#         converted = True
#
#     seed = f''
#     max_line_len = chars
#     del chars
#     NEW_TXT = np.empty((1, 0), dtype=f'<U{max_line_len}')[0]
#     for row_idx in range(len(self.CLEANED_TEXT)):
#         for word_idx in range(len(self.CLEANED_TEXT[row_idx])):
#             new_word = self.CLEANED_TEXT[row_idx][word_idx]
#             if len(seed) + len(new_word) <= max_line_len:
#                 seed += new_word + ' '
#             elif len(seed) + len(new_word) > max_line_len:
#                 NEW_TXT = np.insert(NEW_TXT, len(NEW_TXT), seed.strip(), axis=0)
#                 seed = new_word + ' '
#     if len(seed) > 0:
#         NEW_TXT = np.insert(NEW_TXT, len(NEW_TXT), seed.strip(), axis=0)
#
#     del max_line_len, seed, new_word
#
#     self.CLEANED_TEXT = NEW_TXT
#     del NEW_TXT
#     self.is_list_of_lists = False
#
#     # OBJECT WAS WORKED ON AS LIST OF LISTS, BUT OUTPUT IS LIST OF STRS
#     if converted:
#         # MEANING THAT IS WAS list_of_strs TO START WITH, JUST LEAVE AS IS
#         pass
#     elif not converted:
#         # OTHERWISE WAS LIST OF LISTS TO START, SO CONVERT BACK TO LIST OF LISTS
#         self.as_list_of_lists()
#         map(str.strip, self.CLEANED_TEXT)
#     del converted


