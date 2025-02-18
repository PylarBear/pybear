# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#





class InteractiveTextCleaner:

    def __init__(self):

        self.LEX_LOOK_DICT = {
            'A': 'ADD TO LEXICON', 'D': 'DELETE', 'E': 'EDIT', 'F': 'EDIT ALL',
            'L': 'DELETE ALL', 'S': 'SPLIT', 'U': 'SPLIT ALWAYS', 'K': 'SKIP ONCE',
            'W': 'SKIP ALWAYS', 'Y': 'VIEW LEXICON ADDENDUM', 'Z': 'GO TO MAIN MENU'
        }  # DONT USE 'T' !!!

        if not self.update_lexicon:
            del self.LEX_LOOK_DICT['A']

        self.LEX_LOOK_DISPLAY = ", ".join(list(f'{v}({k.lower()})' for k, v in self.LEX_LOOK_DICT.items()))
        self.lex_look_allowed = "".join(list(self.LEX_LOOK_DICT.keys())).lower()
        self.undo_status = False
        self.CLEANED_TEXT_BACKUP = None

        self.MENU_DICT = {
            'D': {'label': 'delete_empty_rows', 'function': self.delete_empty_rows},
            'R': {'label': 'remove_characters', 'function': self.remove_characters},
            'S': {'label': 'strip', 'function': self._strip},
            'N': {'label': 'normalize', 'function': self.normalize},
            'A': {'label': 'statistics', 'function': self.statistics},
            'K': {'label': 'word_counter', 'function': self.word_counter},
            'V': {'label': 'view_CLEANED_TEXT', 'function': self.view_cleaned_text},
            'U': {'label': 'view_row_uniques', 'function': self.view_row_uniques},
            'O': {'label': 'view_overall_uniques', 'function': self.view_overall_uniques},
            'Y': {'label': 'view_lexicon_addendum', 'function': self.display_lexicon_update},
            'T': {'label': 'remove_stops', 'function': self.remove_stops},
            'J': {'label': 'justify', 'function': self.justify},
            'W': {'label': 'delete_words', 'function': self.delete_words},
            'B': {'label': 'substitute_words', 'function': self.substitute_words},
            'L': {'label': 'as_list_of_lists', 'function': self.as_list_of_lists},
            'I': {'label': 'as_list_of_strs', 'function': self.as_list_of_strs},
            'P': {'label': 'lex_lookup', 'function': self.lex_lookup},
            'C': {'label': 'dump_to_csv', 'function': self.dump_to_csv},
            'X': {'label': 'dump_to_txt', 'function': self.dump_to_txt},
            'E': {'label': f'toggle UNDO (currently {self.undo_status})', 'function': self.toggle_undo},
            'F': {'label': 'undo', 'function': None},
            'Q': {'label': 'quit', 'function': None},
            'Z': {'label': 'accept and exit', 'function': None}
        }

    # END init ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *



    def menu(
            self,
            allowed: str = None,
            disallowed: str = None
    ) -> None:
        # VALIDATION ###################################################
        fxn = inspect.stack()[0][3]

        allowed_key = "".join(self.MENU_DICT.keys()).upper()
        # alloweds = 'ABCDEFIJLNOPQRSTUWXZ'

        if not allowed is None and not disallowed is None:
            self._exception(f'{fxn} >>> CANNOT ENTER BOTH allowed AND disallowed, MUST BE ONE OR THE OTHER OR NEITHER',
                            fxn=fxn)
        elif not allowed is None:
            if not isinstance(allowed, str):
                self._exception(f'{fxn} allowed KWARG REQUIRES str AS INPUT', fxn)
            for _char in allowed:
                if not _char in allowed_key:
                    self._exception(f'INVALID KEY "{_char}" IN allowed, MUST BE IN {allowed_key}.', fxn)
        elif not disallowed is None:
            if not isinstance(disallowed, str):
                self._exception(f'{fxn} disallowed KWARG REQUIRES str AS INPUT', fxn)
            for _char in disallowed:
                if not _char in allowed_key:
                    self._exception(f'INVALID KEY "{_char}" IN disallowed, MUST BE IN {allowed_key}.', fxn)
            allowed = ''.join([_ for _ in allowed_key if not _ in disallowed])
        elif allowed is None and disallowed is None:
            allowed = allowed_key

        allowed = allowed.upper()

        for _ in allowed:
            if _ not in allowed_key:
                self._exception(f'{fxn} allowed KWARG CHARACTERS MUST BE IN *{allowed_key}*', fxn)
        # END VALIDATION ###############################################

        while True:

            # BUILD MENU DISPLAY #######################################
            # MUST BE INSIDE while BECAUSE PRINTOUT IS DYNAMIC BASED ON undo_status AND AVAILABILITY OF BACKUP
            # MANAGE AVAILABILITY OF undo COMMAND BASED ON STATUS OF undo_status AND BACKUP
            if self.undo_status is False or (self.undo_status is True and self.CLEANED_TEXT_BACKUP is None):
                allowed = allowed.replace('F', '')
            elif self.undo_status is True and not self.CLEANED_TEXT_BACKUP is None and 'F' not in allowed:
                allowed += 'F'

            # MUST UPDATE "UNDO TOGGLE" 'label' VALUE EVERY TIME EXPLICITLY,
            # OTHERWISE IT IS NOT UPDATING.  IT APPEARS THAT THE VALUE IS
            # BEING SET THE ONE TIME DURING init, BUT ONCE WRIT THE VALUE
            # IS HARD-CODED AND NO LONGER SEES A {} THERE.
            self.MENU_DICT['E']['label'] = f'toggle UNDO (currently {self.undo_status})'

            display = f""
            for idx, key in enumerate(allowed):
                display += f"{self.MENU_DICT[key]['label']}({key.lower()})".ljust(40)
                if idx % 3 == 2:
                    display += f"\n"
            # END BUILD MENU DISPLAY ###################################
            print()
            print(display)

            selection = vui.validate_user_str(f'\nSelect operation > ', allowed)

            # ONLY CREATE A NEW BACKUP COPY WHEN GOING TO delete_empty_rows,
            # remove_characters, strip, normalize,
            # remove_stops, justify, lex_lookup, delete_words, substitute_words
            if self.undo_status is True and selection in 'DRSNTJPWB':
                self.CLEANED_TEXT_BACKUP = self.CLEANED_TEXT.copy()

            if selection == 'F':
                self.CLEANED_TEXT = self.CLEANED_TEXT_BACKUP.copy()
                self.CLEANED_TEXT_BACKUP = None
                continue  # FORCE BYPASS AROUND BELOW FUNCTION CALL
            elif selection == 'Q':
                self._exception(f'USER TERMINATED', fxn)
            elif selection == 'Z':
                del display
                return self.CLEANED_TEXT  # 4-8-23 ALLOW RETURN OF CLEANED_TEXT FROM menu()

            self.MENU_DICT[selection]['function']()

            print()


    def toggle_undo(self):
        """Turn undo capability ON or OFF to save memory (only used in menu())."""
        if self.undo_status is False:  # CONVERTING OVER TO True
            self.undo_status = True
        elif self.undo_status is True:  # CONVERTING OVER TO False
            self.CLEANED_TEXT_BACKUP = None
            self.undo_status = False


    def as_list_of_lists(self):
        """Convert CLEANED_TEXT object to a possibly ragged vector of vectors, each vector containing split text."""
        if self.is_list_of_lists:
            pass
        elif not self.is_list_of_lists:  # MUST BE LIST OF strs
            # ASSUME THE TEXT STRING CAN BE SEPARATED ON ' '
            self.CLEANED_TEXT = np.fromiter(map(str.split, self.CLEANED_TEXT), dtype=object)
            for row_idx in range(len(self.CLEANED_TEXT)):
                # WONT LET ME DO np.fromiter(map(np.ndarray SO PLUG-N-CHUG
                self.CLEANED_TEXT[row_idx] = np.array(self.CLEANED_TEXT[row_idx], dtype='<U30')

            self.is_list_of_lists = True



    def as_list_of_strs(self):
        """Convert CLEANED_TEXT object to a single vector of strings."""
        if not self.is_list_of_lists:
            pass
        elif self.is_list_of_lists:
            self.CLEANED_TEXT = np.fromiter(map(' '.join, self.CLEANED_TEXT), dtype=object)
            self.is_list_of_lists = False


    def lex_lookup_menu(
            self,
            allowed: Union[str, None] = None,
            disallowed: Union[str, None] = None,
            fxn: Union[str, None] = None
    ):  # ALLOWED ARE 'adelks'

        """Dynamic function for returning variable menu prompts and allowed commands."""

        # pizza only used in lex_lookup

        fxn = fxn or inspect.stack()[0][3]

        if not allowed is None and not disallowed is None:
            ValueError(f'CANNOT ENTER BOTH allowed AND disallowed, ONLY ONE OR THE OTHER OR NEITHER', fxn=fxn)

        elif allowed is None and disallowed is None:
            allowed = self.lex_look_allowed

        elif not allowed is None and disallowed is None:
            # VALIDATE ENTRY FOR allowed kwarg #########################
            if not isinstance(allowed, str):
                TypeError(fxn, f'{inspect.stack()[0][3]}() allowed kwarg must be a single string')

            for _ in allowed:
                if _.upper() not in ans.alphabet_str_upper():
                    ValueError(fxn,
                                    f'lexicon_menu() kwarg allowed CAN ONLY CONTAIN {self.lex_look_allowed.lower()}')
            # END VALIDATE ENTRY FOR allowed kwarg #####################

        elif allowed is None and not disallowed is None:
            # VALIDATE ENTRY FOR disallowed kwarg ######################
            if not isinstance(disallowed, str):
                TypeError(f'disallowed kwarg must be a single string')

            for _ in disallowed:
                if _.upper() not in ans.alphabet_str_upper():
                    ValueError(f'lexicon_menu() kwarg disallowed CAN ONLY CONTAIN {self.lex_look_allowed.lower()}')
            # END VALIDATE ENTRY FOR disallowed kwarg ##################

            allowed = ''.join([_ for _ in self.lex_look_allowed.lower() if _ not in disallowed])

        WIP_DISPLAY = ", ".join(
            [f'{v.upper()}({k.lower()})' for k, v in self.LEX_LOOK_DICT.items() if k.lower() in allowed.lower()])
        wip_allowed = allowed

        return WIP_DISPLAY, wip_allowed


