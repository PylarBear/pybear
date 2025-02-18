# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#




class FileDumpMixin:



    def dump_to_file_wrapper(self, core_write_function, _ext, kwargs):
        """
        Wrapper function for dumping CLEANED_TEXT object to csv or txt
        """

        # pizza used in word_counter, dump_to_csv, dump_to_txt

        converted = False
        if self.is_list_of_lists:
            self.as_list_of_strs()
            converted = True

        while True:
            file_name = input(f'Enter filename > ')
            __ = vui.validate_user_str(f'User entered *{file_name}*  ---  Accept? (y) (n) (a)bort > ', 'YNA')
            if __ == 'Y':
                core_write_function(file_name + _ext, **kwargs)
                print(f'\n*** Dump to {_ext} successful. ***\n')
                break
            elif __ == 'N':
                continue
            elif __ == 'A':
                break

        if converted: self.as_list_of_lists()
        del converted


    def dump_to_csv(self):
        """Dump CLEANED_TEXT object to csv."""

        # pizza only used in lex_lookup

        print(f'\nSaving CLEANED TEXT to csv...')

        converted = False
        if self.is_list_of_lists:
            self.as_list_of_strs()
            converted = True
        _core_fxn = pd.DataFrame(data=self.CLEANED_TEXT.transpose(), columns=[f'CLEANED_DATA']).to_csv

        self.dump_to_file_wrapper(_core_fxn, f'.csv', {'header': True, 'index': False})

        if converted:
            self.as_list_of_lists()
        del converted


    def dump_to_txt(self):
        """Dump CLEANED_TEXT object to txt."""

        # pizza only used in lex_lookup

        print(f'\nSaving CLEANED TEXT to txt file...')

        def _core_fxn(full_path):  # DONT PUT kwargs OR **kwargs IN ()!
            with open(full_path, 'w') as f:
                for line in self.CLEANED_TEXT:
                    f.write(line + '\n')
                f.close()

        self.dump_to_file_wrapper(_core_fxn, f'.txt', {})


