# Authors:
#
#       Bill Sousa
#
# License: BSD 3 clause




import datetime as dt



def validate_user_str(user_prompt:str, options:str) -> str:

    """Validation of a single user-entered alpha character against a list of
    allowed characters.

    Parameters
    ----------
    user_prompt:
        str -  text string displayed to the user at prompt
    options:
        str - a single text string containing the allowed characters

    Returns
    -------
    user_input:
        str - validated user selection

    """

    if not isinstance(user_prompt, str):
        raise TypeError(f"'user_prompt' must be a string")

    if not isinstance(options, str):
        raise TypeError(f"'options' must be a string")


    while True:
        user_input = input(user_prompt).upper()
        if len(user_input) == 1 and user_input in options.upper() and \
                user_input != '' and user_input != options:
            break

    return user_input


def validate_user_mstr(user_prompt:str, options:str, max_len:int=2) -> str:

    """String _validation for multiple alpha character user entry that screens
    by len of entry and allowed options.

    Parameters
    ----------
    user_prompt:
        str -  text string displayed to the user at prompt
    options:
        str - a single text string containing the allowed characters
    max_len:
        int, default = 2 - maximum number of allowed selections

    Returns
    -------
    user_input:
        str - validated user selection(s)

    """

    if not isinstance(user_prompt, str):
        raise TypeError(f"'user_prompt' must be a string")

    if not isinstance(options, str):
        raise TypeError(f"'options' must be a string")

    if 'INT' not in str(type(max_len)).upper():
        raise TypeError(f"'max_len' must be an integer")

    if max_len < 1:
        raise ValueError(f"'max_len' must be >= 1")


    while True:
        user_input = input(user_prompt).upper()
        if len(user_input) <= max_len and user_input != '':
            for char in user_input:
                if char.upper() not in options.upper():
                    continue
            else:
                break
    return user_input


def validate_user_int(
                      user_prompt:str,
                      min:int=float('-inf'),
                      max:int=float('inf')
    ) -> int:

    """Integer _validation for user entry within allowed range.

    Parameters
    ----------
    user_prompt:
        str -  text string displayed to the user at prompt
    min:
        [int,float], default = float('-inf') - minimum allowed entry
    max:
        [int,float], default = float('inf') - maximum allowed entry

    Returns
    -------
    user_input -
        user_input: int - validated user entry

    """

    if not isinstance(user_prompt, str):
        raise TypeError(f"'user_prompt' must be a string")

    try:
        float(min)
    except:
        raise TypeError(f"'min' must be a number")

    try:
        float(max)
    except:
        raise TypeError(f"'max' must be a number")


    err_msg = f'Must enter an integer between {min} and {max}'
    while True:
        try:
            user_input = float(input(user_prompt))
            if int(user_input) != user_input:
                raise Exception
            else:
                user_input = int(user_input)
            if user_input >= min and user_input <= max: break
            else: print(err_msg)
        except:
            print(err_msg)
    del err_msg
    return user_input


def validate_user_float(
                        user_prompt:str,
                        min:float=float('-inf'),
                        max:float=float('inf')
    ) -> float:

    """Number _validation for user float entry within allowed range.

    Parameters
    ----------
    user_prompt:
        str -  text string displayed to the user at prompt
    min:
        float, default = float('-inf') - minimum allowed entry
    max:
        float, default = float('inf') - maximum allowed entry

    Returns
    -------
    user_input -
        user_input: str - validated user entry

    """

    if not isinstance(user_prompt, str):
        raise TypeError(f"'user_prompt' must be a string")

    try:
        float(min)
    except:
        raise TypeError(f"'min' must be a number")

    try:
        float(max)
    except:
        raise TypeError(f"'max' must be a number")

    err_msg = f'Must enter an float between {min} and {max}'
    while True:
        try:
            user_input = float(input(user_prompt))
            if user_input >= min and user_input <= max: break
            else: print(err_msg)
        except:
            print(err_msg)
    del err_msg
    return user_input


def user_entry(prompt: str):

    """String _validation for user-entered string.

    Parameters
    ----------
    user_prompt:
        str -  text string displayed to the user at prompt

    Return
    ------
    user_entry -
        user_entry: str - validated user entry

    """

    while True:
        user_entry = input(f'{prompt} > ')
        if validate_user_str(f'\nUser entered "{user_entry}"... '
                             f'accept? (y/n) > ', 'YN') == 'Y':
            break

    return user_entry


class ValidateUserDate:

    """ Prompt the user for a date entry and optionally validate the input.

    Parameters
    ----------
    user_prompt:
        str - text string displayed to the user at prompt
    user_verify:
        bool - default = False, perform _validation on the entry
    format:
        str - default = 'MM/DD/YYYY', the date format
    min:
        str - default = '01/01/1900', the earliest allowed date
    max:
        str - default = '12/31/2099', the latest allowed date

    Attributes
    ----------
    user_prompt:
        str - text string displayed to the user at prompt
    user_verify:
        bool - default = False, perform _validation on the entry
    format:
        str - the date format
    min:
        str - the earliest allowed date
    max:
        str - the latest allowed date
    user_input:
        str - raw date entered by user
    datetime_input_date:
        date - formatted date
    datetime_min_date:
        str - the earliest allowed date
    datetime_max_date:
        str - the latest allowed date
    startYYYY:
        int - deprecated.
    startMM:
        int - deprecated.
    startDD:
        int - deprecated.
    endYYYY:
        int - deprecated.
    endMM:
        int - deprecated.
    endDD:
        int - deprecated.

    Examples
    --------

    """

    def __init__(self,
                 user_prompt: str,
                 user_verify: bool = False,
                 format: str ='MM/DD/YYYY',
                 min: [str, dt.datetime] ='01/01/1900',
                 max: [str, dt.datetime] ='12/31/2099'
        ):

        if not isinstance(user_prompt, str):
            raise TypeError(f"'user_prompt' must be a string")

        if not isinstance(user_verify, bool):
            raise TypeError(f"'user_verify' must be a boolean")

        if not isinstance(format, str):
            raise TypeError(f"'format' must be a string")

        if not isinstance(min, (str, dt.date)):
            raise TypeError(f"'min' must be a string or datetime.date")

        if not isinstance(max, (str, dt.date)):
            raise TypeError(f"'max' must be a string or datetime.date")


        self.user_prompt = user_prompt
        self.format = format
        self.user_verify = user_verify
        self.min = min
        self.max = max

        while True:
            print('')
            self.user_input = str(input(f'{self.user_prompt} ({self.format}) > '))
            print('')

            if len(self.user_input) != len(self.format):
                print(f'\nCheck format, incorrect length.\n')
                continue

            for idx in range(len(self.format)):
                if self.format[idx].isalpha() != self.user_input[idx].isnumeric():
                    print(f'\nCheck format, incorrect number location(s).\n')
                    continue

            self.datetime_input_date = dt.date(
                                            self._determine_YYYY(self.user_input),
                                            self._determine_MM(self.user_input),
                                            self._determine_DD(self.user_input)
            )
            self.datetime_min_date = dt.date(
                                            self._determine_YYYY(self.min),
                                            self._determine_MM(self.min),
                                            self._determine_DD(self.min)
            )
            self.datetime_max_date = dt.date(
                                             self._determine_YYYY(self.max),
                                             self._determine_MM(self.max),
                                             self._determine_DD(self.max)
            )

            if self.datetime_input_date >= self.datetime_min_date and \
                    self.datetime_input_date <= self.datetime_max_date:
                pass
            else:
                print(f'\nInput date must be between {self.min} and {self.max}.\n')
                continue

            if self.user_verify:
                if validate_user_str(f'User entered {self.user_input}... '
                                     f'Accept? (y/n)? > ', 'YN') == 'Y':
                    break
            else:
                break

            break


    def _determine_YMD_template(
                                self,
                                date_object,
                                search_char='Y',
                                datetime_look_range=range(0,4)
    ):

        __ = ''
        if 'STR' in str(type(date_object)).upper():
            for idx in range(len(date_object)):
                if self.format[idx].upper() == search_char:
                    __ += date_object[idx]
        elif 'DATETIME' in str(type(date_object)).upper():
            for idx in datetime_look_range:
                __ += str(date_object)[idx]
        return int(__)


    def _determine_YYYY(self, date_object):
        __ = str(self._determine_YMD_template(date_object))
        if len(__) == 2: __ = '20' + __
        return int(__)


    def _determine_MM(self, date_object):
        return self._determine_YMD_template(
                                            date_object,
                                            search_char='M',
                                            datetime_look_range=range(5,7)
        )


    def _determine_DD(self, date_object):
        return self._determine_YMD_template(
                                            date_object,
                                            search_char='D',
                                            datetime_look_range=range(8,10)
        )


    def return_user_format(self):
        """Return date as entered by user.

        Parameters
        --------
        None


        Returns
        ------
        user_input: str - raw date entered by user

        """

        return self.user_input


    def return_datetime(self):
        """Return formatted date.

        Parameters
        --------
        None


        Returns
        ------
        datetime_input_date: str - formatted date

        """


        return self.datetime_input_date


    def _return_date_old_way(self):
        """Deprecated. Keep for reference."""
        while True:
            print('')
            self.startYYYY = validate_user_int(f'start YYYY > ',
                                min=1900, max=dt.datetime.now().year)

            if self.startYYYY == dt.datetime.now().year:
                self.startMM = validate_user_int(f'start MM > ',
                                 min=1, max=dt.datetime.now().month)
            else:
                self.startMM = validate_user_int(f'start MM > ', min=1, max=12)

            if self.startYYYY == dt.datetime.now().year and \
                    self.startMM == dt.datetime.now().month:
                self.startDD = validate_user_int(f'start DD > ',
                                 min=1, max=dt.datetime.now().day)
            else:
                self.startDD = validate_user_int(f'start DD > ', min=1, max=31)

            self.endYYYY = validate_user_int(f'end YYYY > ',
                             min=self.startYYYY, max=dt.datetime.now().year)

            if self.endYYYY == self.startYYYY:
                self.endMM = validate_user_int(f'end MM > ',
                                               min=self.startMM, max=12)
                if self.endMM == self.startMM:
                    self.endDD = validate_user_int(f'end DD > ',
                                               min=self.startDD, max=31)
                else:
                    self.endDD = validate_user_int(f'end DD > ', min=1, max=31)
            else:
                self.endMM = validate_user_int(f'end MM > ', min=1, max=12)
                self.endDD = validate_user_int(f'end DD > ', min=1, max=31)

            if self.user_verify:
                if validate_user_str(f'User entered {self.user_input}... '
                                     f'Accept? (y/n)? > ', 'YN') == 'Y':
                    break

            return (dt.date(self.startYYYY, self.startMM, self.startDD),
                    dt.date(self.endYYYY, self.endMM, self.endDD))




















