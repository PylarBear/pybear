####################################################################################################
# CREATE A FUNCTION TO HANDLE VALIDATION OF CHAR_SEQs###############################################

# USING INVASIVE MEANS TO FIND DATA TYPE NOT USING 'type' BECAUSE AN EXPERIMENT W PANDA DF SHOWED THAT
# WHEN UNSPECIFIED, DF IS ASSIGNING 'STR' TO 'INT', BUT CAN'T JUST GO SPECIFYING 'dtype=int OR float'
# ON A DATA PULL INTO DF BECAUSE ASSIGNING EITHER TO ALPHABET CHARS WILL CRASH (VERIFIED TRUE)

# DETERMINES THE DATA TYPE OF AN ALPHA-NUMERIC-SYMBOLIC SEQUENCE, LIKE THE ITERABLES IN A LIST & A STR, FLT, OR INT

# CURRENCY TYPE IS A COP OUT
#CALLED BY validate_data_type
class ValidateCharSeqDataType:
    def __init__(self, char_seq):#, str_ind, num_ind, num_or_str_ind, time_ind, date_ind):
        # 6-13-22 CODE IS NOT RECOGNIZING NEGATIVE NUMBERS.  ACCOMMODATE THAT -1 IS NOT A BIN,
        # AND THAT OPERATIONS WITH BOOL CONVERT THEM TO 0 OR 1
        if isinstance(char_seq, bool): self.char_seq = str(char_seq)
        elif char_seq == -1: self.char_seq = str(-1)
        else:
            try: self.char_seq = str(abs(char_seq))
            except: self.char_seq = str(char_seq)

        # self.all_alphabetic_chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
        self.allmonthsstr = 'januaryfebruarymarchaprilmayjunejulyaugustsepetemberoctobernovemberdecember'
        self.alphanumeric_list = 'abcdefghijklmnopqrstuvwxyz1234567890'
        self.str_list = 'abcdfghijklmnopqrstuvwxyz!@#^&*_=+[{]}\|;:"<>/? '   # e NOT HERE, INT self.num_or_str_list
        self.num_list = '1234567890'
        self.num_or_str_list = 'e$.()%,-'
        self.time_list = ':ampm '  # NUMBERS COME FROM num_list
        self.date_list = '-/,abcdefghijlmnoprstuvy '  # NUMBERS COME FROM num_list

        self.object_list = ('str_ind', 'num_ind', 'num_or_str_ind', 'time_ind', 'date_ind')
        self.BUCKET_LOOKUP = dict((zip(self.object_list, tuple('' for _ in self.object_list))))
        self.LIST_LOOKUP = {'str_ind': self.str_list + self.str_list.upper(),
                            'num_ind': self.num_list,
                            'num_or_str_ind': self.num_or_str_list,
                            'time_ind': self.time_list + self.time_list.upper(),
                            'date_ind': self.date_list + self.date_list.upper()}

        self.TYPE_LIST = ''
        self.int_conv_err_cntr = 0


    def python_type(self):
        return str(type(self.char_seq))


    def python_modified_type(self):
        _ = str(type(self.char_seq)).upper()
        # THE MASTER DICTIONARY OF TYPES IS IN data_validation.validate_modified_object_type

        # ITERATE THRU THE DICT TO FIND IF THE DICT ENTRY MATCHES A SEQ IN THE _ STRING
        from ML_PACKAGE._data_validation import validate_modified_object_type as vmot
        for entry in vmot.OBJ_TYPES():
            if entry.upper() in _:
                __ = vmot.OBJ_TYPES()[entry]
                break
        else:
            __ = 'UNRECOGNIZED OBJECT TYPE'

        return __

    def done(self, __):  # _ = DTYPE MASTER LIST (ALSO __ IN run())
        if len(__) == 1 and __[0] not in ['NUM/STR', 'NUM']:
            return True
        elif len(__) == 0:
            print(f'\nTHE CharSeqDataType.type() ALGORITHM HAS ELIMINATED ALL TYPES FROM CONSIDERATION.')
            print(f'THERE IS SOMETHING WRONG WITH THE LOGIC.\n')
            return True
        else:
            return False

    def try_pop_except_pass(self, __, type_to_pop):
        try:
            return __.remove(type_to_pop)
        except:
            pass

    def type(self):
        # USING A DICTIONARY, FOR ALL 5 BUCKETS, ASSIGN Y/N IF CHAR IN CHAR_SEQ IS IN THAT BUCKET
        for _ in self.char_seq:
            for __ in self.object_list:
                if _ in self.LIST_LOOKUP[__]:
                    self.BUCKET_LOOKUP[__] += 'Y'
                else:
                    self.BUCKET_LOOKUP[__] += 'N'

        # EXTRACT THE ASSIGNMENT RESULTS THAT ARE HELD IN THE DICTIONARY
        self.str_ind = self.BUCKET_LOOKUP['str_ind']
        self.num_ind = self.BUCKET_LOOKUP['num_ind']
        self.num_or_str_ind = self.BUCKET_LOOKUP['num_or_str_ind']
        self.time_ind = self.BUCKET_LOOKUP['time_ind']
        self.date_ind = self.BUCKET_LOOKUP['date_ind']

        # CREATE ITERABLES TO USE IN THE FOLLOWING ALGORITHM TO NARROW DOWN WHICH TYPE
        _ = [self.str_ind, self.num_ind, self.num_or_str_ind, self.time_ind, self.date_ind]
        __ = ['STR', 'NUM', 'NUM/STR', 'TIME', 'DATE']

        # DROP THINGS OUT OF OR KEEP THINGS IN __ BASED RULES IN THE ALGORITHM
        while True:

            # THINGS THAT MAKE IT AUTOMATICALLY A DATE:
            if 'DATESTAMP' in self.char_seq.upper():
                __ = ['DATE']
                break

            # THINGS THAT MAKE IT AUTOMATICALLY A BOOL:
            if self.char_seq.upper() in ['TRUE', 'FALSE']:
                __ = ['BOOL']
                break

            # IF ANY OF THE BUCKETS ARE ALL "N", POP THEM FIRST THING
            for idx in range(len(_)-1, -1, -1):
                if 'Y' not in _[idx]:
                    __.pop(idx)

            # IF LEN IS ZERO AT THIS POINT, THEN STRING (ACTUALLY AN EMPTY STRING)
            if len(__) == 0:
                __ = ['STR']
                break

            # CREATE A THING THAT WONT ALLOW 'STR' TO BE PUT BACK IN BY SUBSEQUENT ALGORITHM IF IT
            # WAS TAKEN OUT AT THE START, THOUGH THIS IS OVERRIDDEN IN SOME PLACES BELOW
            self.never_str = 'N'
            if 'STR' not in __: self.never_str = 'Y'

            if self.done(__): break  # IF TYPE LIST IS LEN 1, DONE, IF LEN 0, CRASH & BURN

            # IF ANY ALPHABETIC CHARS (EXCEPT THOSE ALLOWED BY num_or_str_list), CANNOT BE NUM
            # AND CREATE A THING THAT WON'T ALLOW 'NUM' TO BE PUT BACK IN BY SUBSEQUENT ALGORITHM
            self.never_num = 'N'
            if 'Y' in self.str_ind :
                self.try_pop_except_pass(__, 'NUM')
                self.never_num = 'Y'

            if self.done(__): break

            # IF NO NUMBERS, THEN CAN'T BE A DATE OR TIME, NO MATTER WHAT THE FORMAT OF DATE OR TIME
            if 'Y' not in self.num_ind:
                self.try_pop_except_pass(__, 'DATE')
                self.try_pop_except_pass(__, 'TIME')

            if self.done(__): break

            # IF ANY ALPHABETIC CHARS THAT AREN'T ALLOWED BY DATE OR TIME ARE PRESENT, POP DATE OR TIME
            if 'STR' in __ and 'DATE' in __:
                for char in self.char_seq.lower():
                    if char in self.str_list and char not in self.date_list:
                        __.remove('DATE')
                        break

            if self.done(__): break

            if 'STR' in __ and 'TIME' in __:
                for char in self.char_seq.lower():
                    if char in self.str_list and char not in self.time_list:
                        __.remove('TIME')
                        break

            if self.done(__): break

            # REASONS WHY SOMETHING CANT BE A DATE OR A TIME
            if 'DATE' in __ or 'TIME' in __:
                if self.char_seq[0].lower() not in self.alphanumeric_list or \
                        self.char_seq[-1].lower() not in self.alphanumeric_list or \
                        self.char_seq.count('-') > 2 or \
                        self.char_seq.count('/') > 2 or \
                        self.char_seq.count(':') > 2:
                    self.try_pop_except_pass(__, 'DATE')
                    self.try_pop_except_pass(__, 'TIME')
                    if 'STR' not in __ and self.never_str == 'N':
                        __.append('STR')

                num_count = sum([1 for x in self.char_seq if x in self.num_list])  #COUNT NUMBERS FOR DATE/TIME TEST
                if num_count not in range(3,9):  # MORE THAN 8 NUMS IN char_seq OR LESS THAN 3
                    self.try_pop_except_pass(__, 'DATE')
                    self.try_pop_except_pass(__, 'TIME')
                    if 'STR' not in __:  # OVERRIDE 'NEVER-STR' DECLARATION
                        __.append('STR')

                if self.char_seq.count(',') > 1:   # MORE THAN 1 COMMA, CANT BE A DATE
                    self.try_pop_except_pass(__, 'DATE')
                    if 'STR' not in __ and self.never_str == 'N':
                        __.append('STR')

                # IF A 2 LETTER STR IN char_seq IS ALPHA() AND NOT IN allmonthsstr, THEN NOT A DATE
                ___ = self.char_seq.lower()
                for idx in range(len(___)-1):
                    if sum([1 for x in ___[idx:idx+2] if x.isalpha()]) == 2 and \
                            ___[idx:idx+2] not in self.allmonthsstr.lower():
                        self.try_pop_except_pass(__, 'DATE')

                if self.done(__): break

                # IF char_seq HAS str AND HAS MADE IT HERE, THEN IF IS 'DATE', MUST BE AT LEAST 3 LETTER MONTH IN char_seq
                # THIS IS NOT COMPLETELTY ROBUST LOGIC
                if 'Y' in self.str_ind and 'STR' in __ and 'DATE' in __:
                    MONTHS = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
                    find_month = 'N'
                    for month in MONTHS:
                        if month in self.char_seq.lower():
                            self.try_pop_except_pass(__, 'STR')
                            break
                    else:
                        if sum([1 for _ in self.char_seq if _.isalpha()]) > 0:  # IF AN ALPHA() IS IN HERE, BUT DOESN'T CORRESPOND TO A MONTH STR
                            self.try_pop_except_pass(__, 'DATE') # AND FINISHED LOOP W/O BREAK, NO MONTH SO POP DATE, IT'S JUST A STR

            if self.done(__): break

            # DECIPHER NUM OR STR IF NUM/STR STILL IN LIST OF ELIGIBLE OBJECTS
            if 'NUM/STR' in __:
                self.try_pop_except_pass(__, 'NUM/STR')  # TAKE OUT NUM/STR SINCE THIS STILL WILL RESOLVE NUM/STR QUESTION
                # IF char_seq IS "-", THEN 'STR'
                if self.char_seq == '-':
                    __ = ['STR']
                # IF THE SAME "str_or_num" CHAR IS PRESENT MORE THAN ONCE, MUST BE STR AND NOT NUM
                elif max([self.char_seq.count(_) for _ in self.char_seq if _ in self.num_or_str_list]) > 1:
                    self.try_pop_except_pass(__, 'NUM')
                    if 'STR' not in __ and self.never_str == 'N':
                        __.append('STR')
                # CREATING A RATIO HERE TO DECIDE WHETHER A NUM/STR IS A NUM OR A STR
                # IF THE RATIO (CNT OF NUMBER_CHARS + 1) / (CNT OF STR_OR_NUMBER CHARs + 1) IS > 1, THEN NUM, ELSE STR
                # LAPLACE SMOOTHING ON THIS TO AVOID DIV/0
                elif sum([1]+[1 for _ in self.num_ind if _ == 'Y']) / sum([1]+[1 for _ in self.num_or_str_ind if _ == 'Y']) > 1:
                    self.try_pop_except_pass(__, 'STR')
                    if 'NUM' not in __ and self.never_num == 'N':
                        __.append('NUM')
                    else:
                        if self.never_str == 'N':  __.append('STR')

                # IF char_seq CAN BE CONVERTED TO FLOAT, THEN IS ['NUM'], OTHERWISE 'STR' MUST STILL BE A POSSIBILITY
                else:
                    try:
                        float(self.char_seq)
                        __ = ['NUM']
                    except:
                        if 'STR' not in __ and self.never_str == 'N':
                            __.append('STR')

            if self.done(__): break


            if __ == ['NUM']:
                __ = ['INT']  # AUTOMATICALLY SET TO 'INT', THE ONLY THING THAT CAN CHANGE THIS IS FINDING NUMB > 0 IN DEC_PLACES
                if '.' in self.char_seq:
                    for char_seq_idx in range(self.char_seq.index('.')+1, len(self.char_seq)):
                        # IF IS SCIENTIFIC NOTATION, THERE WILL BE AN e, IF SO, break ON e
                        if self.char_seq[char_seq_idx] in 'eE': break
                        if int(float(self.char_seq[char_seq_idx])) > 0:
                            __ = ['FLOAT']

                if '$' in self.char_seq:
                    __ = ['CURRENCY']
                    break

                if '-1' in self.char_seq:
                    __ = ['INT']
                    break

                if __ == ['INT']:
                    try:
                        if int(float(self.char_seq)) in [0, 1]:
                            __ = ['BIN']
                            break
                        if float(self.char_seq) == int(float(self.char_seq)):
                            break   # KEEP AS 'INT'
                        __ = ['FLOAT']
                    except:
                        self.int_conv_err_cntr += 1

                if __ == ['FLOAT']:
                    # try:
                    if float(self.char_seq) != int(float(self.char_seq)): break
                    # except:
                    #     self.int_conv_err_cntr += 1

            # AT THIS POINT IF IT'S ['STR', 'TIME'] OR ['STR', 'DATE'] THEN IT'S EITHER 'TIME' OR 'DATE' RESPECTIVELY
            # IF IT'S ['NUM', 'DATE'], THEN LOOK TO SEE IF e OR E is IN self.char_seq, THAT MAKES IT A NUM
            # AS OF 3/2/23 DONT KNOW IF ['NUM', 'TIME'] CAN HAPPEN, COPPING OUT AND JUST MAKING IT HANDLE BOTH 'DATE' AND 'TIME'
            if 'TIME' in __ and 'STR' in __: __ = ['TIME']
            elif 'DATE' in __ and 'STR' in __: __ = ['DATE']
            elif ('TIME' in __ or 'DATE' in __) and 'NUM' in __:
                if ('e' in self.char_seq or 'E' in self.char_seq) and True not in (_char in self.str_list for _char in self.char_seq):
                    if '.' in self.char_seq: __ = ['FLOAT']
                    elif not '.' in self.char_seq: __ = ['INT']
                else:
                    __.pop('NUM')




            #IF IT GETS TO THIS POINT
            if len(__) != 1:
                print(f'CANNOT FIND DATA TYPE FOR {self.char_seq}.  MISSION ABORT.')
                raise TypeError(f'\n\033[91mPROGRAM TERMINATED.  CANNOT FIND DATA TYPE FOR {self.char_seq}')
            break

        # if self.int_conv_err_cntr > 0:
        #     print(f'\n*** THERE WAS A PROBLEM WITH INT CONVERSION. ***\n')

        return __

#END FUNCTION TO HANDLE VALIDATION OF CHAR_SEQs#########################################################
########################################################################################################


