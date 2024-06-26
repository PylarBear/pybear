# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

from pybear.feature_extraction.text._TextCleaner import TextCleaner as tc
import pytest
import numpy as np
from unittest.mock import patch
import io

# PIZZA 23_01_27   TEST CODE WORKS AND TextCleaner() VERIFIED


WORDS = [
    f'\n'
    f'\n'
    f'\n'
    f'\n'
    f''
    f'FouR sc!orE an@#d sEVMn y!ears ago our fathers brought forth on '
    f'this continent, a new nation, conceived in Liberty, and dedicated '
    f'to the proposition that all men are created equal.',  ###
    f'Now we are engaged in a great civil war, testing whether that '
    f'nation, or any nation so conceived and so   dedicated, can long '
    f'endure.',  ###
    f'We are met on a great battle-field of that war.', ###
    f'We  have come   to dedicate a portion of that field, as a final '
    f'resting place for those who here gave    their lives that that '
    f'nation might live.', ###
    f''
    f'It is altogether fittingand proper that we should do this.' ###
    f'But, in a larger sense, we cannot     dedicate -- we cannot '
    f'consecrate -- we cannot hallow -- this ground.', ###
    f'The brave men, living and dead, who struggled here, have '
    f'consecrated it, far above our poor power to     add or detract.', ###
    f'The world will little note, nor long remember what we say here, but '
    f'it can never forget what they did here.', ###
    f'It is for us the living, rat##$#her, to be dedicated here to the '
    f'unfinished work which they who fought here have thus far so nobly '
    f'advanced.', ###
    f'It is rather for us to be here dedicated to the great task '
    f'remaining before us!', ###
    f'That from these honored dead we take increased devotion to that '
    f'cause for which they gave the last full      measure of d3votion', ###
    f'\n'
    f'Th!#%!at      we HERE highly resolve that these dea!#d shall not '
    f'have died in vain.    ',  ###
    f'that this NaTiOn, under God, sHALL h@a@v@e a n@ew birth of freedom.', ###
    f'     A#$@nd that go#@$ernment of the people, by the people, for the '
    f'people, shall not perish from the earth.' ###
    f'\n'
    f'\n'
    f'\n'
    f'\n'
    ]



########################################################################
########################################################################
# FOR DATA RETURNED AS LIST OF STRINGS #################################


class TestReturnedAsListOfStrings:

    @staticmethod
    @pytest.fixture
    def AS_STR_EMPTIES_DELETED():
        return [
            '\n\n\n\nFouR sc!orE an@#d sEVMn y!ears ago our fathers brought '
            'forth on this continent, a new nation, conceived in Liberty, and '
            'dedicated to the proposition that all men are created equal.',
            'Now we are engaged in a great civil war, testing whether that '
            'nation, or any nation so conceived and so   dedicated, can long '
            'endure.',
            'We are met on a great battle-field of that war.',
            'We  have come   to dedicate a portion of that field, as a final '
            'resting place for those who here gave    their lives that that '
            'nation might live.',
            'It is altogether fittingand proper that we should do this.But, '
            'in a larger sense, we cannot     dedicate -- we cannot '
            'consecrate -- we cannot hallow -- this ground.',
            'The brave men, living and dead, who struggled here, have '
            'consecrated it, far above our poor power to     add or detract.',
            'The world will little note, nor long remember what we say here, '
            'but it can never forget what they did here.',
            'It is for us the living, rat##$#her, to be dedicated here to the '
            'unfinished work which they who fought here have thus far so nobly '
            'advanced.',
            'It is rather for us to be here dedicated to the great task '
            'remaining before us!',
            'That from these honored dead we take increased devotion to that '
            'cause for which they gave the last full      measure of d3votion',
            '\nTh!#%!at      we HERE highly resolve that these dea!#d shall '
            'not have died in vain.    ',
            'that this NaTiOn, under God, sHALL h@a@v@e a n@ew birth of '
            'freedom.',
            '     A#$@nd that go#@$ernment of the people, by the people, for '
            'the people, shall not perish from the earth.\n\n\n\n'
        ]

    @staticmethod
    @pytest.fixture
    def AS_STR_CHARS_REMOVED():
        return [
            'FouR scorE and sEVMn years ago our fathers brought forth on this '
            'continent a new nation conceived in Liberty and dedicated to the '
            'proposition that all men are created equal',
            'Now we are engaged in a great civil war testing whether that '
            'nation or any nation so conceived and so   dedicated can long '
            'endure',
            'We are met on a great battlefield of that war',
            'We  have come   to dedicate a portion of that field as a final '
            'resting place for those who here gave    their lives that that '
            'nation might live',
            'It is altogether fittingand proper that we should do thisBut in '
            'a larger sense we cannot     dedicate  we cannot consecrate  we '
            'cannot hallow  this ground',
            'The brave men living and dead who struggled here have consecrated '
            'it far above our poor power to     add or detract',
            'The world will little note nor long remember what we say here '
            'but it can never forget what they did here',
            'It is for us the living rather to be dedicated here to the '
            'unfinished work which they who fought here have thus far so '
            'nobly advanced',
            'It is rather for us to be here dedicated to the great task '
            'remaining before us',
            'That from these honored dead we take increased devotion to that '
            'cause for which they gave the last full      measure of d3votion',
            'That      we HERE highly resolve that these dead shall not have '
            'died in vain    ',
            'that this NaTiOn under God sHALL have a new birth of freedom',
            '     And that goernment of the people by the people for the '
            'people shall not perish from the earth'
        ]

    @staticmethod
    @pytest.fixture
    def AS_STR_STRIPPED():
        return [
            'FouR scorE and sEVMn years ago our fathers brought forth on this '
            'continent a new nation conceived in Liberty and dedicated to '
            'the proposition that all men are created equal',
            'Now we are engaged in a great civil war testing whether that '
            'nation or any nation so conceived and so dedicated can long '
            'endure',
            'We are met on a great battlefield of that war',
            'We have come to dedicate a portion of that field as a final '
            'resting place for those who here gave their lives that that '
            'nation might live',
            'It is altogether fittingand proper that we should do thisBut '
            'in a larger sense we cannot dedicate we cannot consecrate we '
            'cannot hallow this ground',
            'The brave men living and dead who struggled here have '
            'consecrated it far above our poor power to add or detract',
            'The world will little note nor long remember what we say here '
            'but it can never forget what they did here',
            'It is for us the living rather to be dedicated here to the '
            'unfinished work which they who fought here have thus far so '
            'nobly advanced',
            'It is rather for us to be here dedicated to the great task '
            'remaining before us',
            'That from these honored dead we take increased devotion to that '
            'cause for which they gave the last full measure of d3votion',
            'That we HERE highly resolve that these dead shall not have died '
            'in vain',
            'that this NaTiOn under God sHALL have a new birth of freedom',
            'And that goernment of the people by the people for the people '
            'shall not perish from the earth'
        ]

    @staticmethod
    @pytest.fixture
    def AS_STR_NORMALIZED():
        return [
            'FOUR SCORE AND SEVMN YEARS AGO OUR FATHERS BROUGHT FORTH ON THIS '
            'CONTINENT A NEW NATION CONCEIVED IN LIBERTY AND DEDICATED TO THE '
            'PROPOSITION THAT ALL MEN ARE CREATED EQUAL',
            'NOW WE ARE ENGAGED IN A GREAT CIVIL WAR TESTING WHETHER THAT '
            'NATION OR ANY NATION SO CONCEIVED AND SO DEDICATED CAN LONG '
            'ENDURE',
            'WE ARE MET ON A GREAT BATTLEFIELD OF THAT WAR',
            'WE HAVE COME TO DEDICATE A PORTION OF THAT FIELD AS A FINAL '
            'RESTING PLACE FOR THOSE WHO HERE GAVE THEIR LIVES THAT THAT '
            'NATION MIGHT LIVE',
            'IT IS ALTOGETHER FITTINGAND PROPER THAT WE SHOULD DO THISBUT IN '
            'A LARGER SENSE WE CANNOT DEDICATE WE CANNOT CONSECRATE WE CANNOT '
            'HALLOW THIS GROUND',
            'THE BRAVE MEN LIVING AND DEAD WHO STRUGGLED HERE HAVE CONSECRATED '
            'IT FAR ABOVE OUR POOR POWER TO ADD OR DETRACT',
            'THE WORLD WILL LITTLE NOTE NOR LONG REMEMBER WHAT WE SAY HERE '
            'BUT IT CAN NEVER FORGET WHAT THEY DID HERE',
            'IT IS FOR US THE LIVING RATHER TO BE DEDICATED HERE TO THE '
            'UNFINISHED WORK WHICH THEY WHO FOUGHT HERE HAVE THUS FAR SO '
            'NOBLY ADVANCED',
            'IT IS RATHER FOR US TO BE HERE DEDICATED TO THE GREAT TASK '
            'REMAINING BEFORE US',
            'THAT FROM THESE HONORED DEAD WE TAKE INCREASED DEVOTION TO THAT '
            'CAUSE FOR WHICH THEY GAVE THE LAST FULL MEASURE OF D3VOTION',
            'THAT WE HERE HIGHLY RESOLVE THAT THESE DEAD SHALL NOT HAVE DIED '
            'IN VAIN',
            'THAT THIS NATION UNDER GOD SHALL HAVE A NEW BIRTH OF FREEDOM',
            'AND THAT GOERNMENT OF THE PEOPLE BY THE PEOPLE FOR THE PEOPLE '
            'SHALL NOT PERISH FROM THE EARTH'
        ]

    @staticmethod
    @pytest.fixture
    def AS_STR_LOOKEDUP():
        return [
             'FOUR SCORE AND SEVEN YEARS AGO OUR FATHERS BROUGHT FORTH ON '
             'THIS CONTINENT A NEW NATION CONCEIVED IN LIBERTY AND DEDICATED '
             'TO THE PROPOSITION THAT ALL MEN ARE CREATED EQUAL',
             'NOW WE ARE ENGAGED IN A GREAT CIVIL WAR TESTING WHETHER THAT '
             'NATION OR ANY NATION SO CONCEIVED AND SO DEDICATED CAN LONG '
             'ENDURE',
             'WE ARE MET ON A GREAT BATTLEFIELD OF THAT WAR',
             'WE HAVE COME TO DEDICATE A PORTION OF THAT FIELD AS A FINAL '
             'RESTING PLACE FOR THOSE WHO HERE GAVE THEIR LIVES THAT THAT '
             'NATION MIGHT LIVE',
             'IT IS ALTOGETHER FITTING AND PROPER THAT WE SHOULD DO THIS BUT '
             'IN A LARGER SENSE WE CANNOT DEDICATE WE CANNOT CONSECRATE WE '
             'CANNOT HALLOW THIS GROUND',
             'THE BRAVE MEN LIVING AND DEAD WHO STRUGGLED HERE HAVE '
             'CONSECRATED IT FAR ABOVE OUR POOR POWER TO ADD OR DETRACT',
             'THE WORLD WILL LITTLE NOTE NOR LONG REMEMBER WHAT WE SAY HERE '
             'BUT IT CAN NEVER FORGET WHAT THEY DID HERE',
             'IT IS FOR US THE LIVING RATHER TO BE DEDICATED HERE TO THE '
             'UNFINISHED WORK WHICH THEY WHO FOUGHT HERE HAVE THUS FAR SO '
             'NOBLY ADVANCED',
             'IT IS RATHER FOR US TO BE HERE DEDICATED TO THE GREAT TASK '
             'REMAINING BEFORE US',
             'THAT FROM THESE HONORED DEAD WE TAKE INCREASED DEVOTION TO THAT '
             'CAUSE FOR WHICH THEY GAVE THE LAST FULL MEASURE OF DEVOTION',
             'THAT WE HERE HIGHLY RESOLVE THAT THESE DEAD SHALL NOT HAVE DIED '
             'IN VAIN',
             'THAT THIS NATION UNDER GOD SHALL HAVE A NEW BIRTH OF FREEDOM',
             'AND THAT GOVERNMENT OF THE PEOPLE BY THE PEOPLE FOR THE PEOPLE '
             'SHALL NOT PERISH FROM THE EARTH'
        ]

    @staticmethod
    @pytest.fixture
    def AS_STR_ROW_UNIQUES():
        return [
            ['A', 'AGO', 'ALL', 'AND', 'ARE', 'BROUGHT', 'CONCEIVED',
             'CONTINENT', 'CREATED', 'DEDICATED', 'EQUAL', 'FATHERS', 'FORTH',
             'FOUR', 'IN', 'LIBERTY', 'MEN', 'NATION', 'NEW', 'ON', 'OUR',
             'PROPOSITION', 'SCORE', 'SEVEN', 'THAT', 'THE', 'THIS', 'TO',
             'YEARS'],
            ['A', 'AND', 'ANY', 'ARE', 'CAN', 'CIVIL', 'CONCEIVED',
             'DEDICATED', 'ENDURE', 'ENGAGED', 'GREAT', 'IN', 'LONG',
             'NATION', 'NOW', 'OR', 'SO', 'TESTING', 'THAT', 'WAR', 'WE',
             'WHETHER'],
            ['A', 'ARE', 'BATTLEFIELD', 'GREAT', 'MET', 'OF', 'ON', 'THAT',
             'WAR', 'WE'],
            ['A', 'AS', 'COME', 'DEDICATE', 'FIELD', 'FINAL', 'FOR', 'GAVE',
             'HAVE', 'HERE', 'LIVE', 'LIVES', 'MIGHT', 'NATION', 'OF',
             'PLACE', 'PORTION', 'RESTING', 'THAT', 'THEIR', 'THOSE', 'TO',
             'WE', 'WHO'],
            ['A', 'ALTOGETHER', 'AND', 'BUT', 'CANNOT', 'CONSECRATE',
             'DEDICATE', 'DO', 'FITTING', 'GROUND', 'HALLOW', 'IN', 'IS',
             'IT', 'LARGER', 'PROPER', 'SENSE', 'SHOULD', 'THAT', 'THIS',
             'WE'],
            ['ABOVE', 'ADD', 'AND', 'BRAVE', 'CONSECRATED', 'DEAD', 'DETRACT',
             'FAR', 'HAVE', 'HERE', 'IT', 'LIVING', 'MEN', 'OR', 'OUR',
             'POOR', 'POWER', 'STRUGGLED', 'THE', 'TO', 'WHO'],
            ['BUT', 'CAN', 'DID', 'FORGET', 'HERE', 'IT', 'LITTLE', 'LONG',
             'NEVER', 'NOR', 'NOTE', 'REMEMBER', 'SAY', 'THE', 'THEY', 'WE',
             'WHAT', 'WILL', 'WORLD'],
            ['ADVANCED', 'BE', 'DEDICATED', 'FAR', 'FOR', 'FOUGHT', 'HAVE',
             'HERE', 'IS', 'IT', 'LIVING', 'NOBLY', 'RATHER', 'SO', 'THE',
             'THEY', 'THUS', 'TO', 'UNFINISHED', 'US', 'WHICH', 'WHO',
             'WORK'],
            ['BE', 'BEFORE', 'DEDICATED', 'FOR', 'GREAT', 'HERE', 'IS', 'IT',
             'RATHER', 'REMAINING', 'TASK', 'THE', 'TO', 'US'],
            ['CAUSE', 'DEAD', 'DEVOTION', 'FOR', 'FROM', 'FULL', 'GAVE',
             'HONORED', 'INCREASED', 'LAST', 'MEASURE', 'OF', 'TAKE', 'THAT',
             'THE', 'THESE', 'THEY', 'TO', 'WE', 'WHICH'],
            ['DEAD', 'DIED', 'HAVE', 'HERE', 'HIGHLY', 'IN', 'NOT', 'RESOLVE',
             'SHALL', 'THAT', 'THESE', 'VAIN', 'WE'],
            ['A', 'BIRTH', 'FREEDOM', 'GOD', 'HAVE', 'NATION', 'NEW', 'OF',
             'SHALL', 'THAT', 'THIS', 'UNDER'],
            ['AND', 'BY', 'EARTH', 'FOR', 'FROM', 'GOVERNMENT', 'NOT', 'OF',
             'PEOPLE', 'PERISH', 'SHALL', 'THAT', 'THE']
        ]

    @staticmethod
    @pytest.fixture
    def AS_STR_ROW_UNIQUES_COUNTS():
        return [
         [1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
          1, 1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
         [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1,
          1, 1],
         [1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 4],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1],
         [1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1,
          1],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2],
         [1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 4]
        ]

    @staticmethod
    @pytest.fixture
    def AS_STR_ALL_UNIQUES():
        return [
        'A', 'ABOVE', 'ADD', 'ADVANCED', 'AGO', 'ALL', 'ALTOGETHER', 'AND',
        'ANY', 'ARE', 'AS', 'BATTLEFIELD', 'BE', 'BEFORE', 'BIRTH', 'BRAVE',
        'BROUGHT', 'BUT', 'BY', 'CAN', 'CANNOT', 'CAUSE', 'CIVIL', 'COME',
        'CONCEIVED', 'CONSECRATE', 'CONSECRATED', 'CONTINENT', 'CREATED',
        'DEAD', 'DEDICATE', 'DEDICATED',
        'DETRACT', 'DEVOTION', 'DID', 'DIED', 'DO', 'EARTH', 'ENDURE',
        'ENGAGED', 'EQUAL', 'FAR', 'FATHERS', 'FIELD', 'FINAL', 'FITTING',
        'FOR', 'FORGET', 'FORTH', 'FOUGHT', 'FOUR', 'FREEDOM', 'FROM', 'FULL',
        'GAVE', 'GOD', 'GOVERNMENT', 'GREAT', 'GROUND', 'HALLOW', 'HAVE',
        'HERE', 'HIGHLY', 'HONORED', 'IN', 'INCREASED', 'IS', 'IT', 'LARGER',
        'LAST', 'LIBERTY', 'LITTLE', 'LIVE', 'LIVES', 'LIVING', 'LONG',
        'MEASURE', 'MEN', 'MET', 'MIGHT', 'NATION', 'NEVER', 'NEW', 'NOBLY',
        'NOR', 'NOT', 'NOTE', 'NOW', 'OF', 'ON', 'OR', 'OUR', 'PEOPLE',
        'PERISH', 'PLACE', 'POOR', 'PORTION', 'POWER', 'PROPER', 'PROPOSITION',
        'RATHER', 'REMAINING', 'REMEMBER', 'RESOLVE', 'RESTING', 'SAY',
        'SCORE', 'SENSE', 'SEVEN', 'SHALL', 'SHOULD', 'SO', 'STRUGGLED',
        'TAKE', 'TASK', 'TESTING', 'THAT', 'THE', 'THEIR', 'THESE', 'THEY',
        'THIS', 'THOSE', 'THUS', 'TO', 'UNDER', 'UNFINISHED', 'US', 'VAIN',
        'WAR', 'WE', 'WHAT', 'WHETHER', 'WHICH', 'WHO', 'WILL', 'WORK',
        'WORLD', 'YEARS'
        ]

    @staticmethod
    @pytest.fixture
    def AS_STR_ALL_UNIQUES_COUNTS():

        return [
            7, 1, 1, 1, 1, 1, 1, 6, 1, 3, 1, 1, 2, 1, 1, 1, 1, 2, 1, 2, 3, 1, 1,
            1, 2, 1, 1, 1, 1, 3, 2, 4, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1,
            5, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 3, 1, 1, 5, 8, 1, 1, 4, 1, 3, 5, 1,
            1, 1, 1, 1, 1, 2, 2, 1, 2, 1, 1, 5, 1, 2, 1, 1, 2, 1, 1, 5, 2, 2, 2,
            3, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 3, 1, 1, 1,
            1, 13, 11, 1, 2, 3, 4, 1, 1, 8, 1, 1, 3, 1, 2, 10, 2, 1, 2, 3, 1, 1,
            1, 1
        ]

    @staticmethod
    @pytest.fixture
    def AS_STR_STOPS_REMOVED():
        return [
        'FOUR SCORE SEVEN YEARS AGO FATHERS BROUGHT FORTH CONTINENT NEW '
        'NATION CONCEIVED LIBERTY DEDICATED PROPOSITION MEN CREATED EQUAL',
        'ENGAGED CIVIL WAR TESTING WHETHER NATION NATION CONCEIVED '
        'DEDICATED ENDURE',
        'MET BATTLEFIELD WAR',
        'DEDICATE PORTION FIELD FINAL RESTING PLACE THOSE GAVE LIVES '
        'NATION MIGHT LIVE',
        'ALTOGETHER FITTING PROPER LARGER SENSE CANNOT DEDICATE CANNOT '
        'CONSECRATE CANNOT HALLOW GROUND',
        'BRAVE MEN LIVING DEAD STRUGGLED CONSECRATED ABOVE POOR POWER ADD DETRACT',
        'WORLD NOTE NOR REMEMBER FORGET',
        'LIVING RATHER DEDICATED UNFINISHED WORK FOUGHT THUS NOBLY ADVANCED',
        'RATHER DEDICATED TASK REMAINING',
        'HONORED DEAD TAKE INCREASED DEVOTION CAUSE GAVE FULL MEASURE DEVOTION',
        'HIGHLY RESOLVE DEAD DIED VAIN',
        'NATION GOD NEW BIRTH FREEDOM',
        'GOVERNMENT PEOPLE PEOPLE PEOPLE PERISH EARTH'
    ]


    # delete_empty_rows     Remove textless rows from data.
    # remove_characters     Keep only allowed or removed disallowed characters
    #                       from entire CLEANED_TEXT object.
    # _strip                Remove multiple spaces and leading and trailing
    #                       spaces from all text in CLEAND_TEXT object.
    # normalize             Set all text in CLEANED_TEXT object to upper case
    #                       (default) or lower case.
    # lex_lookup            Scan entire CLEANED_TEXT object and prompt for
    #                       handling of words not in LEXICON.  <<<<<
    # return_row_uniques    Return a potentially ragged vector containing the
    #                       unique words for each row in CLEANED_TEXT object.
    # return_overall_uniques Return unique words in the entire CLEANED_TEXT
    #                       object.
    # remove_stops          Remove stop words from the entire CLEANED_TEXT
    #                       object.
    # dump_to_csv           Dump CLEANED_TEXT object to csv.
    # dump_to_txt           Dump CLEANED_TEXT object to txt.

    # @classmethod
    # @pytest.fixture(scope="class", autouse=True)
    # def setup_class(self):
    TestClass = tc.TextCleaner(WORDS, update_lexicon=False)
    TestClass.as_list_of_strs()


    def test_deleted_empty_rows(self, AS_STR_EMPTIES_DELETED):

        self.TestClass.delete_empty_rows()
        assert np.array_equiv(self.TestClass.CLEANED_TEXT,
                              AS_STR_EMPTIES_DELETED
        )


    def test_remove_characters(self, AS_STR_CHARS_REMOVED):

        self.TestClass.remove_characters()
        assert np.array_equiv(self.TestClass.CLEANED_TEXT,
                              AS_STR_CHARS_REMOVED
        )


    def test_strip(self, AS_STR_STRIPPED):

        self.TestClass._strip()
        assert np.array_equiv(self.TestClass.CLEANED_TEXT, AS_STR_STRIPPED)


    def test_normalize(self, AS_STR_NORMALIZED):

        self.TestClass.normalize()
        assert np.array_equiv(self.TestClass.CLEANED_TEXT, AS_STR_NORMALIZED)


    # THIS IS INTERACTIVE
    def test_lex_lookup(self, AS_STR_LOOKEDUP):

        user_inputs = "e\nSEVEN\nY\nY\nY\ne\nDEVOTION\nY\ne\nGOVERNMENT\nY\n"
        with patch('sys.stdin', io.StringIO(user_inputs)):
            self.TestClass.lex_lookup()

        assert np.array_equiv(self.TestClass.CLEANED_TEXT, AS_STR_LOOKEDUP)


    def test_return_row_uniques(self, AS_STR_ROW_UNIQUES, AS_STR_ROW_UNIQUES_COUNTS):

        # COMPARE ROW UNIQUES ROW BY ROW
        for idx, ROW in enumerate(self.TestClass.return_row_uniques(return_counts=False)):
            assert np.array_equiv(ROW, AS_STR_ROW_UNIQUES[idx])


        # COMPARE ROW UNIQUES IN TOTALITY
        ROW_UNIQUES = self.TestClass.return_row_uniques(return_counts=False)
        # CONVERT return_row_uniques() TO list(list()) FOR COMPARISON,
        # UNEXPECTEDLY EXCEPTING WHEN NUMPY

        assert np.array_equiv(ROW_UNIQUES, AS_STR_ROW_UNIQUES)

        del ROW_UNIQUES


        ROW_UNIQUES, ROW_COUNTS = self.TestClass.return_row_uniques(return_counts=True)
        # CONVERT return_row_uniques() TO list(list()) FOR COMPARISON,
        # UNEXPECTEDLY EXCEPTING WHEN NUMPY
        assert np.array_equiv(list(map(list, ROW_UNIQUES)), AS_STR_ROW_UNIQUES)
        assert np.array_equiv(list(map(list, ROW_COUNTS)), AS_STR_ROW_UNIQUES_COUNTS)

        del ROW_UNIQUES, ROW_COUNTS


    def test_return_overall_uniques(self, AS_STR_ALL_UNIQUES, AS_STR_ALL_UNIQUES_COUNTS):

        ALL_UNIQUES = self.TestClass.return_overall_uniques(return_counts=False)
        assert np.array_equiv(ALL_UNIQUES, AS_STR_ALL_UNIQUES)

        del ALL_UNIQUES


        ALL_UNIQUES, ALL_COUNTS = \
            self.TestClass.return_overall_uniques(return_counts=True)
        assert np.array_equiv(ALL_UNIQUES, AS_STR_ALL_UNIQUES)
        assert np.array_equiv(ALL_COUNTS, AS_STR_ALL_UNIQUES_COUNTS)

        del ALL_UNIQUES, ALL_COUNTS


    def test_remove_stops(self, AS_STR_STOPS_REMOVED):

        self.TestClass.remove_stops()
        STOPS_REMOVED = self.TestClass.CLEANED_TEXT
        assert np.array_equiv(STOPS_REMOVED, AS_STR_STOPS_REMOVED)

        del STOPS_REMOVED


    def test_dump_to_csv(self):

        # dump_to_csv               Dump CLEANED_TEXT object to csv.
        user_inputs = "dump_to_csv_test_dump\nY\n"
        with patch('sys.stdin', io.StringIO(user_inputs)):
            self.TestClass.dump_to_csv()


    def test_dump_to_txt(self):

        # dump_to_txt               Dump CLEANED_TEXT object to txt.
        user_inputs = "dump_to_txt_test_dump\nY\n"
        with patch('sys.stdin', io.StringIO(user_inputs)):
            self.TestClass.dump_to_txt()


# END FOR DATA RETURNED AS LIST OF STRINGS #############################
########################################################################
########################################################################



########################################################################
########################################################################
# FOR DATA RETURNED AS LIST OF LISTS ###################################

class TestReturnedAsListOfLists:

    @staticmethod
    @pytest.fixture
    def AS_LISTS_EMPTIES_DELETED():
        return [
            ['FouR', 'sc!orE', 'an@#d', 'sEVMn', 'y!ears', 'ago', 'our',
             'fathers', 'brought', 'forth', 'on', 'this', 'continent,', 'a',
             'new', 'nation,', 'conceived', 'in', 'Liberty,', 'and',
             'dedicated', 'to', 'the', 'proposition', 'that', 'all', 'men',
             'are', 'created', 'equal.'],
            ['Now', 'we', 'are', 'engaged', 'in', 'a', 'great', 'civil',
             'war,', 'testing', 'whether', 'that', 'nation,', 'or', 'any',
             'nation', 'so', 'conceived', 'and', 'so', 'dedicated,', 'can',
             'long', 'endure.'],
            ['We', 'are', 'met', 'on', 'a', 'great', 'battle-field', 'of',
             'that', 'war.'],
            ['We', 'have', 'come', 'to', 'dedicate', 'a', 'portion', 'of',
             'that', 'field,', 'as', 'a', 'final', 'resting', 'place', 'for',
             'those', 'who', 'here', 'gave', 'their', 'lives', 'that', 'that',
             'nation', 'might', 'live.'],
            ['It', 'is', 'altogether', 'fittingand', 'proper', 'that', 'we',
             'should', 'do', 'this.But,', 'in', 'a', 'larger', 'sense,', 'we',
             'cannot', 'dedicate', '--', 'we', 'cannot', 'consecrate', '--',
             'we', 'cannot', 'hallow', '--', 'this', 'ground.'],
            ['The', 'brave', 'men,', 'living', 'and', 'dead,', 'who',
             'struggled', 'here,', 'have', 'consecrated', 'it,', 'far',
             'above', 'our', 'poor', 'power', 'to', 'add', 'or', 'detract.'],
            ['The', 'world', 'will', 'little', 'note,', 'nor', 'long',
             'remember', 'what', 'we', 'say', 'here,', 'but', 'it', 'can',
             'never', 'forget', 'what', 'they', 'did', 'here.'],
            ['It', 'is', 'for', 'us', 'the', 'living,', 'rat##$#her,', 'to',
             'be', 'dedicated', 'here', 'to', 'the', 'unfinished', 'work',
             'which', 'they', 'who', 'fought', 'here', 'have', 'thus', 'far',
             'so', 'nobly', 'advanced.'],
            ['It', 'is', 'rather', 'for', 'us', 'to', 'be', 'here',
             'dedicated', 'to', 'the', 'great', 'task', 'remaining', 'before',
             'us!'],
            ['That', 'from', 'these', 'honored', 'dead', 'we', 'take',
             'increased', 'devotion', 'to', 'that', 'cause', 'for', 'which',
             'they', 'gave', 'the', 'last', 'full', 'measure', 'of',
             'd3votion'],
            ['Th!#%!at', 'we', 'HERE', 'highly', 'resolve', 'that', 'these',
             'dea!#d', 'shall', 'not', 'have', 'died', 'in', 'vain.'],
            ['that', 'this', 'NaTiOn,', 'under', 'God,', 'sHALL', 'h@a@v@e',
             'a', 'n@ew', 'birth', 'of', 'freedom.'],
            ['A#$@nd', 'that', 'go#@$ernment', 'of', 'the', 'people,', 'by',
             'the', 'people,', 'for', 'the', 'people,', 'shall', 'not',
             'perish', 'from', 'the', 'earth.']
        ]

    @staticmethod
    @pytest.fixture
    def AS_LISTS_CHARS_REMOVED():
        return [
            ['FouR', 'scorE', 'and', 'sEVMn', 'years', 'ago', 'our', 'fathers',
             'brought', 'forth', 'on', 'this', 'continent', 'a', 'new',
             'nation', 'conceived', 'in', 'Liberty', 'and', 'dedicated', 'to',
             'the', 'proposition', 'that', 'all', 'men', 'are', 'created',
             'equal'],
            ['Now', 'we', 'are', 'engaged', 'in', 'a', 'great', 'civil', 'war',
             'testing', 'whether', 'that', 'nation', 'or', 'any', 'nation',
             'so', 'conceived', 'and', 'so', 'dedicated', 'can', 'long',
             'endure'],
            ['We', 'are', 'met', 'on', 'a', 'great', 'battlefield', 'of',
             'that', 'war'],
            ['We', 'have', 'come', 'to', 'dedicate', 'a', 'portion', 'of',
             'that', 'field', 'as', 'a', 'final', 'resting', 'place', 'for',
             'those', 'who', 'here', 'gave', 'their', 'lives', 'that', 'that',
             'nation', 'might', 'live'],
            ['It', 'is', 'altogether', 'fittingand', 'proper', 'that', 'we',
             'should', 'do', 'thisBut', 'in', 'a', 'larger', 'sense', 'we',
             'cannot', 'dedicate', 'we', 'cannot', 'consecrate', 'we',
             'cannot', 'hallow', 'this', 'ground'],
            ['The', 'brave', 'men', 'living', 'and', 'dead', 'who',
             'struggled', 'here', 'have', 'consecrated', 'it', 'far', 'above',
             'our', 'poor', 'power', 'to', 'add', 'or', 'detract'],
            ['The', 'world', 'will', 'little', 'note', 'nor', 'long',
             'remember', 'what', 'we', 'say', 'here', 'but', 'it', 'can',
             'never', 'forget', 'what', 'they', 'did', 'here'],
            ['It', 'is', 'for', 'us', 'the', 'living', 'rather', 'to', 'be',
             'dedicated', 'here', 'to', 'the', 'unfinished', 'work', 'which',
             'they', 'who', 'fought', 'here', 'have', 'thus', 'far', 'so',
             'nobly', 'advanced'],
            ['It', 'is', 'rather', 'for', 'us', 'to', 'be', 'here',
             'dedicated', 'to', 'the', 'great', 'task', 'remaining', 'before',
             'us'],
            ['That', 'from', 'these', 'honored', 'dead', 'we', 'take',
             'increased', 'devotion', 'to', 'that', 'cause', 'for', 'which',
             'they', 'gave', 'the', 'last', 'full', 'measure', 'of',
             'd3votion'],
            ['That', 'we', 'HERE', 'highly', 'resolve', 'that', 'these',
             'dead', 'shall', 'not', 'have', 'died', 'in', 'vain'],
            ['that', 'this', 'NaTiOn', 'under', 'God', 'sHALL', 'have', 'a',
             'new', 'birth', 'of', 'freedom'],
            ['And', 'that', 'goernment', 'of', 'the', 'people', 'by', 'the',
             'people', 'for', 'the', 'people', 'shall', 'not', 'perish',
             'from', 'the', 'earth']
         ]


    @staticmethod
    @pytest.fixture
    def AS_LISTS_STRIPPED():
        return [
            ['FouR', 'scorE', 'and', 'sEVMn', 'years', 'ago', 'our',
             'fathers', 'brought', 'forth', 'on', 'this', 'continent', 'a',
             'new', 'nation', 'conceived', 'in', 'Liberty', 'and', 'dedicated',
             'to', 'the', 'proposition', 'that', 'all', 'men', 'are',
             'created', 'equal'],
            ['Now', 'we', 'are', 'engaged', 'in', 'a', 'great', 'civil',
             'war', 'testing', 'whether', 'that', 'nation', 'or', 'any',
             'nation', 'so', 'conceived', 'and', 'so', 'dedicated', 'can',
             'long', 'endure'],
            ['We', 'are', 'met', 'on', 'a', 'great', 'battlefield', 'of',
             'that', 'war'],
            ['We', 'have', 'come', 'to', 'dedicate', 'a', 'portion', 'of',
             'that', 'field', 'as', 'a', 'final', 'resting', 'place', 'for',
             'those', 'who', 'here', 'gave', 'their', 'lives', 'that', 'that',
             'nation', 'might', 'live'],
            ['It', 'is', 'altogether', 'fittingand', 'proper', 'that', 'we',
             'should', 'do', 'thisBut', 'in', 'a', 'larger', 'sense', 'we',
             'cannot', 'dedicate', 'we', 'cannot', 'consecrate', 'we',
             'cannot', 'hallow', 'this', 'ground'],
            ['The', 'brave', 'men', 'living', 'and', 'dead', 'who',
             'struggled', 'here', 'have', 'consecrated', 'it', 'far', 'above',
             'our', 'poor', 'power', 'to', 'add', 'or', 'detract'],
            ['The', 'world', 'will', 'little', 'note', 'nor', 'long',
             'remember', 'what', 'we', 'say', 'here', 'but', 'it', 'can',
             'never', 'forget', 'what', 'they', 'did', 'here'],
            ['It', 'is', 'for', 'us', 'the', 'living', 'rather', 'to', 'be',
             'dedicated', 'here', 'to', 'the', 'unfinished', 'work', 'which',
             'they', 'who', 'fought', 'here', 'have', 'thus', 'far', 'so',
             'nobly', 'advanced'],
            ['It', 'is', 'rather', 'for', 'us', 'to', 'be', 'here',
             'dedicated', 'to', 'the', 'great', 'task', 'remaining', 'before',
             'us'],
            ['That', 'from', 'these', 'honored', 'dead', 'we', 'take',
             'increased', 'devotion', 'to', 'that', 'cause', 'for', 'which',
             'they', 'gave', 'the', 'last', 'full', 'measure', 'of',
             'd3votion'],
            ['That', 'we', 'HERE', 'highly', 'resolve', 'that', 'these',
             'dead', 'shall', 'not', 'have', 'died', 'in', 'vain'],
            ['that', 'this', 'NaTiOn', 'under', 'God', 'sHALL', 'have', 'a',
             'new', 'birth', 'of', 'freedom'],
            ['And', 'that', 'goernment', 'of', 'the', 'people', 'by', 'the',
             'people', 'for', 'the', 'people', 'shall', 'not', 'perish',
             'from', 'the', 'earth']
        ]


    @staticmethod
    @pytest.fixture
    def AS_LISTS_NORMALIZED():
        return [
            ['FOUR', 'SCORE', 'AND', 'SEVMN', 'YEARS', 'AGO', 'OUR',
             'FATHERS', 'BROUGHT', 'FORTH', 'ON', 'THIS', 'CONTINENT', 'A',
             'NEW', 'NATION', 'CONCEIVED', 'IN', 'LIBERTY', 'AND', 'DEDICATED',
             'TO', 'THE', 'PROPOSITION', 'THAT', 'ALL', 'MEN', 'ARE',
             'CREATED', 'EQUAL'],
            ['NOW', 'WE', 'ARE', 'ENGAGED', 'IN', 'A', 'GREAT', 'CIVIL',
             'WAR', 'TESTING', 'WHETHER', 'THAT', 'NATION', 'OR', 'ANY',
             'NATION', 'SO', 'CONCEIVED', 'AND', 'SO', 'DEDICATED', 'CAN',
             'LONG', 'ENDURE'],
            ['WE', 'ARE', 'MET', 'ON', 'A', 'GREAT', 'BATTLEFIELD', 'OF',
             'THAT', 'WAR'],
            ['WE', 'HAVE', 'COME', 'TO', 'DEDICATE', 'A', 'PORTION', 'OF',
             'THAT', 'FIELD', 'AS', 'A', 'FINAL', 'RESTING', 'PLACE', 'FOR',
             'THOSE', 'WHO', 'HERE', 'GAVE', 'THEIR', 'LIVES', 'THAT', 'THAT',
             'NATION', 'MIGHT', 'LIVE'],
            ['IT', 'IS', 'ALTOGETHER', 'FITTINGAND', 'PROPER', 'THAT', 'WE',
             'SHOULD', 'DO', 'THISBUT', 'IN', 'A', 'LARGER', 'SENSE', 'WE',
             'CANNOT', 'DEDICATE', 'WE', 'CANNOT', 'CONSECRATE', 'WE',
             'CANNOT', 'HALLOW', 'THIS', 'GROUND'],
            ['THE', 'BRAVE', 'MEN', 'LIVING', 'AND', 'DEAD', 'WHO',
             'STRUGGLED', 'HERE', 'HAVE', 'CONSECRATED', 'IT', 'FAR', 'ABOVE',
             'OUR', 'POOR', 'POWER', 'TO', 'ADD', 'OR', 'DETRACT'],
            ['THE', 'WORLD', 'WILL', 'LITTLE', 'NOTE', 'NOR', 'LONG',
             'REMEMBER', 'WHAT', 'WE', 'SAY', 'HERE', 'BUT', 'IT', 'CAN',
             'NEVER', 'FORGET', 'WHAT', 'THEY', 'DID', 'HERE'],
            ['IT', 'IS', 'FOR', 'US', 'THE', 'LIVING', 'RATHER', 'TO', 'BE',
             'DEDICATED', 'HERE', 'TO', 'THE', 'UNFINISHED', 'WORK', 'WHICH',
             'THEY', 'WHO', 'FOUGHT', 'HERE', 'HAVE', 'THUS', 'FAR', 'SO',
             'NOBLY', 'ADVANCED'],
            ['IT', 'IS', 'RATHER', 'FOR', 'US', 'TO', 'BE', 'HERE',
             'DEDICATED', 'TO', 'THE', 'GREAT', 'TASK', 'REMAINING', 'BEFORE',
             'US'],
            ['THAT', 'FROM', 'THESE', 'HONORED', 'DEAD', 'WE', 'TAKE',
             'INCREASED', 'DEVOTION', 'TO', 'THAT', 'CAUSE', 'FOR', 'WHICH',
             'THEY', 'GAVE', 'THE', 'LAST', 'FULL', 'MEASURE', 'OF',
             'D3VOTION'],
            ['THAT', 'WE', 'HERE', 'HIGHLY', 'RESOLVE', 'THAT', 'THESE',
             'DEAD', 'SHALL', 'NOT', 'HAVE', 'DIED', 'IN', 'VAIN'],
            ['THAT', 'THIS', 'NATION', 'UNDER', 'GOD', 'SHALL', 'HAVE', 'A',
             'NEW', 'BIRTH', 'OF', 'FREEDOM'],
            ['AND', 'THAT', 'GOERNMENT', 'OF', 'THE', 'PEOPLE', 'BY', 'THE',
             'PEOPLE', 'FOR', 'THE', 'PEOPLE', 'SHALL', 'NOT', 'PERISH',
             'FROM', 'THE', 'EARTH']
        ]

    @staticmethod
    @pytest.fixture
    def AS_LISTS_LOOKEDUP():
        return [
            ['FOUR', 'SCORE', 'AND', 'SEVEN', 'YEARS', 'AGO', 'OUR',
             'FATHERS', 'BROUGHT', 'FORTH', 'ON', 'THIS', 'CONTINENT', 'A',
             'NEW', 'NATION', 'CONCEIVED', 'IN', 'LIBERTY', 'AND', 'DEDICATED',
             'TO', 'THE', 'PROPOSITION', 'THAT', 'ALL', 'MEN', 'ARE',
             'CREATED', 'EQUAL'],
            ['NOW', 'WE', 'ARE', 'ENGAGED', 'IN', 'A', 'GREAT', 'CIVIL', 'WAR',
             'TESTING', 'WHETHER', 'THAT', 'NATION', 'OR', 'ANY', 'NATION',
             'SO', 'CONCEIVED', 'AND', 'SO', 'DEDICATED', 'CAN', 'LONG',
             'ENDURE'],
            ['WE', 'ARE', 'MET', 'ON', 'A', 'GREAT', 'BATTLEFIELD', 'OF',
             'THAT', 'WAR'],
            ['WE', 'HAVE', 'COME', 'TO', 'DEDICATE', 'A', 'PORTION', 'OF',
             'THAT', 'FIELD', 'AS', 'A', 'FINAL', 'RESTING', 'PLACE', 'FOR',
             'THOSE', 'WHO', 'HERE', 'GAVE', 'THEIR', 'LIVES', 'THAT', 'THAT',
             'NATION', 'MIGHT', 'LIVE'],
            ['IT', 'IS', 'ALTOGETHER', 'FITTING', 'AND', 'PROPER', 'THAT',
             'WE', 'SHOULD', 'DO', 'THIS', 'BUT', 'IN', 'A', 'LARGER', 'SENSE',
             'WE', 'CANNOT', 'DEDICATE', 'WE', 'CANNOT', 'CONSECRATE', 'WE',
             'CANNOT', 'HALLOW', 'THIS', 'GROUND'],
            ['THE', 'BRAVE', 'MEN', 'LIVING', 'AND', 'DEAD', 'WHO',
             'STRUGGLED', 'HERE', 'HAVE', 'CONSECRATED', 'IT', 'FAR', 'ABOVE',
             'OUR', 'POOR', 'POWER', 'TO', 'ADD', 'OR', 'DETRACT'],
            ['THE', 'WORLD', 'WILL', 'LITTLE', 'NOTE', 'NOR', 'LONG',
             'REMEMBER', 'WHAT', 'WE', 'SAY', 'HERE', 'BUT', 'IT', 'CAN',
             'NEVER', 'FORGET', 'WHAT', 'THEY', 'DID', 'HERE'],
            ['IT', 'IS', 'FOR', 'US', 'THE', 'LIVING', 'RATHER', 'TO', 'BE',
             'DEDICATED', 'HERE', 'TO', 'THE', 'UNFINISHED', 'WORK', 'WHICH',
             'THEY', 'WHO', 'FOUGHT', 'HERE', 'HAVE', 'THUS', 'FAR', 'SO',
             'NOBLY', 'ADVANCED'],
            ['IT', 'IS', 'RATHER', 'FOR', 'US', 'TO', 'BE', 'HERE',
             'DEDICATED', 'TO', 'THE', 'GREAT', 'TASK', 'REMAINING', 'BEFORE',
             'US'],
            ['THAT', 'FROM', 'THESE', 'HONORED', 'DEAD', 'WE', 'TAKE',
             'INCREASED', 'DEVOTION', 'TO', 'THAT', 'CAUSE', 'FOR', 'WHICH',
             'THEY', 'GAVE', 'THE', 'LAST', 'FULL', 'MEASURE', 'OF',
             'DEVOTION'],
            ['THAT', 'WE', 'HERE', 'HIGHLY', 'RESOLVE', 'THAT', 'THESE',
             'DEAD', 'SHALL', 'NOT', 'HAVE', 'DIED', 'IN', 'VAIN'],
            ['THAT', 'THIS', 'NATION', 'UNDER', 'GOD', 'SHALL', 'HAVE', 'A',
             'NEW', 'BIRTH', 'OF', 'FREEDOM'],
            ['AND', 'THAT', 'GOVERNMENT', 'OF', 'THE', 'PEOPLE', 'BY', 'THE',
             'PEOPLE', 'FOR', 'THE', 'PEOPLE', 'SHALL', 'NOT', 'PERISH',
             'FROM', 'THE', 'EARTH']
        ]

    @staticmethod
    @pytest.fixture
    def AS_LISTS_ROW_UNIQUES():
        return [
            ['A', 'AGO', 'ALL', 'AND', 'ARE', 'BROUGHT', 'CONCEIVED',
             'CONTINENT', 'CREATED', 'DEDICATED', 'EQUAL', 'FATHERS', 'FORTH',
             'FOUR', 'IN', 'LIBERTY', 'MEN', 'NATION', 'NEW', 'ON', 'OUR',
             'PROPOSITION', 'SCORE', 'SEVEN', 'THAT', 'THE', 'THIS', 'TO',
             'YEARS'],
            ['A', 'AND', 'ANY', 'ARE', 'CAN', 'CIVIL', 'CONCEIVED',
             'DEDICATED', 'ENDURE', 'ENGAGED', 'GREAT', 'IN', 'LONG',
             'NATION', 'NOW', 'OR', 'SO', 'TESTING', 'THAT', 'WAR', 'WE',
             'WHETHER'],
            ['A', 'ARE', 'BATTLEFIELD', 'GREAT', 'MET', 'OF', 'ON', 'THAT',
             'WAR', 'WE'],
            ['A', 'AS', 'COME', 'DEDICATE', 'FIELD', 'FINAL', 'FOR', 'GAVE',
             'HAVE', 'HERE', 'LIVE', 'LIVES', 'MIGHT', 'NATION', 'OF',
             'PLACE', 'PORTION', 'RESTING', 'THAT', 'THEIR', 'THOSE', 'TO',
             'WE', 'WHO'],
            ['A', 'ALTOGETHER', 'AND', 'BUT', 'CANNOT', 'CONSECRATE',
             'DEDICATE', 'DO', 'FITTING', 'GROUND', 'HALLOW', 'IN', 'IS',
             'IT', 'LARGER', 'PROPER', 'SENSE', 'SHOULD', 'THAT', 'THIS',
             'WE'],
            ['ABOVE', 'ADD', 'AND', 'BRAVE', 'CONSECRATED', 'DEAD', 'DETRACT',
             'FAR', 'HAVE', 'HERE', 'IT', 'LIVING', 'MEN', 'OR', 'OUR', 'POOR',
             'POWER', 'STRUGGLED', 'THE', 'TO', 'WHO'],
            ['BUT', 'CAN', 'DID', 'FORGET', 'HERE', 'IT', 'LITTLE', 'LONG',
             'NEVER', 'NOR', 'NOTE', 'REMEMBER', 'SAY', 'THE', 'THEY', 'WE',
             'WHAT', 'WILL', 'WORLD'],
            ['ADVANCED', 'BE', 'DEDICATED', 'FAR', 'FOR', 'FOUGHT', 'HAVE',
             'HERE', 'IS', 'IT', 'LIVING', 'NOBLY', 'RATHER', 'SO', 'THE',
             'THEY', 'THUS', 'TO', 'UNFINISHED', 'US', 'WHICH', 'WHO',
             'WORK'],
            ['BE', 'BEFORE', 'DEDICATED', 'FOR', 'GREAT', 'HERE', 'IS', 'IT',
             'RATHER', 'REMAINING', 'TASK', 'THE', 'TO', 'US'],
            ['CAUSE', 'DEAD', 'DEVOTION', 'FOR', 'FROM', 'FULL', 'GAVE',
             'HONORED', 'INCREASED', 'LAST', 'MEASURE', 'OF', 'TAKE', 'THAT',
             'THE', 'THESE', 'THEY', 'TO', 'WE', 'WHICH'],
            ['DEAD', 'DIED', 'HAVE', 'HERE', 'HIGHLY', 'IN', 'NOT', 'RESOLVE',
             'SHALL', 'THAT', 'THESE', 'VAIN', 'WE'],
            ['A', 'BIRTH', 'FREEDOM', 'GOD', 'HAVE', 'NATION', 'NEW', 'OF',
             'SHALL', 'THAT', 'THIS', 'UNDER'],
            ['AND', 'BY', 'EARTH', 'FOR', 'FROM', 'GOVERNMENT', 'NOT', 'OF',
             'PEOPLE', 'PERISH', 'SHALL', 'THAT', 'THE']
         ]

    @staticmethod
    @pytest.fixture
    def AS_LISTS_ROW_UNIQUES_COUNTS():
        return [
            [1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
             1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1,
             1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1,
             1, 1, 1],
            [1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 4],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1,
             1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2],
            [1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 4]
        ]

    @staticmethod
    @pytest.fixture
    def AS_LISTS_ALL_UNIQUES():
        return [
            'A', 'ABOVE', 'ADD', 'ADVANCED', 'AGO', 'ALL', 'ALTOGETHER', 'AND',
            'ANY', 'ARE', 'AS', 'BATTLEFIELD', 'BE', 'BEFORE', 'BIRTH', 'BRAVE',
            'BROUGHT', 'BUT', 'BY', 'CAN', 'CANNOT', 'CAUSE', 'CIVIL', 'COME',
            'CONCEIVED', 'CONSECRATE', 'CONSECRATED', 'CONTINENT', 'CREATED',
            'DEAD', 'DEDICATE', 'DEDICATED', 'DETRACT', 'DEVOTION', 'DID', 'DIED',
            'DO', 'EARTH', 'ENDURE', 'ENGAGED', 'EQUAL', 'FAR', 'FATHERS', 'FIELD',
            'FINAL', 'FITTING', 'FOR', 'FORGET', 'FORTH', 'FOUGHT', 'FOUR',
            'FREEDOM', 'FROM', 'FULL', 'GAVE', 'GOD', 'GOVERNMENT', 'GREAT',
            'GROUND','HALLOW', 'HAVE', 'HERE', 'HIGHLY', 'HONORED', 'IN',
            'INCREASED', 'IS', 'IT', 'LARGER', 'LAST', 'LIBERTY', 'LITTLE',
            'LIVE', 'LIVES', 'LIVING', 'LONG', 'MEASURE', 'MEN', 'MET', 'MIGHT',
            'NATION', 'NEVER', 'NEW', 'NOBLY', 'NOR', 'NOT', 'NOTE', 'NOW', 'OF',
            'ON', 'OR', 'OUR', 'PEOPLE', 'PERISH', 'PLACE', 'POOR', 'PORTION',
            'POWER', 'PROPER', 'PROPOSITION', 'RATHER', 'REMAINING', 'REMEMBER',
            'RESOLVE', 'RESTING', 'SAY', 'SCORE', 'SENSE', 'SEVEN', 'SHALL',
            'SHOULD', 'SO', 'STRUGGLED', 'TAKE', 'TASK', 'TESTING', 'THAT', 'THE',
            'THEIR', 'THESE', 'THEY', 'THIS', 'THOSE', 'THUS', 'TO', 'UNDER',
            'UNFINISHED', 'US', 'VAIN', 'WAR', 'WE', 'WHAT', 'WHETHER', 'WHICH',
            'WHO', 'WILL', 'WORK', 'WORLD', 'YEARS'
        ]


    @staticmethod
    @pytest.fixture
    def AS_LISTS_STOPS_REMOVED():
        return [
            ['FOUR', 'SCORE', 'SEVEN', 'YEARS', 'AGO', 'FATHERS', 'BROUGHT',
            'FORTH', 'CONTINENT', 'NEW', 'NATION', 'CONCEIVED', 'LIBERTY',
            'DEDICATED', 'PROPOSITION', 'ALL', 'MEN', 'CREATED', 'EQUAL'],
            ['NOW', 'ENGAGED', 'GREAT', 'CIVIL', 'WAR', 'TESTING', 'WHETHER',
            'NATION', 'ANY', 'NATION', 'CONCEIVED', 'DEDICATED', 'CAN', 'LONG',
            'ENDURE'],
            ['MET', 'GREAT', 'BATTLEFIELD', 'WAR'],
            ['HAVE', 'COME', 'DEDICATE', 'PORTION', 'FIELD', 'FINAL', 'RESTING',
            'PLACE', 'THOSE', 'GAVE', 'LIVES', 'NATION', 'MIGHT', 'LIVE'],
            ['ALTOGETHER', 'FITTING', 'PROPER', 'SHOULD', 'LARGER', 'SENSE',
            'CANNOT', 'DEDICATE', 'CANNOT', 'CONSECRATE', 'CANNOT', 'HALLOW',
            'GROUND'],
            ['BRAVE', 'MEN', 'LIVING', 'DEAD', 'STRUGGLED', 'HAVE', 'CONSECRATED',
            'FAR', 'ABOVE', 'POOR', 'POWER', 'ADD', 'DETRACT'],
            ['WORLD', 'LITTLE', 'NOTE', 'NOR', 'LONG', 'REMEMBER', 'SAY', 'CAN',
            'NEVER', 'FORGET', 'DID'],
            ['LIVING', 'RATHER', 'DEDICATED', 'UNFINISHED', 'WORK', 'WHICH',
            'FOUGHT', 'HAVE', 'THUS', 'FAR', 'NOBLY', 'ADVANCED'],
            ['RATHER', 'DEDICATED', 'GREAT', 'TASK', 'REMAINING', 'BEFORE'],
            ['THESE', 'HONORED', 'DEAD', 'TAKE', 'INCREASED', 'DEVOTION',
            'CAUSE', 'WHICH', 'GAVE', 'LAST', 'FULL', 'MEASURE', 'DEVOTION'],
            ['HIGHLY', 'RESOLVE', 'THESE', 'DEAD', 'SHALL', 'HAVE', 'DIED',
            'VAIN'],
            ['NATION', 'UNDER', 'GOD', 'SHALL', 'HAVE', 'NEW', 'BIRTH',
            'FREEDOM'],
            ['GOVERNMENT', 'PEOPLE', 'PEOPLE', 'PEOPLE', 'SHALL', 'PERISH',
            'EARTH']
        ]

    @staticmethod
    @pytest.fixture
    def AS_LISTS_ALL_UNIQUES_COUNTS():
        return [
        7, 1, 1, 1, 1, 1, 1, 6, 1, 3, 1, 1, 2, 1, 1, 1, 1, 2, 1, 2, 3, 1, 1,
        1, 2, 1, 1, 1, 1, 3, 2, 4, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1,
        5, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 3, 1, 1, 5, 8, 1, 1, 4, 1, 3, 5, 1,
        1, 1, 1, 1, 1, 2, 2, 1, 2, 1, 1, 5, 1, 2, 1, 1, 2, 1, 1, 5, 2, 2, 2,
        3, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 3, 1, 1, 1,
        1, 13, 11, 1, 2,3, 4, 1, 1, 8, 1, 1, 3, 1, 2, 10, 2, 1, 2, 3, 1, 1,
        1, 1
       ]


    # delete_empty_rows    Remove textless rows from data.
    # remove_characters    Keep only allowed or removed disallowed characters
    #                       from entire CLEANED_TEXT object.
    # _strip               Remove multiple spaces and leading and trailing
    #                       spaces from all text in CLEAND_TEXT object.
    # normalize            Set all text in CLEANED_TEXT object to upper case
    #                       (default) or lower case.
    # lex_lookup           Scan entire CLEANED_TEXT object and prompt for
    #                       handling of words not in LEXICON.  <<<<<
    # return_row_uniques   Return a potentially ragged vector containing the
    #                       unique words for each row in CLEANED_TEXT object.
    # return_overall_uniques Return unique words in the entire CLEANED_TEXT
    #                       object.
    # remove_stops         Remove stop words from the entire CLEANED_TEXT
    #                       object.
    # dump_to_csv          Dump CLEANED_TEXT object to csv.
    # dump_to_txt          Dump CLEANED_TEXT object to txt.


    TestClass = tc.TextCleaner(WORDS, update_lexicon=False)
    TestClass.as_list_of_lists()


    def test_delete_empty_rows(self, AS_LISTS_EMPTIES_DELETED):

        self.TestClass.delete_empty_rows()
        # list(list()) INSTEAD OF NP TO GET TO PASS THIS TEST
        assert np.array_equiv(self.TestClass.CLEANED_TEXT,
                               AS_LISTS_EMPTIES_DELETED)


    def test_remove_characters(self, AS_LISTS_CHARS_REMOVED):

        self.TestClass.remove_characters()
        # list(list()) INSTEAD OF NP TO GET TO PASS THIS TEST
        assert np.array_equiv(list(map(list, self.TestClass.CLEANED_TEXT)),
                          AS_LISTS_CHARS_REMOVED)


    def test_strip(self, AS_LISTS_STRIPPED):

        self.TestClass._strip()
        # list(list()) INSTEAD OF NP TO GET TO PASS THIS TEST
        assert np.array_equiv(list(map(list, self.TestClass.CLEANED_TEXT)),
                          AS_LISTS_STRIPPED)


    def test_normalize(self, AS_LISTS_NORMALIZED):

        self.TestClass.normalize()
        # list(list()) INSTEAD OF NP TO GET TO PASS THIS TEST
        assert np.array_equiv(list(map(list, self.TestClass.CLEANED_TEXT)),
                          AS_LISTS_NORMALIZED)



    # THIS IS INTERACTIVE
    def test_lex_lookup(self, AS_LISTS_LOOKEDUP):
        user_inputs = "e\nSEVEN\nY\nY\nY\ne\nDEVOTION\nY\ne\nGOVERNMENT\nY\n"
        with patch('sys.stdin', io.StringIO(user_inputs)):
            self.TestClass.lex_lookup()
            # list(list()) INSTEAD OF NP TO GET TO PASS THIS TEST
            assert np.array_equiv(list(map(list, self.TestClass.CLEANED_TEXT)),
                              AS_LISTS_LOOKEDUP)


    def test_return_row_uniques(self, AS_LISTS_ROW_UNIQUES, AS_LISTS_ROW_UNIQUES_COUNTS):

        # COMPARE ROW UNIQUES ROW BY ROW
        for idx, ROW in enumerate(self.TestClass.return_row_uniques()):
            assert np.array_equiv(ROW, AS_LISTS_ROW_UNIQUES[idx])


        ROW_UNIQUES = self.TestClass.return_row_uniques(return_counts=False)
        # CONVERT return_row_uniques() TO list(list()) FOR COMPARISON,
        # UNEXPECTEDLY EXCEPTING WHEN NUMPY
        assert np.array_equiv(list(map(list, ROW_UNIQUES)), AS_LISTS_ROW_UNIQUES)

        del ROW_UNIQUES


        ROW_UNIQUES, ROW_COUNTS = \
            self.TestClass.return_row_uniques(return_counts=True)
        # CONVERT return_row_uniques() TO list(list()) FOR COMPARISON,
        # UNEXPECTEDLY EXCEPTING WHEN NUMPY
        assert np.array_equiv(list(map(list, ROW_UNIQUES)), AS_LISTS_ROW_UNIQUES)
        assert np.array_equiv(list(map(list, ROW_COUNTS)), AS_LISTS_ROW_UNIQUES_COUNTS)


        del ROW_UNIQUES, ROW_COUNTS


    def test_return_overall_uniques(self, AS_LISTS_ALL_UNIQUES, AS_LISTS_ALL_UNIQUES_COUNTS):

        ALL_UNIQUES = self.TestClass.return_overall_uniques(return_counts=False)
        assert np.array_equiv(ALL_UNIQUES, AS_LISTS_ALL_UNIQUES)

        del ALL_UNIQUES


        ALL_UNIQUES, ALL_COUNTS = \
            self.TestClass.return_overall_uniques(return_counts=True)
        assert np.array_equiv(ALL_UNIQUES, AS_LISTS_ALL_UNIQUES)
        assert np.array_equiv(ALL_COUNTS, AS_LISTS_ALL_UNIQUES_COUNTS)

        del ALL_UNIQUES, ALL_COUNTS


    def test_remove_stops(self, AS_LISTS_STOPS_REMOVED):

        self.TestClass.remove_stops()
        STOPS_REMOVED = self.TestClass.CLEANED_TEXT
        # CONVERT return_row_uniques() TO list(list()) FOR COMPARISON,
        # UNEXPECTEDLY EXCEPTING WHEN NUMPY
        assert np.array_equiv(list(map(list, STOPS_REMOVED)), AS_LISTS_STOPS_REMOVED)

        del STOPS_REMOVED


    # this is interactive
    def test_dump_to_csv(self):

        # dump_to_csv               Dump CLEANED_TEXT object to csv.
        user_inputs = "dump_to_csv_test_dump\nY\n"
        with patch('sys.stdin', io.StringIO(user_inputs)):
            self.TestClass.dump_to_csv()

    # this is interactive
    def test_dump_to_txt(self):

        # dump_to_txt               Dump CLEANED_TEXT object to txt.
        user_inputs = "dump_to_txt_test_dump\nY\n"
        with patch('sys.stdin', io.StringIO(user_inputs)):
            self.TestClass.dump_to_txt()



# END FOR DATA RETURNED AS LIST OF LISTS ###############################
########################################################################
########################################################################














