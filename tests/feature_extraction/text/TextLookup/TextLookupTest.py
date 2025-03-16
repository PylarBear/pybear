# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest
from unittest.mock import patch

import io

import numpy as np

from pybear.feature_extraction.text._TextLookup.TextLookup import TextLookup




class TestTextLookup:



    @staticmethod
    @pytest.fixture(scope='module')
    def _kwargs():
        return {
            'update_lexicon': True,
            'skip_numbers': True,
            'auto_split': True,
            'auto_add_to_lexicon': False,
            'auto_delete': False,
            'DELETE_ALWAYS': None,
            'REPLACE_ALWAYS': None,
            'SKIP_ALWAYS': None,
            'SPLIT_ALWAYS': None,
            'verbose': False
        }


    @staticmethod
    @pytest.fixture(scope='module')
    def _X():
        return [
            ["ACTIVITY", "APPRECIATE", "ANIMAL", "ANTICIPATE", "BEAUTIFUL"],
            ["BENEATH", "BENEFIT", "BRINGING", "BRIGHT", "CAREFUL"],
            ["CARING", "CATCHING", "TEACOMPOST", "CELEBRATE", "CIRCUMSTANCE"],
            ["COMMON", "CREATIVITY", "CURIOUS", "DANGER", "FLOOBASTIC"],
            ["DESTINY", "DESIRE", "DIVINE", "DREAMING", "EDUCATE"],
            ["ELITE", "ENCOURAGE", "EXCITEMENT", "EXPECT", "FAITHFUL"],
            ["FANTASTIC", "FAVORITE", "FRIEND", "FRIENDLY", "QUACKTIVATE"],
            ["GATHERING", "GENEROUS", "GENERATE", "GLORIOUS", "HARMONY"],
            ["HELPFUL", "HOPEFUL", "HONESTY", "HUMANITY", "INFLUENCE"],
            ["INSIGHT", "INTEREST", "INFLUENCER", "JOYFUL", "JUDGEMENT"],
            ["KINDNESS", "KNOWLEDGE", "LEADER", "LEARNING", "LIBERATE"],
            ["LIFE", "LIGHT", "SMORFIC", "MAGNIFICIENT", "MEANING"],
            ["MEMORIES", "MIND", "MOTIVATION", "NATIONAL", "NATURE"],
            ["OPTIMISTIC", "ORDERLY", "OPPORTUNITY", "PATIENCE", "PASSION"],
            ["PEACEFUL", "PERFECT", "PERSISTENT", "PLEASURE", "POSITIVE"],
            ["POWERFUL", "PROGRESS", "PURPOSE", "QUALITY", "QUEST"],
            ["REACHING", "REALITY", "RESPECTFUL", "SINCERE", "SKILLFUL"],
            ["SPIRITUAL", "STRATEGY", "SUCCESS", "SUPPORT", "TALENT"],
            ["THOUGHTFUL", "TREMENDOUS", "UNITY", "USEFUL", "VISION"],
            ["WEALTH", "WISDOM", "WORTHY", "ZENITH", "ZESTFUL"],
            ["ABUNDANT", "ADVENTURE", "AMBITION", "ANCIENT", "ARTIST"],
            ["AWAKEN", "BELIEVE", "BLESSING", "CALM", "CAREER"],
            ["CHALLENGE", "CHARACTER", "CLARITY", "COMMIT", "COURAGE"],
            ["CREATIVE", "CURRENT", "DELIGHT", "DESTROY", "JUMBLYWUMP"],
            ["DREAMER", "ELATION", "EMPATHY", "ENERGY", "ENDEAVOR"],
            ["ENGAGE", "ENLIGHTEN", "EXPLORER", "FOCUS", "FOREVER"],
            ["FRIENDS", "GAIN", "GREATNESS", "HEROIC", "HOPE"],
            ["HORIZON", "IDEAL", "IGNITE", "INSPIRE", "JOY"],
            ["JOURNEY", "JUSTICE", "LEGACY", "LIFELESS", "LOVEABLE"],
            ["MASTER", "MYSTIC", "NOBLE", "OBSERVE", "PEACE"],
            ["PERSIST", "PLEASANT", "PROSPER", "REFLECT", "RELIABLE"],
            ["REMARKABLE", "RESOURCEFUL", "RESTORE", "SHARE", "SIMPLIFY"],
            ["SKILLED", "SOAR", "STRENGTH", "SUBLIME", "TRIUMPH"],
            ["UNITY", "VISIONARY", "WEALTHY", "WISDOM", "YOUTHFUL"],
            ["AMAZIN", "BEAUTIFULL", "CREATING", "DILIGENCE", "BLOOMTRIX"],
            ["EXPECTATION", "EXCITING", "FLEXABILITY", "FREEDOM", "GLOURY"],
            ["HARMONIOUS", "HEROISM", "INSPIRATION", "MINDFUL", "ZIGTROPE"],
            ["PERSISTACE", "PROGRESSIVE", "TRULY", "VALUEABLE", "VICTORY"],
            ["FLAPDOO", "TORTAGLOOM", "STARDUSK", "GLENSHWINK", "ZONKING"],
            ["SNORLUX", "CRUMBLEWAX", "TORTAGLOOM", "GLIMPLER", "SNIRKIFY"]
        ]


    @staticmethod
    @pytest.fixture(scope='module')
    def exp():
        return [
            ['ACTIVITY', 'APPRECIATE', 'ANIMAL', 'ANTICIPATE', 'BEAUTIFUL'],
            ['BENEATH', 'BENEFIT', 'BRINGING', 'BRIGHT', 'CAREFUL'],
            ['CARING', 'CATCHING', 'TEA', 'COMPOST', 'CELEBRATE', 'CIRCUMSTANCE'],
            ['COMMON', 'CREATIVITY', 'CURIOUS', 'DANGER'],
            ['DESTINY', 'DESIRE', 'DIVINE', 'DREAMING', 'EDUCATE'],
            ['ELITE', 'ENCOURAGE', 'EXCITEMENT', 'EXPECT', 'FAITHFUL'],
            ['FANTASTIC', 'FAVORITE', 'FRIEND', 'FRIENDLY'],
            ['GATHERING', 'GENEROUS', 'GENERATE', 'GLORIOUS', 'HARMONY'],
            ['HELPFUL', 'HOPEFUL', 'HONESTY', 'HUMANITY', 'INFLUENCE'],
            ['INSIGHT', 'INTEREST', 'INFLUENCER', 'JOYFUL', 'JUDGEMENT'],
            ['KINDNESS', 'KNOWLEDGE', 'LEADER', 'LEARNING', 'LIBERATE'],
            ['LIFE', 'LIGHT', 'MAGNIFICENT', 'MEANING'],
            ['MEMORIES', 'MIND', 'MOTIVATION', 'NATIONAL', 'NATURE'],
            ['OPTIMISTIC', 'ORDERLY', 'OPPORTUNITY', 'PATIENCE', 'PASSION'],
            ['PEACEFUL', 'PERFECT', 'PERSISTENT', 'PLEASURE', 'POSITIVE'],
            ['POWERFUL', 'PROGRESS', 'PURPOSE', 'QUALITY', 'QUEST'],
            ['REACHING', 'REALITY', 'RESPECTFUL', 'SINCERE', 'SKILLFUL'],
            ['SPIRITUAL', 'STRATEGY', 'SUCCESS', 'SUPPORT', 'TALENT'],
            ['THOUGHTFUL', 'TREMENDOUS', 'UNITY', 'USEFUL', 'VISION'],
            ['WEALTH', 'WISDOM', 'WORTHY', 'ZENITH', 'ZESTFUL'],
            ['ABUNDANT', 'ADVENTURE', 'AMBITION', 'ANCIENT', 'ARTIST'],
            ['AWAKEN', 'BELIEVE', 'BLESSING', 'CALM', 'CAREER'],
            ['CHALLENGE', 'CHARACTER', 'CLARITY', 'COMMIT', 'COURAGE'],
            ['CREATIVE', 'CURRENT', 'DELIGHT', 'DESTROY'],
            ['DREAMER', 'ELATION', 'EMPATHY', 'ENERGY', 'ENDEAVOR'],
            ['ENGAGE', 'ENLIGHTEN', 'EXPLORER', 'FOCUS', 'FOREVER'],
            ['FRIENDS', 'GAIN', 'GREATNESS', 'HEROIC', 'HOPE'],
            ['HORIZON', 'IDEAL', 'IGNITE', 'INSPIRE', 'JOY'],
            ['JOURNEY', 'JUSTICE', 'LEGACY', 'LIFELESS', 'LOVEABLE'],
            ['MASTER', 'MYSTIC', 'NOBLE', 'OBSERVE', 'PEACE'],
            ['PERSIST', 'PLEASANT', 'PROSPER', 'REFLECT', 'RELIABLE'],
            ['REMARKABLE', 'RESOURCEFUL', 'RESTORE', 'SHARE', 'SIMPLIFY'],
            ['SKILLED', 'SOAR', 'STRENGTH', 'SUBLIME', 'TRIUMPH'],
            ['UNITY', 'VISIONARY', 'WEALTHY', 'WISDOM', 'YOUTHFUL'],
            ['AMAZING', 'BEAUTIFUL', 'CREATING', 'DILIGENCE', 'BLOOM', 'TRIX'],
            ['EXPECTATION', 'EXCITING', 'FLEXIBILITY', 'FREEDOM', 'GLORY'],
            ['HARMONIOUS', 'HEROISM', 'INSPIRATION', 'MINDFUL', 'ZIG', 'TROPE'],
            ['PERSISTENCE', 'PROGRESSIVE', 'TRULY', 'VALUEABLE', 'VICTORY'],
            ['STAR', 'DUSK'],
            ['CRUMBLE', 'WAX'],
    ]



    def test_accuracy(self, _kwargs, _X, exp):

        TestCls = TextLookup(**_kwargs)

        a = f"d\nl\ne\nMAGNIFICENT\ny\nd\nl\nu\n2\nBLOOM\ny\nTRIX\n"
        b = f"y\ny\na\nf\nBEAUTIFUL\ny\ne\nAMAZING\ny\nf\nGLORY\ny\n"
        c = f"n\nf\nFLEXIBILTY\nn\nFLEXIBILITY\ny\ny\nn\nw\nn\nf\n"
        d = f"PERSISTENCE\ny\nl\nl\ny\nl\nd\nd\nd\nc\ny\nl\nc\n"

        user_inputs = a + b + c + d
        with patch('sys.stdin', io.StringIO(user_inputs)):
            out = TestCls.transform(_X)


        assert all(map(np.array_equal, out, exp))


        assert np.array_equal(
            TestCls.LEXICON_ADDENDUM,
            ['TRIX']
        )

        assert TestCls.KNOWN_WORDS[-1] == 'TRIX'

        assert np.array_equal(
            list(TestCls.SPLIT_ALWAYS.keys()),
            ['BLOOMTRIX']
        )

        assert np.array_equal(
            list(TestCls.SPLIT_ALWAYS.values()),
            [['BLOOM', 'TRIX']]
        )

        assert np.array_equal(
            TestCls.DELETE_ALWAYS,
            ['QUACKTIVATE', 'JUMBLYWUMP', 'ZONKING',
             'GLENSHWINK', 'TORTAGLOOM', 'SNORLUX']
        )

        assert np.array_equal(
            list(TestCls.REPLACE_ALWAYS.keys()),
            ['BEAUTIFULL', 'GLOURY', 'FLEXABILITY', 'PERSISTACE']
        )

        assert np.array_equal(
            list(TestCls.REPLACE_ALWAYS.values()),
            ['BEAUTIFUL', 'GLORY', 'FLEXIBILITY', 'PERSISTENCE']
        )

        assert np.array_equal(
            TestCls.SKIP_ALWAYS,
            ['VALUEABLE']
        )









