# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.feature_extraction.text._TextLookup.TextLookup import TextLookup as TL





if __name__ == '__main__':

    trfm = TL(
        update_lexicon = True,
        skip_numbers = True,
        auto_split = True,
        auto_add_to_lexicon = False,
        auto_delete = False,
        verbose = False
    )


    X = [
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

    trfm.fit(X)

    out = trfm.transform(X)


    [print(_) for _ in out]






