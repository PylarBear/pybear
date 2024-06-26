from pybear.feature_extraction.text._TextCleaner import TextCleaner as tc


TEST_TEXT = [
    "It was the best of times, "
    "it was the worst of times, "
    "it was the age of wisdom, "
    "it was the age of foolishness, "
    "it was the epoch of belief, "
    "it was the epoch of incredulity, "
    "it was the season of light, "
    "it was the season of darkness, "
    "it was the spring of hope, "
    "it was the winter of despair."
]

TestClass = tc.TextCleaner(TEST_TEXT)

TestClass.remove_stops()

[print(f'***{_}***') for _ in TestClass.CLEANED_TEXT]







