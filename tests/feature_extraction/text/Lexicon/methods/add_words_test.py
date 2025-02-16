# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

from pybear.feature_extraction.text._Lexicon._methods._add_words import _add_words





class TestAddWords:

    # def _add_words(
    #     WORDS: Union[str, Sequence[str]],
    #     lexicon_folder_path: str,
    #     character_validation: Optional[bool] = True,
    #     majuscule_validation: Optional[bool] = True
    # ) -> None:



    # validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *

    @pytest.mark.parametrize('junk_WORDS',
        (-2.7, -1, 0, 1, 2.7, True, None, [0, 1], (1,), {1,2}, {'a':1}, lambda x: x)
    )
    def test_rejects_junk_WORDS(self, junk_WORDS):

        with pytest.raises(TypeError):

            _add_words(
                junk_WORDS,
                lexicon_folder_path='sam i am',
                character_validation=False,
                majuscule_validation=False
            )


    @pytest.mark.parametrize('junk_path',
        (-2.7, -1, 0, 1, 2.7, True, None, [0, 1], (1,), {1,2}, {'a':1}, lambda x: x)
    )
    def test_rejects_junk_lexicon_folder_path(self, junk_path):

        with pytest.raises(TypeError):

            _add_words(
                'ULTRACREPIDARIAN',
                lexicon_folder_path=junk_path,
                character_validation=False,
                majuscule_validation=False
            )


    @pytest.mark.parametrize('junk_cv',
        (-2.7, -1, 0, 1, 2.7, None, [0, 1], (1,), {1,2}, {'a':1}, lambda x: x)
    )
    def test_rejects_junk_character_validation(self, junk_cv):

        with pytest.raises(TypeError):

            _add_words(
                'CREPUSCULAR',
                lexicon_folder_path='/somewhere/out/there',
                character_validation=junk_cv,
                majuscule_validation=False
            )

    @pytest.mark.parametrize('junk_mv',
        (-2.7, -1, 0, 1, 2.7, None, [0, 1], (1,), {1, 2}, {'a': 1}, lambda x: x)
    )
    def test_rejects_junk_majuscule_validation(self, junk_mv):
        with pytest.raises(TypeError):
            _add_words(
                'PETRICHOR',
                lexicon_folder_path='/somewhere/out/there',
                character_validation=False,
                majuscule_validation=junk_mv
            )

    # END validation ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *


    """
    file_base = f'lexicon_'

    file_identifiers: list[str] = _identify_sublexicon(WORDS)

    for file_letter in file_identifiers:

        full_path = os.path.join(
            lexicon_folder_path,
            file_base + file_letter.lower() + '.txt'
        )

        with open(full_path, 'r') as f:
            raw_text = np.fromiter(f, dtype='<U40')

        OLD_SUB_LEXICON = np.char.replace(raw_text, f'\n', f' ')
        del raw_text

        PERTINENT_WORDS = [w for w in WORDS if w[0].lower() == file_letter]

        NEW_LEXICON = np.hstack((OLD_SUB_LEXICON, PERTINENT_WORDS))

        # MUST USE uniques TO TAKE OUT ANY NEW WORDS ALREADY IN LEXICON (AND SORT)
        NEW_LEXICON = np.unique(NEW_LEXICON)

        with open(full_path, 'w') as f:
            for line in NEW_LEXICON:
                f.write(line + f'\n')
            f.close()

    print(f'\n*** Lexicon update successful. ***\n')
    
    """





