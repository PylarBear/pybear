# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

from pybear.feature_extraction.text._TextJoiner.TextJoiner \
    import TextJoiner



class TestTextJoiner:


    # def __init__(
    #         self,
    #         *,
    #         sep: Union[None, str]=None,
    #     ) -> None:


    @staticmethod
    @pytest.fixture
    def good_X():
        return [
            'A long time ago, in a galaxy far, far away.',
            'It is a period of civil war.',
            'Rebel spaceships, striking from a hidden base, have won their '
            'first victory against the evil Galactic Empire.'
        ]

























