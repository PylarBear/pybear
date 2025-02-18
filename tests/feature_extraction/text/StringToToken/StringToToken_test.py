# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

from pybear.feature_extraction.text._StringToToken.StringToToken \
    import StringToToken



class TestStringToToken:

    @staticmethod
    @pytest.fixture
    def good_X():
        return [
            'A long time ago, in a galaxy far, far away.',
            'It is a period of civil war.',
            'Rebel spaceships, striking from a hidden base, have won their '
            'first victory against the evil Galactic Empire.'
        ]


    #     def __init__(
    #             self,
    #             *,
    #             sep: Union[None, str]=None,
    #             maxsplit: int=-1
    #         ) -> None:

    @pytest.mark.parametrize('junk_kwarg',
        (0, 1, True, 3.14, min, [0,1], (0,1), {0,1}, {'a':1}, lambda x: x)
    )
    def test_sep_rejects_junk(self, junk_kwarg, good_X):
        with pytest.raises(TypeError):
            StringToToken(sep=junk_kwarg).transform(good_X)


    @pytest.mark.parametrize('junk_kwarg',
        (True, None, 'junk', 3.14, min, [0,1], (0,1), {0,1}, {'a':1}, lambda x: x)
    )
    def test_maxsplit_rejects_junk(self, junk_kwarg, good_X):
        with pytest.raises(TypeError):
            StringToToken(maxsplit=junk_kwarg).transform(good_X)







    # takes ParallelPostFit
    # _wrapped_by_dask_ml senses ParallelPostFit
    # when wrapped, internally returns np.array
    # when not wrapped, takes Iterable[str] or Iterable[Iterable[str]] and
    #       returns list[list[str]]]
















