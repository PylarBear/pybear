# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#

import pytest

import numpy as np

from model_selection.GSTCV._GSTCVMixin._validation._threshold_checker import \
    _threshold_checker



class TestThresholdChecker:

# def _threshold_checker(
#         __thresholds: Union[None, Iterable[Union[int, float]], Union[int, float]],
#         is_from_kwargs: bool,
#         idx: int
#     ) -> npt.NDArray[np.float64]:


    @staticmethod
    @pytest.fixture
    def good_threshes():
        return np.linspace(0,1,21)


    # 'is_from_kwargs' ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
    @pytest.mark.parametrize('junk_ifk',
        (0, 1, 3.14, None, min, 'junk', [0,1], {'a':1}, lambda x: x)
    )
    def test_rejects_non_bool(self, good_threshes, junk_ifk):

        with pytest.raises(TypeError):
            _threshold_checker(good_threshes, junk_ifk, 0)


    @pytest.mark.parametrize('good_ifk', (True, False))
    def test_accepts_bool(self, good_threshes, good_ifk):
        # 'is_from_kwargs'
        out = _threshold_checker(good_threshes, good_ifk, 0)
        assert np.array_equiv(out, good_threshes)

    # END 'is_from_kwargs' ** * ** * ** * ** * ** * ** * ** * ** * ** *

    # 'idx' ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** *
    @pytest.mark.parametrize('junk_idx',
        (-1, True, False, 3.14, None, min, 'junk', [0,1], {'a':1}, lambda x: x)
    )
    def test_idx_rejects_junk(self, good_threshes, junk_idx):
        # 'is_from_kwargs'
        with pytest.raises(TypeError):
            _threshold_checker(good_threshes, True, junk_idx)


    @pytest.mark.parametrize('good_idx', (0, 1, 100))
    def test_idx_accepts_int(self, good_threshes, good_idx):
        out = _threshold_checker(good_threshes, True, good_idx)
        assert np.array_equiv(out, good_threshes)

    # END 'idx' ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * ** * *


    @pytest.mark.parametrize('_ifk', (True, False))
    @pytest.mark.parametrize('_idx', (0, 5, 10))
    @pytest.mark.parametrize('junk_thresh',
        (True, False, 'trash', min, {'a':1}, lambda x: x)
    )
    def test_rejects_non_num_none_iter(self, junk_thresh, _ifk, _idx):
        with pytest.raises(TypeError):
            _threshold_checker(junk_thresh, _ifk, _idx)


    @pytest.mark.parametrize('_ifk', (True, False))
    @pytest.mark.parametrize('_idx', (0, 5, 10))
    def test_none_returns_linspace(self, good_threshes, _ifk, _idx):
        assert np.array_equiv(
            _threshold_checker(None, _ifk, _idx),
            good_threshes
        )


    @pytest.mark.parametrize('_ifk', (True, False))
    @pytest.mark.parametrize('_idx', (0, 5, 10))
    @pytest.mark.parametrize('bad_thresh', (-1, 2, 10))
    def test_rejects_thresh_out_of_range(self, bad_thresh, _ifk, _idx):
        with pytest.raises(ValueError):
            _threshold_checker(bad_thresh, _ifk, _idx)


    @pytest.mark.parametrize('_ifk', (True, False))
    @pytest.mark.parametrize('_idx', (0, 5, 10))
    @pytest.mark.parametrize('good_thresh',
        (0, 0.5, 1, [0,0.5,1], (0.25, 0.5, 0.75), {0.7, 0.8, 0.9})
    )
    def test_accepts_good_thresh(self, good_thresh, _ifk, _idx):

        try:
            good_thresh = list(good_thresh)
        except:
            good_thresh = [good_thresh]

        assert np.array_equiv(
            _threshold_checker(good_thresh, _ifk, _idx),
            np.array(list(good_thresh))
        )


    @pytest.mark.parametrize('_ifk', (True, False))
    @pytest.mark.parametrize('_idx', (0, 5, 10))
    def test_junk_in_list(self, _ifk, _idx):

        with pytest.raises(TypeError):
            _threshold_checker(['a','b','c'], _ifk, _idx)

        with pytest.raises(ValueError):
            _threshold_checker([-3.14, 3.14], _ifk, _idx)











































