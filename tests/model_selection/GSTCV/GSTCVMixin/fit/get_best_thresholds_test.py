# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



import pytest

import numpy as np

from pybear.model_selection.GSTCV._GSTCVMixin._fit._get_best_thresholds import \
    _get_best_thresholds





class TestGetBestThresholds:

    # def _get_best_thresholds(
    #         _TEST_FOLD_x_THRESHOLD_x_SCORER__SCORE: MaskedHolderType,
    #         _THRESHOLDS: npt.NDArray[Union[float, int]],
    # ) -> NDArrayHolderType:

    # 5 splits
    # 11 thresholds
    # 4 scorers

    @staticmethod
    @pytest.fixture(scope='class')
    def _TEST_FOLD_x_THRESHOLD_x_SCORER__SCORE():

        return np.ma.masked_array(
            np.random.uniform(0, 1, (5, 11, 4)),
            dtype=np.float64
        )


    @staticmethod
    @pytest.fixture(scope='class')
    def _TEST_FOLD_x_THRESHOLD_x_SCORER__SCORE_partially_masked():
        data = np.ma.masked_array(
            np.random.uniform(0, 1, (5, 11, 4)),
            dtype=np.float64
        )
        for _fold in range(5):
            data[[1,3], :, :] = np.ma.masked
        return data


    @staticmethod
    @pytest.fixture(scope='class')
    def _TEST_FOLD_x_THRESHOLD_x_SCORER__SCORE_fully_masked():
        data = np.ma.masked_array(
            np.random.uniform(0, 1, (5, 11, 4)),
            dtype=np.float64
        )
        for _fold in range(5):
            data[:, :, :] = np.ma.masked
        return data



    @staticmethod
    @pytest.fixture(scope='class')
    def _THRESHOLDS():

        return np.linspace(0, 1, 11, dtype=np.float64)






    @pytest.mark.parametrize('junk_holder',
        (-1, 3.14, True, min, 'junk', [0,1], (0,1), {0,1}, {'a':1}, lambda x: x)
    )
    def test_rejects_junk_holder(self, junk_holder,
        _TEST_FOLD_x_THRESHOLD_x_SCORER__SCORE, _THRESHOLDS):

        with pytest.raises(Exception):
            _get_best_thresholds(
                junk_holder,
                _THRESHOLDS
            )


    def test_accuracy_not_masked(self,
        _TEST_FOLD_x_THRESHOLD_x_SCORER__SCORE, _THRESHOLDS
    ):

        out = _get_best_thresholds(
            _TEST_FOLD_x_THRESHOLD_x_SCORER__SCORE,
            _THRESHOLDS
        )


        assert isinstance(out, np.ma.masked_array)
        assert out.dtype == np.uint16
        assert len(out) == _TEST_FOLD_x_THRESHOLD_x_SCORER__SCORE.shape[2]


        _MEAN_THRESH_x_SCORER = \
            np.mean(_TEST_FOLD_x_THRESHOLD_x_SCORER__SCORE, axis=0)

        _MEAN_THRESH_x_SCORER = _MEAN_THRESH_x_SCORER.transpose()

        BEST_THRESH_IDXS = []
        for scorer_idx, scores  in enumerate(_MEAN_THRESH_x_SCORER):

            BEST_THRESH_IDXS.append(np.argmax(scores))


        assert np.array_equiv(out, BEST_THRESH_IDXS)




    def test_accuracy_partially_masked(self,
        _TEST_FOLD_x_THRESHOLD_x_SCORER__SCORE_partially_masked,
        _THRESHOLDS):

        out = _get_best_thresholds(
            _TEST_FOLD_x_THRESHOLD_x_SCORER__SCORE_partially_masked,
            _THRESHOLDS
        )


        assert isinstance(out, np.ma.masked_array)
        assert out.dtype == np.uint16
        assert len(out) == \
           _TEST_FOLD_x_THRESHOLD_x_SCORER__SCORE_partially_masked.shape[2]


        _MEAN_THRESH_x_SCORER = \
            np.mean(
                _TEST_FOLD_x_THRESHOLD_x_SCORER__SCORE_partially_masked,
                axis=0
            )

        _MEAN_THRESH_x_SCORER = _MEAN_THRESH_x_SCORER.transpose()

        BEST_THRESH_IDXS = []
        for scorer_idx, scores  in enumerate(_MEAN_THRESH_x_SCORER):

            BEST_THRESH_IDXS.append(np.argmax(scores))


        assert np.array_equiv(out, BEST_THRESH_IDXS)






    def test_accuracy_fully_masked(self,
        _TEST_FOLD_x_THRESHOLD_x_SCORER__SCORE_fully_masked,
        _THRESHOLDS
    ):

        out = _get_best_thresholds(
                _TEST_FOLD_x_THRESHOLD_x_SCORER__SCORE_fully_masked,
                _THRESHOLDS
        )


        assert isinstance(out, np.ma.masked_array)
        assert out.dtype == np.uint16
        assert len(out) == \
            _TEST_FOLD_x_THRESHOLD_x_SCORER__SCORE_fully_masked.shape[2]


        _MEAN_THRESH_x_SCORER = \
            np.mean(
                _TEST_FOLD_x_THRESHOLD_x_SCORER__SCORE_fully_masked,
                axis=0
            )

        _MEAN_THRESH_x_SCORER = _MEAN_THRESH_x_SCORER.transpose()

        BEST_THRESH_IDXS = []
        for scorer_idx, scores  in enumerate(_MEAN_THRESH_x_SCORER):

            BEST_THRESH_IDXS.append(np.argmax(scores))

        # this is bad, that masked array returns argmax as zero when
        # a layer is completely masked. so must catch if all fits excepted
        # immediately following fits.
        assert np.array_equiv(out, [0,0,0,0])

        assert np.array_equiv(out, BEST_THRESH_IDXS)











