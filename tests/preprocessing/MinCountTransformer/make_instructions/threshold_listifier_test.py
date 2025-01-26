# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.preprocessing.MinCountTransformer._make_instructions. \
    _threshold_listifier import _threshold_listifier

import numpy as np

import pytest



class TestThresholdListifier:


    # def _threshold_listifier(
    #     _n_features_in: int,
    #     *_threshold: Union[int, Iterable[int]]
    # ) -> Union[list[int], tuple[list[int], ...]]:


    @pytest.mark.parametrize('junk',
        (-2.7, -1, 0, 2.7, True, None, 'trash', {'a':1}, lambda x: x)
    )
    def test_rejects_junk(self, junk):

        # just catch anything dont worry about Type or Value Error
        with pytest.raises(Exception):
            _threshold_listifier(junk, 2)

        with pytest.raises(Exception):
            _threshold_listifier(2, junk)

        with pytest.raises(Exception):
            _threshold_listifier(2, junk, junk, junk, junk)


    def test_rejects_bad_threshold(self):

        n_features_in = 2

        # just catch anything dont worry about Type or Value Error
        # less than 2
        with pytest.raises(Exception):
            _threshold_listifier(n_features_in, 1)

        # empty
        with pytest.raises(Exception):
            _threshold_listifier(n_features_in, [])

        # too short
        with pytest.raises(Exception):
            _threshold_listifier(
                n_features_in, [_ for _ in range(n_features_in-1)]
            )

        # too long
        with pytest.raises(Exception):
            _threshold_listifier(
                n_features_in, [_ for _ in range(n_features_in+1)]
            )

        # non-int
        with pytest.raises(Exception):
            _threshold_listifier(
                n_features_in, list('abcdefghijk')[:n_features_in]
            )


    @pytest.mark.parametrize(
        'n_thresholds,threshold',
        (
            (1, 'int'),
            (1, 'list[int]'),
            (2, 'list[int]'),
            (3, 'list[int]'),
            (4, 'list[int]')
        )
    )
    @pytest.mark.parametrize('n_features_in', (2, 11, 37))
    def test_accepts_good_and_accuracy(
        self, n_thresholds, threshold, n_features_in
    ):

        if threshold == 'int':  # n_thresholds must == 1
            value = int(np.random.randint(2, 10))  # must be >= 2
            out = _threshold_listifier(n_features_in, value)

            assert isinstance(out, list)
            assert len(out) == n_features_in
            assert all(map(lambda x: x==value, out))
        else:
            THRESHOLD = []
            for _ in range(n_thresholds):
                # keep this 'value' thing for array_equal when only 1 list-likes
                value = list(np.random.randint(1, 10, n_features_in))
                THRESHOLD.append(value)
            out = _threshold_listifier(n_features_in, *THRESHOLD)

            if n_thresholds == 1:
                assert isinstance(out, list)
                assert len(out) == n_features_in
                assert all(map(isinstance, out, (int for _ in out)))
                assert np.array_equal(out, value)
            else:
                assert isinstance(out, tuple)
                assert len(out) == n_thresholds
                assert all(map(isinstance, out, (list for _ in out)))
                for idx, set_of_thresholds in enumerate(out):
                    assert all(map(
                        isinstance,
                        set_of_thresholds,
                        (int for _ in set_of_thresholds)
                    ))
                    assert np.array_equal(set_of_thresholds, THRESHOLD[idx])











