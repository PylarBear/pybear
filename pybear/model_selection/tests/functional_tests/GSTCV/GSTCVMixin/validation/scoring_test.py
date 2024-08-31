# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#


import pytest

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)

from model_selection.GSTCV._GSTCVMixin._validation._scoring import \
    _validate_scoring



class TestValidateScoring:


    @pytest.mark.parametrize('junk_scoring',
        (0, 1, True, False, None, np.nan)
    )
    def test_rejects_anything_not_str_callable_dict_iterable(self, junk_scoring):

        with pytest.raises(TypeError):
            _validate_scoring(junk_scoring)


    @pytest.mark.parametrize('junk_scoring',
        ('junk', 'garbage', 'trash', 'rubbish', 'waste', 'refuse')
    )
    def test_rejects_bad_strs(self, junk_scoring):

        with pytest.raises(ValueError):
            _validate_scoring(junk_scoring)


    @pytest.mark.parametrize('good_scoring',
        ('accuracy', 'balanced_accuracy', 'precision', 'recall')
    )
    def test_accepts_good_strs(self, good_scoring):

        out = _validate_scoring(good_scoring)
        assert isinstance(out, dict)
        assert len(out) == 1
        assert good_scoring in out
        assert callable(out[good_scoring])
        assert float(out[good_scoring]([1, 0, 1, 1], [1, 0, 0, 1]))


    @pytest.mark.parametrize('junk_scoring',
        (lambda x: 'junk', lambda x: [0,1], lambda x,y: min, lambda x,y: x)
    )
    def test_rejects_non_num_callables(self, junk_scoring):

        with pytest.raises(ValueError):
            _validate_scoring(junk_scoring)


    def test_accepts_good_callable(self):

        good_callable = lambda y1, y2: np.sum(np.array(y2)-np.array(y1))

        out = _validate_scoring(good_callable)
        assert isinstance(out, dict)
        assert len(out) == 1
        assert 'score' in out
        assert callable(out['score'])
        assert float(out['score']([1, 0, 1, 1], [1, 0, 0, 1]))


    @pytest.mark.parametrize('junk_scoring', ([], (), {}))
    def test_rejects_empty(self, junk_scoring):

        with pytest.raises(ValueError):
            _validate_scoring(junk_scoring)


    @pytest.mark.parametrize('junk_lists',
        ([1,2,3], ('a','b','c'), {0,1,2}, ['trash', 'garbage', 'junk'])
    )
    def test_rejects_junk_lists(self, junk_lists):

        with pytest.raises((TypeError, ValueError)):
            _validate_scoring(junk_lists)


    @pytest.mark.parametrize('good_lists',
        (['precision', 'recall'], ('accuracy','balanced_accuracy'),
         {'f1', 'balanced_accuracy', 'recall', 'precision'})
    )
    def test_accepts_good_lists(self, good_lists):
        out = _validate_scoring(good_lists)
        assert isinstance(out, dict)
        assert len(out) == len(good_lists)
        for metric in good_lists:
            assert metric in out
            assert callable(out[metric])
            assert float(out[metric]([1,0,1,1], [1,0,0,1]))


    @pytest.mark.parametrize('junk_dicts',
        ({'a':1, 'b':2}, {0:1, 1:2}, {0:[1,2,3], 1:[2,3,4]},
         {'metric1': lambda y1, y2: 'trash', 'metric2': lambda x: 1})
    )
    def test_rejects_junk_dicts(self, junk_dicts):

        with pytest.raises(ValueError):
            _validate_scoring(junk_dicts)



    @pytest.mark.parametrize('good_dict',
        ({'accuracy': accuracy_score, 'f1': f1_score, 'recall': recall_score},
         {'metric1': precision_score, 'metric2': recall_score})
    )
    def test_accepts_good_dicts(self, good_dict):

        out = _validate_scoring(good_dict)
        assert isinstance(out, dict)
        assert len(out) == len(good_dict)
        for metric in good_dict:
            assert metric in out
            assert callable(out[metric])
            assert float(out[metric]([0,1,0,1],[1,0,0,1]))




















