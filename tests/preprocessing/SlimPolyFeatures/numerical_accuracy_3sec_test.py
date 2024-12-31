# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



from pybear.preprocessing.SlimPolyFeatures.SlimPolyFeatures import \
    SlimPolyFeatures as SlimPoly

import numpy as np

from sklearn.preprocessing import OneHotEncoder as OHE

import pytest







class FixtureMixin:


    @staticmethod
    @pytest.fixture(scope='module')
    def _shape():
        return (9, 3)


    @staticmethod
    @pytest.fixture(scope='function')
    def _kwargs():

        return {
            'degree': 2,
            'min_degree': 1,
            'interaction_only': True,
            'scan_X': False,
            'keep': 'first',
            'sparse_output': False,
            'feature_name_combiner': 'as_indices',
            'equal_nan': True,
            'rtol': 1e-5,
            'atol': 1e-8,
            'n_jobs': 1
        }


    @staticmethod
    @pytest.fixture(scope='module')
    def _X_np(_X_factory, _shape):

        return _X_factory(
            _dupl=None,
            _format='np',
            _dtype='flt',
            _has_nan=False,
            _constants=None,
            _columns=None,
            _zeros=None,
            _shape=_shape
        )


    @staticmethod
    @pytest.fixture(scope='module')
    def _columns(_master_columns, _shape):
        return _master_columns.copy()[:_shape[1]]



class TestBasicCaseNoDuplsNoConstantsInPoly(FixtureMixin):

    @pytest.mark.parametrize('min_degree', (1,2))
    @pytest.mark.parametrize('intx_only', (True, False))
    def test_basic_case(
        self, _X_np, _columns, _kwargs, _shape, min_degree, intx_only
    ):

        _kwargs['min_degree'] = min_degree
        _kwargs['interaction_only'] = intx_only

        TestCls = SlimPoly(**_kwargs)
        out = TestCls.fit_transform(_X_np)

        if min_degree == 1:
            if intx_only:
                assert out.shape[1] == 6
                assert np.array_equal(out[:, 0], _X_np[:, 0])
                assert np.array_equal(out[:, 1], _X_np[:, 1])
                assert np.array_equal(out[:, 2], _X_np[:, 2])
                assert np.array_equal(out[:, 3], _X_np[:, (0, 1)].prod(1))
                assert np.array_equal(out[:, 4], _X_np[:, (0, 2)].prod(1))
                assert np.array_equal(out[:, 5], _X_np[:, (1, 2)].prod(1))
            elif not intx_only:
                assert out.shape[1] == 9
                assert np.array_equal(out[:, 0], _X_np[:, 0])
                assert np.array_equal(out[:, 1], _X_np[:, 1])
                assert np.array_equal(out[:, 2], _X_np[:, 2])
                assert np.array_equal(out[:, 3], _X_np[:, (0, 0)].prod(1))
                assert np.array_equal(out[:, 4], _X_np[:, (0, 1)].prod(1))
                assert np.array_equal(out[:, 5], _X_np[:, (0, 2)].prod(1))
                assert np.array_equal(out[:, 6], _X_np[:, (1, 1)].prod(1))
                assert np.array_equal(out[:, 7], _X_np[:, (1, 2)].prod(1))
                assert np.array_equal(out[:, 8], _X_np[:, (2, 2)].prod(1))

        elif min_degree == 2:
            if intx_only:
                assert out.shape[1] == 3
                assert np.array_equal(out[:, 0], _X_np[:, (0, 1)].prod(1))
                assert np.array_equal(out[:, 1], _X_np[:, (0, 2)].prod(1))
                assert np.array_equal(out[:, 2], _X_np[:, (1, 2)].prod(1))
            elif not intx_only:
                assert out.shape[1] == 6
                assert np.array_equal(out[:, 0], _X_np[:, (0, 0)].prod(1))
                assert np.array_equal(out[:, 1], _X_np[:, (0, 1)].prod(1))
                assert np.array_equal(out[:, 2], _X_np[:, (0, 2)].prod(1))
                assert np.array_equal(out[:, 3], _X_np[:, (1, 1)].prod(1))
                assert np.array_equal(out[:, 4], _X_np[:, (1, 2)].prod(1))
                assert np.array_equal(out[:, 5], _X_np[:, (2, 2)].prod(1))


class TestRiggedCasePolyHasConstantsAndDupls(FixtureMixin):

    # A POLY EXPANSION ON A ONE HOT ENCODED COLUMN, ALL INTERACTION FEATURES
    # ARE COLUMNS OF ZEROS OR DUPLICATE

    # ALSO TEST MULTIPLE PARTIAL FITS & TRANSFORMS

    @pytest.mark.parametrize('min_degree', (1,2))
    @pytest.mark.parametrize('intx_only', (True, False))
    def test_constant_and_dupl(
        self, _kwargs, _shape, min_degree, intx_only
    ):

        _kwargs['degree'] = 2
        _kwargs['min_degree'] = min_degree
        _kwargs['interaction_only'] = intx_only

        TestCls = SlimPoly(**_kwargs)

        partial_fits = 5
        BATCH_XS = []
        for batch in range(partial_fits):
            while True:
                _X = np.random.choice(
                    list('abc'), _shape[0], replace=True
                ).reshape((-1,1))
                if len(np.unique(_X.ravel())) == 3:
                    BATCH_XS.append(_X)
                    break

        assert len(BATCH_XS) == partial_fits

        _ohe = OHE(sparse_output=False)
        for batch_idx, batch in enumerate(BATCH_XS):
            BATCH_XS[batch_idx] = _ohe.fit_transform(batch)

        for _X_np in BATCH_XS:

            out = TestCls.fit_transform(_X_np)

            if min_degree == 1:
                if intx_only:
                    assert out.shape[1] == 3
                    assert np.array_equal(out[:, 0], _X_np[:, 0])
                    assert np.array_equal(out[:, 1], _X_np[:, 1])
                    assert np.array_equal(out[:, 2], _X_np[:, 2])
                elif not intx_only:
                    assert out.shape[1] == 3
                    assert np.array_equal(out[:, 0], _X_np[:, 0])
                    assert np.array_equal(out[:, 1], _X_np[:, 1])
                    assert np.array_equal(out[:, 2], _X_np[:, 2])
            elif min_degree == 2:
                if intx_only:
                    assert isinstance(out, np.ndarray)
                    assert out.shape[1] == 0
                elif not intx_only:
                    assert isinstance(out, np.ndarray)
                    assert out.shape[1] == 0


class TestRiggedCaseAllIntxAreDupl(FixtureMixin):

    @pytest.mark.parametrize('min_degree', (1,2))
    @pytest.mark.parametrize('intx_only', (True, False))
    def test_dupls(
        self, _X_factory, _columns, _kwargs, _shape, min_degree, intx_only
    ):

        _kwargs['degree'] = 2
        _kwargs['min_degree'] = min_degree
        _kwargs['interaction_only'] = intx_only


        # with this rigging:
        # - all squared columns != original column.
        # - all interactions equal the same thing, so there should be
        #       only one feature in poly.
        _X_np = np.array(
            [
                [2, 0, 0],
                [2, 0, 0],
                [2, 0, 0],
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 1],
                [0, 0, 2],
                [0, 2, 0],
                [0, 2, 0]
            ],
            dtype=np.uint8
        )

        TestCls = SlimPoly(**_kwargs)
        out = TestCls.fit_transform(_X_np)

        if min_degree == 1:
            if intx_only:
                assert out.shape[1] == 4
                assert np.array_equal(out[:, 0], _X_np[:, 0])
                assert np.array_equal(out[:, 1], _X_np[:, 1])
                assert np.array_equal(out[:, 2], _X_np[:, 2])
                assert np.array_equal(out[:, 3], _X_np[:, (0, 1)].prod(1))
            elif not intx_only:
                assert out.shape[1] == 7
                assert np.array_equal(out[:, 0], _X_np[:, 0])
                assert np.array_equal(out[:, 1], _X_np[:, 1])
                assert np.array_equal(out[:, 2], _X_np[:, 2])
                assert np.array_equal(out[:, 3], _X_np[:, (0, 0)].prod(1))
                assert np.array_equal(out[:, 4], _X_np[:, (0, 1)].prod(1))
                assert np.array_equal(out[:, 5], _X_np[:, (1, 1)].prod(1))
                assert np.array_equal(out[:, 6], _X_np[:, (2, 2)].prod(1))

        elif min_degree == 2:
            if intx_only:
                assert out.shape[1] == 1
                assert np.array_equal(out[:, 0], _X_np[:, (0, 1)].prod(1))
            elif not intx_only:
                assert out.shape[1] == 4
                assert np.array_equal(out[:, 0], _X_np[:, (0, 0)].prod(1))
                assert np.array_equal(out[:, 1], _X_np[:, (0, 1)].prod(1))
                assert np.array_equal(out[:, 2], _X_np[:, (1, 1)].prod(1))
                assert np.array_equal(out[:, 3], _X_np[:, (2, 2)].prod(1))


class TestOneColumnX(FixtureMixin):

    @pytest.mark.parametrize('min_degree', (1,3))
    def test_one_column(self, _kwargs, _shape, min_degree):

        # interaction_only must always be False for single column

        _kwargs['degree'] = 3
        _kwargs['min_degree'] = min_degree
        _kwargs['interaction_only'] = False

        TestCls = SlimPoly(**_kwargs)

        _X_np = np.random.uniform(0, 1, (_shape[0], 1))

        out = TestCls.fit_transform(_X_np)

        if min_degree == 1:
            assert isinstance(out, np.ndarray)
            assert out.shape[1] == 3
            assert np.array_equal(out[:, 0], _X_np[:, 0])
            assert np.array_equal(out[:, 1], _X_np[:, (0, 0)].prod(1))
            assert np.array_equal(out[:, 2], _X_np[:, (0, 0, 0)].prod(1))
        elif min_degree == 2:
            assert isinstance(out, np.ndarray)
            assert out.shape[1] == 2
            assert np.array_equal(out[:, 0], _X_np[:, (0, 0)].prod(1))
            assert np.array_equal(out[:, 1], _X_np[:, (0, 0, 0)].prod(1))
        elif min_degree == 3:
            assert out.shape[1] == 1
            assert np.array_equal(out[:, 0], _X_np[:, (0, 0, 0)].prod(1))








